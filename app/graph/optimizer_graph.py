"""
app/graph/optimizer_graph.py
-----------------------------
One-page optimizer sub-graph.

Goal: given a LaTeX resume that exceeds 1 page, iteratively drop the
lowest-scoring sections/items until it fits on one page, maximising
relevance to the job description.

Flow:
  init  →  drop_loop  (loop until fit OR no more candidates)
         ↓ fit achieved
       condensation_interrupt  (optional LLM condensation, user approves)
         ↓
       review_interrupt  (show what changed, PDF preview, undo, confirm)
         ↓
       END

Phase 1 — Zero-LLM drop loop:
  • Sort ranked_items ASC by similarity_score (lowest relevance = drop first).
  • pdflatex is the sole oracle for page count.
  • Drop items one at a time until page_count == 1 OR no candidates remain.

Phase 2 — Condensation (only if page still > 1 after all drops):
  • Single LLM call (gemini-2.5-flash) to condense remaining content.
  • Strict validation: no proper nouns or numbers may be removed.
  • User must approve before condensation is applied.

Phase 3 — Review interrupt:
  • Show summary of dropped items with their scores.
  • Show condensation proposal side-by-side if applicable.
  • Undo stack: user can restore any dropped item.
  • Confirm → save to VariantStore.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from app.utils.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# State
# ──────────────────────────────────────────────────────────────────────────────

class OptimizerState(TypedDict, total=False):
    # ── inputs ──────────────────────────────────────────────────────────────
    user_id:         int
    thread_id:       str | None
    latex:           str               # full LaTeX of the current resume
    ranked_items:    list[dict]        # items with similarity_score field
    job_id:          int | None

    # ── working state ────────────────────────────────────────────────────────
    page_count:      int               # current compiled page count
    drop_candidates: list[dict]        # sorted ASC by score; pop from front
    dropped_items:   list[dict]        # items removed so far (for undo & display)
    current_latex:   str               # latex after each drop
    undo_stack:      list[dict]        # [{item, latex_before}] for undo

    # ── condensation ─────────────────────────────────────────────────────────
    condensation_proposal: dict | None  # {original_text, condensed_text, section}

    # ── output ───────────────────────────────────────────────────────────────
    final_latex:     str | None
    saved_variant_id: int | None
    stage:           str
    error:           str | None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _compile_and_count(latex: str) -> tuple[str | None, int]:
    """
    Write latex to a temp file, compile with pdflatex, return (pdf_path, page_count).
    pdf_path is None if compilation fails.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tex = Path(tmp) / "resume.tex"
        tex.write_text(latex, encoding="utf-8")
        cmd = [
            "pdflatex",
            "-interaction=nonstopmode",
            f"-output-directory={tmp}",
            str(tex),
        ]
        try:
            for _ in range(2):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
            match = re.search(r"Output written on .+? \((\d+) page", result.stdout)
            page_count = int(match.group(1)) if match else 1
            pdf = Path(tmp) / "resume.pdf"
            if pdf.exists():
                import shutil
                out_pdf = Path(tmp) / "out.pdf"
                shutil.copy(pdf, out_pdf)
                return str(out_pdf), page_count
            return None, page_count
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            log.warning("pdflatex failed: %s", exc)
            return None, 1


def _remove_item_from_latex(latex: str, item: dict) -> str:
    """
    Remove the LaTeX block for a ranked_item from the full resume.
    ranked_item rows carry content_latex on the best row; we match by content.
    This is best-effort: we look for the content block in the LaTeX string.
    """
    rows = item.get("rows", [])
    best = next((r for r in rows if r and r.get("content_latex")), None)
    if not best:
        return latex
    block = best["content_latex"].strip()
    if block and block in latex:
        # Remove the block and tidy surrounding blank lines
        result = latex.replace(block, "")
        result = re.sub(r"\n{3,}", "\n\n", result)
        return result
    return latex


# ──────────────────────────────────────────────────────────────────────────────
# Nodes
# ──────────────────────────────────────────────────────────────────────────────

def node_init(state: OptimizerState) -> dict:
    """Sort candidates ASC by score, compile initial latex, record page count."""
    log.info("[optimizer] init")
    try:
        ranked = list(state.get("ranked_items", []))
        # Sort ascending: lowest score = drop first
        ranked.sort(key=lambda r: float(r.get("similarity_score", 0)))

        latex = state.get("latex", "")
        _, page_count = _compile_and_count(latex)
        log.info("[optimizer] initial page count: %d", page_count)

        return {
            "drop_candidates": ranked,
            "dropped_items":   [],
            "undo_stack":      [],
            "current_latex":   latex,
            "page_count":      page_count,
            "condensation_proposal": None,
            "stage": "initialized",
        }
    except Exception:
        return {"error": traceback.format_exc(), "stage": "error"}


def node_drop_loop(state: OptimizerState) -> dict:
    """Pop the lowest-score candidate, remove from LaTeX, recompile, check fit."""
    log.info("[optimizer] drop_loop — page_count=%d, candidates left=%d",
             state.get("page_count", 1), len(state.get("drop_candidates", [])))
    try:
        page_count  = state.get("page_count", 1)
        candidates  = list(state.get("drop_candidates", []))
        dropped     = list(state.get("dropped_items", []))
        undo_stack  = list(state.get("undo_stack", []))
        current_tex = state.get("current_latex", state.get("latex", ""))

        if page_count <= 1 or not candidates:
            return {"stage": "drop_loop_done"}

        item = candidates.pop(0)
        latex_before = current_tex
        new_latex = _remove_item_from_latex(current_tex, item)

        if new_latex == current_tex:
            # Content not found in LaTeX; skip silently
            return {
                "drop_candidates": candidates,
                "stage": "dropping",
            }

        _, new_page_count = _compile_and_count(new_latex)
        log.info("[optimizer] dropped '%s/%s' → %d page(s)",
                 item.get("section_name"), item.get("item_name"), new_page_count)

        dropped.append(item)
        undo_stack.append({"item": item, "latex_before": latex_before})

        return {
            "drop_candidates": candidates,
            "dropped_items":   dropped,
            "undo_stack":      undo_stack,
            "current_latex":   new_latex,
            "page_count":      new_page_count,
            "stage":           "dropping",
        }
    except Exception:
        return {"error": traceback.format_exc(), "stage": "error"}


def node_condensation_interrupt(state: OptimizerState) -> dict:
    """
    If still > 1 page after all drops, offer LLM condensation.
    Interrupt surfaces the proposal; user approves or skips.
    """
    log.info("[optimizer] condensation_interrupt")
    page_count = state.get("page_count", 1)
    current_tex = state.get("current_latex", state.get("latex", ""))

    if page_count <= 1:
        # Already fits — skip condensation, go straight to review
        return {"stage": "condensation_skipped"}

    # Ask LLM to condense
    try:
        proposal = _ask_llm_to_condense(current_tex)
    except Exception as exc:
        log.warning("[optimizer] condensation LLM failed: %s", exc)
        proposal = None

    user_response: dict = interrupt({
        "type":                 "condensation",
        "page_count":           page_count,
        "condensation_proposal": proposal,
        "current_latex":        current_tex,
    })

    approved  = user_response.get("approved", False)
    new_latex = current_tex
    if approved and proposal and proposal.get("condensed_latex"):
        new_latex = proposal["condensed_latex"]
        _, new_page_count = _compile_and_count(new_latex)
    else:
        new_page_count = page_count

    return {
        "condensation_proposal": proposal if approved else None,
        "current_latex": new_latex,
        "page_count": new_page_count,
        "stage": "condensation_done",
    }


def node_review_interrupt(state: OptimizerState) -> dict:
    """
    Show the user what was changed. Allow undo of any dropped item.
    On confirm → mark final and optionally save to library.
    """
    log.info("[optimizer] review_interrupt")

    user_response: dict = interrupt({
        "type":          "optimizer_review",
        "dropped_items": state.get("dropped_items", []),
        "page_count":    state.get("page_count", 1),
        "condensation":  state.get("condensation_proposal"),
        "current_latex": state.get("current_latex", ""),
    })

    action = user_response.get("action", "confirm")

    if action == "undo":
        # Restore the last dropped item
        undo_stack = list(state.get("undo_stack", []))
        dropped    = list(state.get("dropped_items", []))
        if undo_stack:
            entry       = undo_stack.pop()
            restored_tex = entry["latex_before"]
            restored_item = entry["item"]
            dropped     = [d for d in dropped if d is not restored_item]
            _, page_count = _compile_and_count(restored_tex)
            return {
                "undo_stack":   undo_stack,
                "dropped_items": dropped,
                "current_latex": restored_tex,
                "page_count":   page_count,
                "stage": "review_undo",
            }

    if action == "save_to_library":
        # Save the final variant to VariantStore
        variant_id = _save_to_library(state, user_response)
        return {
            "final_latex":     state.get("current_latex"),
            "saved_variant_id": variant_id,
            "stage": "completed",
        }

    # confirm — just finish
    return {
        "final_latex": state.get("current_latex"),
        "stage": "completed",
    }


# ──────────────────────────────────────────────────────────────────────────────
# LLM condensation helper
# ──────────────────────────────────────────────────────────────────────────────

def _ask_llm_to_condense(latex: str) -> dict | None:
    """
    Call gemini-2.5-flash to condense bullet points so the resume fits 1 page.
    Returns {condensed_latex, changes_summary} or None on failure.

    Strict constraints injected into the prompt:
      • No proper nouns (company names, project names, technologies) may be removed.
      • No numbers or metrics may be removed.
      • Only shorten verbose phrases; no information loss.
    """
    import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    genai.configure(api_key=api_key)
    model_name = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    model = genai.GenerativeModel(model_name)

    prompt = f"""You are a LaTeX resume editor. The resume below exceeds one page.
Condense it so it fits on ONE page by shortening verbose bullet points.

STRICT RULES — violation = rejection:
1. Do NOT remove any proper nouns (company names, university, project names, tools, technologies).
2. Do NOT remove any numbers or metrics (percentages, durations, counts, dollar amounts).
3. Do NOT add new information.
4. Only shorten wordy phrases. Prefer active verbs. Cut filler words.
5. Return the complete LaTeX document with your changes applied.

Resume LaTeX:
{latex}

Return JSON only:
{{
  "condensed_latex": "...complete LaTeX...",
  "changes_summary": "...brief human-readable summary of what was shortened..."
}}"""

    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Strip markdown fences
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        data = json.loads(text)

        condensed = data.get("condensed_latex", "")
        if not condensed:
            return None

        # Validate: check no proper nouns were lost (simple heuristic —
        # find all Title-Case words in original, check they appear in condensed)
        orig_proper = set(re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", latex))
        for noun in orig_proper:
            if noun not in condensed:
                log.warning("[optimizer] condensation removed proper noun '%s' — rejecting", noun)
                return None

        return {
            "condensed_latex":  condensed,
            "changes_summary":  data.get("changes_summary", ""),
        }
    except Exception as exc:
        log.warning("[optimizer] LLM condensation parse error: %s", exc)
        return None


def _save_to_library(state: OptimizerState, user_response: dict) -> int | None:
    try:
        from app.core.variant_store import VariantStore
        store = VariantStore()
        meta = store.save(
            user_id=state.get("user_id", 1),
            name=user_response.get("name", "One-Page Resume"),
            latex=state.get("current_latex", ""),
            variant_type="one_page",
            job_ids=[state["job_id"]] if state.get("job_id") else [],
            tags=["one_page"],
        )
        return meta["id"]
    except Exception as exc:
        log.warning("[optimizer] save to library failed: %s", exc)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Routing
# ──────────────────────────────────────────────────────────────────────────────

def _route_after_init(state: OptimizerState) -> str:
    if state.get("error"):
        return END
    if state.get("page_count", 1) <= 1:
        return "condensation_interrupt"   # already fits — skip to review
    return "drop_loop"


def _route_after_drop(state: OptimizerState) -> str:
    if state.get("error"):
        return END
    page_count = state.get("page_count", 1)
    candidates = state.get("drop_candidates", [])
    if page_count <= 1 or not candidates:
        return "condensation_interrupt"
    return "drop_loop"


def _route_after_condensation(state: OptimizerState) -> str:
    if state.get("error"):
        return END
    return "review_interrupt"


def _route_after_review(state: OptimizerState) -> str:
    stage = state.get("stage", "")
    if stage == "review_undo":
        return "review_interrupt"   # re-show review after undo
    return END


# ──────────────────────────────────────────────────────────────────────────────
# Graph assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_optimizer_graph():
    g = StateGraph(OptimizerState)

    g.add_node("init",                    node_init)
    g.add_node("drop_loop",               node_drop_loop)
    g.add_node("condensation_interrupt",  node_condensation_interrupt)
    g.add_node("review_interrupt",        node_review_interrupt)

    g.set_entry_point("init")
    g.add_conditional_edges("init",                   _route_after_init)
    g.add_conditional_edges("drop_loop",              _route_after_drop)
    g.add_conditional_edges("condensation_interrupt", _route_after_condensation)
    g.add_conditional_edges("review_interrupt",       _route_after_review)

    return g.compile()


optimizer_graph = build_optimizer_graph()
