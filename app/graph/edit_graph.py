# app/graph/edit_graph.py
#
# LangGraph StateGraph for the interactive resume-editing workflow.
#
# Full flow
# ---------
#   parse_jd  →  parse_resume  →  retrieve
#     →  [INTERRUPT: item_selection]
#   build_edit_state  →  generate_suggestions
#     →  [INTERRUPT: section_review]  (loops per section/item)
#   apply_edit  ──► more sections? ──► [INTERRUPT: section_review]
#                └─ done ──► score_resume
#                              └─ [INTERRUPT: refine_or_finish]
#                                    ├─ "refine" → generate_suggestions (with score feedback)
#                                    └─ "finish" → END
#
# Human-in-the-loop pattern
# -------------------------
#   • The graph calls `interrupt(payload)` to pause and surface data
#     to the caller.
#   • The caller resumes with `Command(resume=user_response)`.
#   • All graph state is serialisable so the MemorySaver checkpointer
#     can persist it across HTTP requests.

from __future__ import annotations

import traceback
import concurrent.futures
from typing import Any

from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import EditGraphState
from app.models.jd import ParsedJD
from app.models.edit import ResumeEditCycle, ResumeEditState
from app.core.retriever import FLAT_SECTIONS, ATOMIC_SECTIONS
import app.core.pipeline as pl
from app.utils import get_logger

log = get_logger(__name__)


# ======================================================================
# Helper: rebuild EditAgent from serialised state
# ======================================================================

def _rebuild_agent(state: EditGraphState):
    """
    Reconstruct an EditAgent from the serialised edit_cycle_dict stored
    in the graph state.  The agent is ephemeral — it is never stored in
    state itself; only the cycle dict is persisted.
    """
    from app.agents.edit_agent import EditAgent

    cycle_dict = state.get("edit_cycle_dict")
    parsed_jd  = ParsedJD(**state["parsed_jd"])

    if cycle_dict is None:
        raise ValueError("edit_cycle_dict is not in state — call build_edit_state first")

    cycle = ResumeEditCycle.model_validate(cycle_dict)

    # EditAgent.__init__ calls init_root which would reset the cycle.
    # We bypass that by constructing the agent and wiring the cycle directly.
    agent = object.__new__(EditAgent)
    agent.editing_cycle     = cycle
    agent.section_name      = ""
    agent.item_name         = ""
    agent.special_instruction = ""

    from app.utils.llm import llm as _llm
    from app.utils import SQLHandler
    agent.llmhandler = _llm
    agent.db         = SQLHandler()

    return agent


def _serialise_cycle(agent) -> dict:
    return agent.editing_cycle.model_dump()


# ======================================================================
# Node: parse_and_retrieve  (replaces parse_jd + parse_resume + retrieve)
#
# JD and resume parsing run in parallel via ThreadPoolExecutor, then the
# retriever searches ALL sections in ChromaDB by JD-embedding similarity
# (not filtered to any specific resume_id) so the best version of each
# section-item is surfaced regardless of which editing session produced it.
# ======================================================================

def node_parse_and_retrieve(state: EditGraphState) -> dict:
    """
    Phase 1 — parse JD + load/parse resume in parallel.
    Phase 2 — retrieve best section versions scoped to the user's own resumes.

    Resume source priority:
      1. resume_path   → newly uploaded file (parse + store, then retrieve only that resume)
      2. existing_resume_id → specific DB resume chosen for personal info; retrieve across
                              ALL resumes belonging to that resume's user_id
      3. neither        → user-profile mode: load most-recent resume for personal info and
                          retrieve across all resumes for that user_id
    """
    log.info("[graph] parse_and_retrieve (JD + resume in parallel)")
    try:
        existing_id = state.get("existing_resume_id")
        resume_path = state.get("resume_path")
        user_id     = state["user_id"]

        # ── Phase 1: parse JD and resume concurrently ──────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jd_fut = pool.submit(pl.parse_jd, state["jd_text"], user_id)

            if resume_path:
                res_fut = pool.submit(pl.parse_resume, resume_path, user_id)
            elif existing_id:
                log.info("[graph] loading existing resume from DB: id=%s", existing_id)
                res_fut = pool.submit(pl.load_resume_from_db, existing_id)
            else:
                log.info("[graph] user-profile mode: loading latest resume for user_id=%s", user_id)
                res_fut = pool.submit(pl.load_latest_resume_for_user, user_id)

            job_id, parsed_jd        = jd_fut.result()
            resume_id, parsed_resume = res_fut.result()

        # ── Phase 2: retrieve, scoped to this user's resumes ───────────
        # Get all resume IDs for this user so retrieval picks the most
        # JD-relevant version of each section-item from their history.
        user_resume_ids = pl.get_user_resume_ids(user_id)
        log.info("[graph] user_id=%s has %d resume(s): %s", user_id, len(user_resume_ids), user_resume_ids)

        raw_results = pl.retrieve(
            job_id=job_id,
            parsed_jd=parsed_jd,
            resume_ids=user_resume_ids if user_resume_ids else None,
        )
        ranked = pl.rank_and_filter(raw_results)

        return {
            "job_id":        job_id,
            "parsed_jd":     parsed_jd.model_dump(),
            "resume_id":     resume_id,
            "parsed_resume": parsed_resume.model_dump(),
            "ranked_items":  ranked,
            "stage":         "retrieved",
        }
    except Exception as exc:
        log.error("[graph] parse_and_retrieve failed: %s", exc)
        return {"error": traceback.format_exc(), "stage": "error"}


# ======================================================================
# Node: item_selection  (INTERRUPT)
# ======================================================================

def node_item_selection(state: EditGraphState) -> dict:
    """
    Pause and surface the ranked_items to the user.
    The user sends back a list of items they want included.

    Interrupt payload:
        {
            "type": "item_selection",
            "ranked_items": [...],          # full list from retriever
        }

    Expected resume value (from client):
        {
            "selected_items": [...]         # subset of ranked_items
        }
    """
    user_response: dict = interrupt({
        "type":         "item_selection",
        "ranked_items": state["ranked_items"],
    })
    return {
        "selected_items": user_response.get("selected_items", state["ranked_items"]),
        "stage": "items_selected",
    }


# ======================================================================
# Node: build_edit_state
# ======================================================================

def node_build_edit_state(state: EditGraphState) -> dict:
    log.info("[graph] build_edit_state")
    try:
        parsed_jd   = ParsedJD(**state["parsed_jd"])
        edit_state  = pl.build_starting_edit_state(state["selected_items"])
        cycle_id    = state.get("cycle_id", 1)
        agent       = pl.build_edit_agent(edit_state, parsed_jd, cycle_id)

        # Build the sections_pending queue from the selected items
        pending: list[dict] = []
        for entry in state["selected_items"]:
            pending.append({
                "section": entry["section_name"],
                "item":    entry["item_name"],
            })

        return {
            "edit_cycle_dict": _serialise_cycle(agent),
            "sections_pending": pending,
            "stage": "edit_state_built",
        }
    except Exception as exc:
        log.error("[graph] build_edit_state failed: %s", exc)
        return {"error": traceback.format_exc(), "stage": "error"}


# ======================================================================
# Node: generate_suggestions
# ======================================================================

def node_generate_suggestions(state: EditGraphState) -> dict:
    log.info("[graph] generate_suggestions")
    try:
        agent    = _rebuild_agent(state)
        feedback = state.get("score_feedback")

        if feedback:
            pl.generate_suggestions_with_score_feedback(agent, feedback)
        else:
            pl.generate_suggestions(agent)

        # Always rebuild sections_pending from selected_items so that refine
        # cycles start the review loop from scratch.
        pending: list[dict] = [
            {"section": entry["section_name"], "item": entry["item_name"]}
            for entry in state.get("selected_items", [])
        ]

        return {
            "edit_cycle_dict":   _serialise_cycle(agent),
            "sections_pending":  pending,
            "sections_reviewed": [],
            "stage": "suggestions_ready",
        }
    except Exception as exc:
        log.error("[graph] generate_suggestions failed: %s", exc)
        return {"error": traceback.format_exc(), "stage": "error"}


# ======================================================================
# Node: section_review  (INTERRUPT)
# ======================================================================

def node_section_review(state: EditGraphState) -> dict:
    """
    Pop the next section from sections_pending, surface the current LaTeX
    and any LLM suggestion to the user.

    Interrupt payload:
        {
            "type":    "section_review",
            "section": str,
            "item":    str | None,
            "current_latex": str,
        }

    Expected resume value (from client):
        {
            "proposal":            "accept" | "reject" | "paraphrase" | "another_suggestion",
            "special_instruction": str | None,
            "rejection_reason":    str | None,
        }
    """
    pending  = list(state.get("sections_pending", []))
    reviewed = list(state.get("sections_reviewed", []))

    if not pending:
        return {"stage": "review_done"}

    current = pending[0]
    agent   = _rebuild_agent(state)

    # Fetch current LaTeX and LLM suggestions for this section/item
    sec_state = agent.editing_cycle.current_state
    if current["item"] is None:
        sec_obj = sec_state.get_section(current["section"])
        current_latex     = sec_obj.updated_section   if sec_obj else ""
        lines_to_change   = sec_obj.lines_to_change   if sec_obj else []
        suggested_changes = sec_obj.suggested_changes if sec_obj else []
    else:
        _, item_obj = sec_state.get_item(current["section"], current["item"])
        current_latex     = item_obj.updated_section   if item_obj else ""
        lines_to_change   = item_obj.lines_to_change   if item_obj else []
        suggested_changes = item_obj.suggested_changes if item_obj else []

    # Build per-section history for the UI (root → current node, deduplicated by content)
    history: list[dict] = []
    seen_content: set = set()
    for node in agent.editing_cycle.history():
        if current["item"] is None:
            sec_obj = node.resume_state.get_section(current["section"])
            content = sec_obj.updated_section if sec_obj else None
        else:
            _, item_obj = node.resume_state.get_item(current["section"], current["item"] or "")
            content = item_obj.updated_section if item_obj else None
        if content is not None and content not in seen_content:
            seen_content.add(content)
            history.append({
                "node_id": node.node_id,
                "action":  node.action or "root",
                "content": content,
            })

    user_response: dict = interrupt({
        "type":              "section_review",
        "section":           current["section"],
        "item":              current["item"],
        "current_latex":     current_latex,
        "lines_to_change":   lines_to_change,
        "suggested_changes": suggested_changes,
        "can_go_back":       len(reviewed) > 0,
        "section_index":     len(reviewed),          # 0-based index in queue
        "total_sections":    len(reviewed) + len(pending),
        "history":           history,
    })

    # NOTE: do NOT append current to sections_reviewed here.
    # node_apply_edit does that after the user's response is known,
    # so that "back" can pop the correct previous section.
    return {
        "current_review":    {**current, **user_response},
        "sections_pending":  pending[1:],   # pop reviewed item
        "sections_reviewed": reviewed,      # unchanged until apply_edit
        "stage": "applying_edit",
    }


# ======================================================================
# Node: apply_edit
# ======================================================================

def node_apply_edit(state: EditGraphState) -> dict:
    log.info("[graph] apply_edit")
    try:
        review   = state["current_review"]
        proposal = review.get("proposal", "accept")

        reviewed        = list(state.get("sections_reviewed", []))
        pending         = list(state.get("sections_pending", []))
        current_section = {"section": review["section"], "item": review.get("item")}

        # ── back-navigation ────────────────────────────────────────────────
        if proposal == "back":
            # sections_reviewed does NOT yet contain the current section
            # (it was not added in node_section_review), so reviewed[-1] is
            # the genuine previous section.
            if not reviewed:
                # Already at first section — just re-queue current
                return {
                    "sections_pending":  [current_section] + pending,
                    "sections_reviewed": [],
                    "stage": "edit_applied",
                }
            prev_section = reviewed[-1]
            return {
                "sections_pending":  [prev_section, current_section] + pending,
                "sections_reviewed": reviewed[:-1],
                "stage": "edit_applied",
            }

        # ── restore a specific historical version ──────────────────────────
        if proposal == "restore":
            node_id = int(review.get("node_id", 0))
            agent   = _rebuild_agent(state)
            cycle   = agent.editing_cycle
            sec_name  = review["section"]
            item_name = review.get("item")

            target_node = cycle.nodes.get(node_id)
            if target_node is None:
                return {"sections_reviewed": reviewed + [current_section], "stage": "edit_applied"}

            if item_name:
                _, target_item = target_node.resume_state.get_item(sec_name, item_name)
                if target_item is None:
                    return {"sections_reviewed": reviewed + [current_section], "stage": "edit_applied"}
                _, cur_item = cycle.current_state.get_item(sec_name, item_name)
                from app.models.edit import ItemEditState
                restored = ItemEditState(
                    section_name=sec_name,
                    item_name=item_name,
                    section_previous_state=cur_item.updated_section if cur_item else "",
                    lines_to_change=[],
                    suggested_changes=[],
                    updated_section=target_item.updated_section,
                )
                cur_items = list(getattr(cycle.current_state, sec_name) or [])
                new_items = [restored if (it and it.item_name == item_name) else it for it in cur_items]
                new_state = cycle.current_state.model_copy(update={sec_name: new_items})
            else:
                target_sec = target_node.resume_state.get_section(sec_name)
                if target_sec is None:
                    return {"sections_reviewed": reviewed + [current_section], "stage": "edit_applied"}
                cur_sec = cycle.current_state.get_section(sec_name)
                from app.models.edit import SectionEditState
                restored = SectionEditState(
                    section_name=sec_name,
                    section_previous_state=cur_sec.updated_section if cur_sec else "",
                    lines_to_change=[],
                    suggested_changes=[],
                    updated_section=target_sec.updated_section,
                )
                new_state = cycle.current_state.model_copy(update={sec_name: restored})

            cycle.push(new_state, action=f"restore_v{node_id}", section_name=sec_name)
            return {
                "edit_cycle_dict":   _serialise_cycle(agent),
                "sections_reviewed": reviewed + [current_section],
                "stage": "edit_applied",
            }

        # ── accept ────────────────────────────────────────────────────────
        if proposal == "accept":
            # Suggestion already in cycle — nothing to change.
            return {
                "sections_reviewed": reviewed + [current_section],
                "stage": "edit_applied",
            }

        # ── LLM-generating actions: paraphrase / custom_instruction / another_suggestion ─
        # "custom_instruction" is paraphrase with a special_instruction string;
        # map it so edit_agent receives a recognised proposal name.
        llm_proposal = "paraphrase" if proposal == "custom_instruction" else proposal

        agent = _rebuild_agent(state)
        pl.apply_edit(
            edit_agent=agent,
            section_name=review["section"],
            proposal=llm_proposal,
            item_name=review.get("item"),
            special_instruction=review.get("special_instruction"),
            rejection_reason=review.get("rejection_reason"),
        )

        # Re-queue the same section so the user can review the LLM result
        # before advancing.  Do NOT add to sections_reviewed yet.
        return {
            "edit_cycle_dict":  _serialise_cycle(agent),
            "sections_pending": [current_section] + pending,
            "sections_reviewed": reviewed,
            "stage": "edit_applied",
        }
    except Exception as exc:
        log.error("[graph] apply_edit failed: %s", exc)
        return {"error": traceback.format_exc(), "stage": "error"}


# ======================================================================
# Node: score_resume
# ======================================================================

def node_score_resume(state: EditGraphState) -> dict:
    log.info("[graph] score_resume")
    try:
        agent     = _rebuild_agent(state)
        parsed_jd = ParsedJD(**state["parsed_jd"])
        score     = pl.score_resume(
            edit_agent=agent,
            parsed_jd=parsed_jd,
            app_id=0,
            resume_id=state.get("resume_id", 0),
        )
        feedback = pl.format_score_feedback(score)
        return {
            "score_result":   score.model_dump(),
            "score_feedback": feedback,
            "stage":          "scored",
        }
    except Exception as exc:
        log.error("[graph] score_resume failed: %s", exc)
        return {"error": traceback.format_exc(), "stage": "error"}


# ======================================================================
# Node: refine_or_finish  (INTERRUPT)
# ======================================================================

def node_refine_or_finish(state: EditGraphState) -> dict:
    """
    Show the score to the user and ask whether they want another
    suggestion pass (informed by score feedback) or are done.

    Interrupt payload:
        {
            "type":           "refine_or_finish",
            "score_result":   dict,
            "score_feedback": str,
        }

    Expected resume value:
        {
            "choice": "refine" | "finish"
        }
    """
    user_response: dict = interrupt({
        "type":           "refine_or_finish",
        "score_result":   state.get("score_result"),
        "score_feedback": state.get("score_feedback"),
    })
    return {"stage": user_response.get("choice", "finish")}


# ======================================================================
# Conditional edges
# ======================================================================

def route_after_error(state: EditGraphState) -> str:
    if state.get("error"):
        return "error_end"
    return "continue"


def route_after_review_or_done(state: EditGraphState) -> str:
    """After apply_edit: more sections pending? Loop; otherwise score."""
    if state.get("sections_pending"):
        return "more_sections"
    return "all_done"


def route_after_refine_or_finish(state: EditGraphState) -> str:
    return "refine" if state.get("stage") == "refine" else "finish"


# ======================================================================
# Build the graph
# ======================================================================

def build_edit_graph() -> StateGraph:
    g = StateGraph(EditGraphState)

    # Add nodes
    g.add_node("parse_and_retrieve",   node_parse_and_retrieve)
    g.add_node("item_selection",       node_item_selection)
    g.add_node("build_edit_state",     node_build_edit_state)
    g.add_node("generate_suggestions", node_generate_suggestions)
    g.add_node("section_review",       node_section_review)
    g.add_node("apply_edit",           node_apply_edit)
    g.add_node("score_resume",         node_score_resume)
    g.add_node("refine_or_finish",     node_refine_or_finish)

    # Entry point
    g.set_entry_point("parse_and_retrieve")

    # Linear path until item selection
    g.add_edge("parse_and_retrieve", "item_selection")
    g.add_edge("item_selection", "build_edit_state")
    g.add_edge("build_edit_state", "generate_suggestions")
    g.add_edge("generate_suggestions", "section_review")
    g.add_edge("section_review", "apply_edit")

    # After apply_edit: loop back for remaining sections or move to scoring
    g.add_conditional_edges(
        "apply_edit",
        route_after_review_or_done,
        {
            "more_sections": "section_review",
            "all_done":      "score_resume",
        },
    )

    g.add_edge("score_resume", "refine_or_finish")

    # After showing score: refine (re-run suggestions) or finish
    g.add_conditional_edges(
        "refine_or_finish",
        route_after_refine_or_finish,
        {
            "refine": "generate_suggestions",
            "finish": END,
        },
    )

    return g


# ======================================================================
# Compiled graph singleton (with in-memory checkpointer)
# ======================================================================

_checkpointer = MemorySaver()
edit_graph    = build_edit_graph().compile(checkpointer=_checkpointer, interrupt_before=[])
