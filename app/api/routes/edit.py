# app/api/routes/edit.py
#
# Edit-session endpoints — backed by the LangGraph edit_graph.
#
# Session lifecycle
# -----------------
#   POST /edit/start                   → starts the graph, returns thread_id + first interrupt
#   POST /edit/{tid}/resume            → resume after an interrupt with the user's response
#   GET  /edit/{tid}/state             → inspect full graph state (debug / polling)
#   GET  /edit/{tid}/preview           → compile current LaTeX to a preview string
#   POST /edit/{tid}/score             → score current state and return feedback
#   POST /edit/{tid}/finish            → mark session complete, save to DB
#   GET  /edit/{tid}/diff              → full-resume original vs final diff
#   GET  /edit/sessions/{user_id}      → list past sessions for a user

from __future__ import annotations

import uuid
import concurrent.futures

import subprocess
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, model_validator

from langgraph.types import Command

from app.graph.edit_graph import edit_graph, _rebuild_agent, _checkpointer, register_thread
from app.models.jd import ParsedJD
from app.models.resume import ParsedResume
from app.agents.generator_agent import generate as _generate_fn
import app.core.pipeline as pl
from app.utils import get_logger, SQLHandler
from app.core.retriever import FLAT_SECTIONS, ATOMIC_SECTIONS

log = get_logger(__name__)
router = APIRouter(prefix="/edit", tags=["Edit Session"])


# ======================================================================
# Shared parallel generation helper
# ======================================================================

def _run_generation_parallel(
    gen_types: list[str],
    resume,
    jd,
    custom_instruction: str | None,
    no_jd: bool,
) -> tuple[dict, list[str]]:
    """
    Dispatch each generation type to a thread pool.
    Returns (results_dict, errors_list).

    Because _generate_fn checks the generation cache first, types that
    were already generated (same inputs) return instantly from cache —
    the ThreadPoolExecutor overhead is negligible for cache hits.
    """
    results: dict  = {}
    errors:  list  = []

    def _one(gen_type: str) -> tuple[str, str]:
        return gen_type, _generate_fn(
            resume=resume, jd=jd, type=gen_type,
            custom_instruction=custom_instruction,
            no_jd=no_jd,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gen_types) or 1) as pool:
        futures = {pool.submit(_one, t): t for t in gen_types}
        for future in concurrent.futures.as_completed(futures):
            gen_type = futures[future]
            try:
                _, result = future.result()
                results[gen_type] = result
            except Exception as exc:
                log.error("[generate] %s failed: %s", gen_type, exc)
                errors.append(f"{gen_type}: {exc}")

    return results, errors


# ======================================================================
# Helpers
# ======================================================================

def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _get_interrupt(thread_id: str) -> dict | None:
    """Return the current interrupt payload, or None if graph is done."""
    state = edit_graph.get_state(_config(thread_id))
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


# ======================================================================
# Request / response models
# ======================================================================

class StartSessionRequest(BaseModel):
    user_id:            int
    jd_text:            str
    resume_path:        str | None = None   # absolute path to uploaded .tex file
    existing_resume_id: int | None = None   # use a specific DB resume for personal info
    custom_instruction: str | None = None   # global instruction applied to all edit prompts
    # If neither resume_path nor existing_resume_id is supplied the graph loads
    # the most-recent resume for user_id automatically (user-profile mode).


class StartSessionResponse(BaseModel):
    thread_id: str
    interrupt: dict | None   # first interrupt payload (item_selection)


class ResumeRequest(BaseModel):
    response: dict   # the user's answer to the current interrupt


class ResumeResponse(BaseModel):
    interrupt: dict | None   # next interrupt payload, or None when finished
    stage:     str | None


class StateResponse(BaseModel):
    stage:              str | None
    error:              str | None
    score_result:       dict | None
    generation_results: dict | None
    parsed_jd:          dict | None = None
    parsed_resume:      dict | None = None


class PreviewResponse(BaseModel):
    latex: str


class ScoreResponse(BaseModel):
    score_result:   dict
    score_feedback: str


class SessionGenerateRequest(BaseModel):
    generation_types:   list[str]         # ["coverletter", "email", "outreachmessage"]
    custom_instruction: str | None = None
    no_jd:              bool       = False


class SessionGenerateResponse(BaseModel):
    results: dict
    errors:  str | None = None


class QuickGenerateRequest(BaseModel):
    user_id:            int
    jd_text:            str | None = None  # required when no_jd=False; ignored when no_jd=True
    generation_types:   list[str]          # ["coverletter", "email", "outreachmessage"]
    # resume_id is optional: if supplied it is used for personal info; otherwise
    # the most-recent resume for user_id is used automatically.
    resume_id:          int | None = None
    custom_instruction: str | None = None
    no_jd:              bool       = False


class QuickGenerateResponse(BaseModel):
    results:   dict
    parsed_jd: dict | None = None
    errors:    str | None  = None


class GenerateFromItemsRequest(BaseModel):
    generation_types:   list[str]
    selected_items:     list[dict] | None = None  # None → use all ranked_items from state
    custom_instruction: str | None = None
    no_jd:              bool       = False


# ======================================================================
# POST /edit/start
# ======================================================================

@router.post("/start", response_model=StartSessionResponse)
def start_session(req: StartSessionRequest):
    """
    Start a new editing session.

    Runs the graph until the first interrupt (item_selection) and returns
    the thread_id plus the interrupt payload.
    """
    thread_id = str(uuid.uuid4())
    config    = _config(thread_id)

    initial_state: dict = {
        "user_id":  req.user_id,
        "jd_text":  req.jd_text,
        "cycle_id": 1,
    }
    if req.custom_instruction and req.custom_instruction.strip():
        initial_state["custom_instruction"] = req.custom_instruction.strip()
    if req.existing_resume_id:
        initial_state["existing_resume_id"] = req.existing_resume_id
    else:
        initial_state["resume_path"] = req.resume_path

    try:
        edit_graph.invoke(initial_state, config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    register_thread(thread_id)
    interrupt_payload = _get_interrupt(thread_id)

    # Persist session row — job_id is now in state after parse_and_retrieve
    try:
        state_after = edit_graph.get_state(config).values
        job_id = state_after.get("job_id")
        db = SQLHandler()
        db.create_edit_session(
            user_id=req.user_id,
            thread_id=thread_id,
            job_id=job_id,
            custom_instruction=req.custom_instruction,
        )
    except Exception as exc:
        log.warning("[edit] failed to persist session to DB: %s", exc)

    return StartSessionResponse(thread_id=thread_id, interrupt=interrupt_payload)


# ======================================================================
# POST /edit/{thread_id}/resume
# ======================================================================

@router.post("/{thread_id}/resume", response_model=ResumeResponse)
def resume_session(thread_id: str, req: ResumeRequest):
    """
    Resume a paused session by supplying the user's response to the
    current interrupt.

    The graph runs until the next interrupt (or until it finishes).
    """
    config = _config(thread_id)

    # Verify the thread exists
    try:
        edit_graph.get_state(config)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    try:
        edit_graph.invoke(Command(resume=req.response), config=config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    interrupt_payload = _get_interrupt(thread_id)
    current_state     = edit_graph.get_state(config).values
    return ResumeResponse(
        interrupt=interrupt_payload,
        stage=current_state.get("stage"),
    )


# ======================================================================
# GET /edit/{thread_id}/state
# ======================================================================

@router.get("/{thread_id}/state", response_model=StateResponse)
def get_session_state(thread_id: str):
    """Inspect the current graph state (useful for debugging or polling)."""
    config = _config(thread_id)
    try:
        values = edit_graph.get_state(config).values
    except Exception:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    return StateResponse(
        stage=values.get("stage"),
        error=values.get("error"),
        score_result=values.get("score_result"),
        generation_results=values.get("generation_results"),
        parsed_jd=values.get("parsed_jd"),
        parsed_resume=values.get("parsed_resume"),
    )


# ======================================================================
# GET /edit/{thread_id}/preview
# ======================================================================

@router.get("/{thread_id}/preview", response_model=PreviewResponse)
def build_preview(
    thread_id:        str,
    user_id:          int | None   = None,
    section_order:    str | None   = None,   # comma-separated list, e.g. "experience,education,skills"
    section_space:    float | None = None,
    subsection_space: float | None = None,
    font_size:        int | None   = None,
):
    """
    Build the current edit state into a compilable LaTeX string.
    Accepts optional layout overrides as query parameters.
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values
        if not state_values.get("edit_cycle_dict"):
            raise HTTPException(status_code=400, detail="Edit state not ready yet.")

        agent         = _rebuild_agent(state_values)
        order_list    = section_order.split(",") if section_order else None
        latex = pl.build_preview(
            edit_agent       = agent,
            user_id          = user_id,
            section_order    = order_list,
            section_space    = section_space,
            subsection_space = subsection_space,
            font_size        = font_size,
        )
        return PreviewResponse(latex=latex)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# POST /edit/{thread_id}/score
# ======================================================================

@router.post("/{thread_id}/score", response_model=ScoreResponse)
def score_session(thread_id: str):
    """
    Score the current resume state against the JD.
    Returns per-dimension scores and a formatted feedback string.
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values
        if not state_values.get("edit_cycle_dict"):
            raise HTTPException(status_code=400, detail="Edit state not ready yet.")

        agent     = _rebuild_agent(state_values)
        parsed_jd = ParsedJD(**state_values["parsed_jd"])
        score     = pl.score_resume(
            edit_agent=agent,
            parsed_jd=parsed_jd,
            resume_id=state_values.get("resume_id", 0),
        )
        feedback  = pl.format_score_feedback(score)
        return ScoreResponse(
            score_result=score.model_dump(),
            score_feedback=feedback,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# POST /edit/{thread_id}/generate
# ======================================================================

@router.post("/{thread_id}/generate", response_model=SessionGenerateResponse)
def generate_from_session(thread_id: str, req: SessionGenerateRequest):
    """
    Generate documents (cover letter, email, outreach) using the tailored
    resume from the current edit session.  Can be called at any point after
    the edit state is built (i.e. after item_selection is confirmed).
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values
        if not state_values.get("edit_cycle_dict"):
            raise HTTPException(status_code=400, detail="Edit state not ready yet.")

        agent           = _rebuild_agent(state_values)
        parsed_jd       = ParsedJD(**state_values["parsed_jd"])
        original_resume = ParsedResume(**state_values["parsed_resume"])
        tailored_resume = pl.collapse_edit_state_to_resume(
            agent, original_resume.personal_info
        )

        results, errors = _run_generation_parallel(
            gen_types=req.generation_types,
            resume=tailored_resume,
            jd=parsed_jd,
            custom_instruction=req.custom_instruction,
            no_jd=req.no_jd,
        )
        return SessionGenerateResponse(
            results=results,
            errors="; ".join(errors) if errors else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# POST /edit/{thread_id}/generate-from-items
# ======================================================================

@router.post("/{thread_id}/generate-from-items", response_model=SessionGenerateResponse)
def generate_from_items(thread_id: str, req: GenerateFromItemsRequest):
    """
    Generate documents from a set of retrieved section-items WITHOUT
    starting (or completing) the editing cycle.

    Called right after the item_selection interrupt — the caller passes
    the user-selected items so the system can build a tailored resume
    and generate the requested documents immediately.

    selected_items format: same list[dict] as ranked_items in the
    item_selection interrupt payload.
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values

        if "parsed_jd" not in state_values or "parsed_resume" not in state_values:
            raise HTTPException(
                status_code=400,
                detail="Session not ready — call /edit/start first.",
            )

        parsed_jd     = ParsedJD(**state_values["parsed_jd"])
        parsed_resume = ParsedResume(**state_values["parsed_resume"])

        items = req.selected_items if req.selected_items is not None else state_values.get("ranked_items", [])
        if not items:
            raise HTTPException(status_code=400, detail="No sections available to generate from.")

        edit_state = pl.build_starting_edit_state(items)
        agent      = pl.build_edit_agent(edit_state, parsed_jd, cycle_id=0)
        tailored   = pl.collapse_edit_state_to_resume(agent, parsed_resume.personal_info)

        results, errors = _run_generation_parallel(
            gen_types=req.generation_types,
            resume=tailored,
            jd=parsed_jd,
            custom_instruction=req.custom_instruction,
            no_jd=req.no_jd,
        )
        return SessionGenerateResponse(
            results=results,
            errors="; ".join(errors) if errors else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# POST /edit/quick-generate
# ======================================================================

@router.post("/quick-generate", response_model=QuickGenerateResponse)
def quick_generate(req: QuickGenerateRequest):
    """
    Generate documents WITHOUT starting an edit session.

    Parses the JD, runs JD-similarity retrieval across ALL saved resume
    sections (picking the best version of each section-item automatically),
    assembles a tailored ParsedResume, then generates the requested documents.

    Useful at the start of a workflow when the user wants cover letter / email
    / outreach message without going through the full editing flow.
    """
    try:
        # ── Cold-outreach mode: no JD required ─────────────────────────
        if req.no_jd:
            if req.resume_id is not None:
                _, parsed_resume = pl.load_resume_from_db(req.resume_id)
            else:
                _, parsed_resume = pl.load_latest_resume_for_user(req.user_id)

            results, errors = _run_generation_parallel(
                gen_types=req.generation_types,
                resume=parsed_resume,
                jd=None,
                custom_instruction=req.custom_instruction,
                no_jd=True,
            )
            return QuickGenerateResponse(
                results=results,
                parsed_jd=None,
                errors="; ".join(errors) if errors else None,
            )

        # ── Normal mode: parse JD + retrieve best sections ─────────────
        jd_text = (req.jd_text or "").strip()
        if not jd_text:
            raise HTTPException(
                status_code=422,
                detail="jd_text is required when no_jd is False.",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            jd_fut  = pool.submit(pl.parse_jd, jd_text, req.user_id)
            if req.resume_id is not None:
                res_fut = pool.submit(pl.load_resume_from_db, req.resume_id)
            else:
                res_fut = pool.submit(pl.load_latest_resume_for_user, req.user_id)

            job_id, parsed_jd = jd_fut.result()
            _, parsed_resume  = res_fut.result()

        # ── Retrieve best sections scoped to this user's resumes ───────
        user_resume_ids = pl.get_user_resume_ids(req.user_id)
        raw_results = pl.retrieve(
            job_id=job_id,
            parsed_jd=parsed_jd,
            resume_ids=user_resume_ids if user_resume_ids else None,
        )
        ranked = pl.rank_and_filter(raw_results)

        # ── Build tailored resume from best sections ────────────────────
        edit_state = pl.build_starting_edit_state(ranked)
        agent      = pl.build_edit_agent(edit_state, parsed_jd, cycle_id=0)
        tailored   = pl.collapse_edit_state_to_resume(agent, parsed_resume.personal_info)

        # ── Generate all docs in parallel (cache hits return instantly) ─
        results, errors = _run_generation_parallel(
            gen_types=req.generation_types,
            resume=tailored,
            jd=parsed_jd,
            custom_instruction=req.custom_instruction,
            no_jd=False,
        )
        return QuickGenerateResponse(
            results=results,
            parsed_jd=parsed_jd.model_dump(),
            errors="; ".join(errors) if errors else None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# GET /edit/{thread_id}/export
# ======================================================================

@router.get("/{thread_id}/export")
def export_session(
    thread_id:        str,
    format:           str        = "tex",   # "tex" | "pdf"
    section_order:    str | None = None,
    section_space:    float | None = None,
    subsection_space: float | None = None,
    font_size:        int | None   = None,
):
    """
    Export the current resume as a downloadable .tex or compiled .pdf file.

    Available at any point during an edit session (not just at the finish screen).
    For `format=pdf`, pdflatex must be installed and in PATH.
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values
        if not state_values.get("edit_cycle_dict"):
            raise HTTPException(status_code=400, detail="Edit state not ready yet.")

        agent      = _rebuild_agent(state_values)
        order_list = section_order.split(",") if section_order else None
        latex      = pl.build_preview(
            edit_agent       = agent,
            user_id          = state_values.get("user_id"),
            section_order    = order_list,
            section_space    = section_space,
            subsection_space = subsection_space,
            font_size        = font_size,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    short_id = thread_id[:8]

    if format == "tex":
        return Response(
            content=latex.encode("utf-8"),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="resume_{short_id}.tex"'},
        )

    if format == "pdf":
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tex_file = Path(tmpdir) / "resume.tex"
                tex_file.write_text(latex, encoding="utf-8")
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, str(tex_file)],
                    capture_output=True,
                    timeout=60,
                )
                pdf_file = Path(tmpdir) / "resume.pdf"
                if not pdf_file.exists():
                    stderr = result.stderr.decode(errors="replace")[:400]
                    raise HTTPException(status_code=500, detail=f"pdflatex failed: {stderr}")
                pdf_bytes = pdf_file.read_bytes()
        except HTTPException:
            raise
        except FileNotFoundError:
            raise HTTPException(
                status_code=501,
                detail="pdflatex is not installed or not in PATH. Use format=tex instead.",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="resume_{short_id}.pdf"'},
        )

    raise HTTPException(status_code=422, detail=f"Unknown format '{format}'. Use 'tex' or 'pdf'.")


# ======================================================================
# POST /edit/{thread_id}/finish
# ======================================================================

class FinishResponse(BaseModel):
    thread_id:   str
    final_latex: str | None
    session_id:  int | None
    documents_saved: int


@router.post("/{thread_id}/finish", response_model=FinishResponse)
def finish_session(
    thread_id:    str,
    section_order: str | None = None,
    documents:    dict = {},
):
    """
    Mark an edit session as complete.

    Builds the final LaTeX from current state and persists it (along with
    any generated documents) to the DB.  Safe to call even if the graph
    has already reached END.

    Query params:
        section_order   — comma-separated section order for the final LaTeX
        documents       — {doc_type: content} JSON body (optional)
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values
    except Exception:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")

    # Build final LaTeX if edit state is available
    final_latex: str | None = None
    if state_values.get("edit_cycle_dict"):
        try:
            agent      = _rebuild_agent(state_values)
            order_list = section_order.split(",") if section_order else None
            final_latex = pl.build_preview(
                edit_agent    = agent,
                user_id       = state_values.get("user_id"),
                section_order = order_list,
            )
        except Exception as exc:
            log.warning("[edit] could not build final LaTeX at finish: %s", exc)

    # Persist to DB
    session_id: int | None = None
    docs_saved = 0
    try:
        db     = SQLHandler()
        job_id = state_values.get("job_id")
        db.finish_edit_session(
            thread_id=thread_id,
            final_latex=final_latex,
            job_id=job_id,
        )
        row = db.get_edit_session_by_thread(thread_id)
        if row:
            session_id = row["id"]
            if documents and session_id:
                db.save_session_documents(session_id, documents)
                docs_saved = len(documents)
    except Exception as exc:
        log.warning("[edit] failed to persist finish to DB: %s", exc)

    return FinishResponse(
        thread_id=thread_id,
        final_latex=final_latex,
        session_id=session_id,
        documents_saved=docs_saved,
    )


# ======================================================================
# GET /edit/sessions/{user_id}
# ======================================================================

class SessionSummary(BaseModel):
    id:                 int
    thread_id:          str
    status:             str
    company:            str | None
    role:               str | None
    custom_instruction: str | None
    created_at:         str
    finished_at:        str | None


@router.get("/sessions/{user_id}", response_model=list[SessionSummary])
def list_sessions(
    user_id: int,
    status:  str | None = None,
    limit:   int        = 50,
):
    """
    List past edit sessions for a user, newest first.
    Optional `status` filter: active | completed | abandoned.
    """
    try:
        db   = SQLHandler()
        rows = db.list_edit_sessions(user_id=user_id, status=status, limit=limit)
        return [SessionSummary(**r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ======================================================================
# GET /edit/{thread_id}/diff
# ======================================================================

class SectionDiff(BaseModel):
    original: str
    final:    str
    changed:  bool


class ItemDiff(BaseModel):
    item:     str
    original: str
    final:    str
    changed:  bool


class ResumeDiff(BaseModel):
    flat_sections:   dict[str, SectionDiff]
    atomic_sections: dict[str, list[ItemDiff]]
    has_changes:     bool


@router.get("/{thread_id}/diff", response_model=ResumeDiff)
def get_resume_diff(thread_id: str):
    """
    Return a full-resume diff: original (root state) vs current state.

    Both flat sections (education, skills, …) and atomic sections
    (experience items, project items) are compared.
    """
    config = _config(thread_id)
    try:
        state_values = edit_graph.get_state(config).values
        if not state_values.get("edit_cycle_dict"):
            raise HTTPException(status_code=400, detail="Edit state not ready yet.")

        agent = _rebuild_agent(state_values)
        cycle = agent.editing_cycle

        root_state    = cycle.nodes[0].resume_state
        current_state = cycle.current_state

        flat: dict[str, SectionDiff] = {}
        for sec in FLAT_SECTIONS:
            root_sec    = root_state.get_section(sec)
            current_sec = current_state.get_section(sec)
            if root_sec is None and current_sec is None:
                continue
            orig  = root_sec.updated_section    if root_sec    else ""
            final = current_sec.updated_section if current_sec else ""
            flat[sec] = SectionDiff(original=orig, final=final, changed=(orig != final))

        atomic: dict[str, list[ItemDiff]] = {}
        for sec in ATOMIC_SECTIONS:
            root_items    = getattr(root_state,    sec, []) or []
            current_items = getattr(current_state, sec, []) or []
            # Build name→item maps for reliable matching
            root_by_name    = {it.item_name: it for it in root_items    if it}
            current_by_name = {it.item_name: it for it in current_items if it}
            all_names = list(dict.fromkeys(
                [it.item_name for it in root_items    if it] +
                [it.item_name for it in current_items if it]
            ))
            diffs: list[ItemDiff] = []
            for name in all_names:
                orig  = root_by_name[name].updated_section    if name in root_by_name    else ""
                final = current_by_name[name].updated_section if name in current_by_name else ""
                diffs.append(ItemDiff(item=name, original=orig, final=final, changed=(orig != final)))
            if diffs:
                atomic[sec] = diffs

        has_changes = (
            any(s.changed for s in flat.values()) or
            any(d.changed for items in atomic.values() for d in items)
        )
        return ResumeDiff(
            flat_sections=flat,
            atomic_sections=atomic,
            has_changes=has_changes,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
