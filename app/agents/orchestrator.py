# orchestrator.py
#
# Thin facade over pipeline.py.  Keeps singleton helpers and exposes
# the high-level verbs (parse_jd, retrieve, edit, score, generate)
# as module-level functions so callers never need to manage objects.

from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.models.resume import PersonalInfo, ResumeSection, ParsedResume
from app.models.jd import ParsedJD
from app.models.edit import ResumeEditCycle, ResumeEditState, SectionEditState, ItemEditState
from app.core.retriever import ATOMIC_SECTIONS, FLAT_SECTIONS
from app.utils import get_logger, SQLHandler
from app.agents.generator_agent import generate as _generate_content
from app.agents.score_agent import score as _score_resume

import app.core.pipeline as _pl

log = get_logger(__name__)


# ------------------------------------------------------------------
# Lazy user context (fetched on first call, not at import time)
# ------------------------------------------------------------------

_sql_helper: SQLHandler | None = None
_personal_info: PersonalInfo | None = None


def _get_sql() -> SQLHandler:
    global _sql_helper
    if _sql_helper is None:
        _sql_helper = SQLHandler()
    return _sql_helper


def get_personal_info(name: str) -> PersonalInfo | None:
    global _personal_info
    if _personal_info is None:
        row = _get_sql().fetch_one("users", {"name": name})
        if row:
            _personal_info = PersonalInfo(**row)
    return _personal_info


# ------------------------------------------------------------------
# Stage 1 — JD parsing
# ------------------------------------------------------------------

def parse_jd(jd: str, user_id: int) -> tuple[int, ParsedJD]:
    return _pl.parse_jd(jd_text=jd, user_id=user_id)


# ------------------------------------------------------------------
# Stage 2 — Resume parsing
# ------------------------------------------------------------------

def parse_resume(resume_path: str | Path, user_id: int) -> tuple[int, ParsedResume]:
    return _pl.parse_resume(resume_path=resume_path, user_id=user_id)


# ------------------------------------------------------------------
# Stage 3 — Retrieval
# ------------------------------------------------------------------

def retrieve(job_id: int, parsed: ParsedJD, **kwargs) -> list[dict]:
    """
    Run retrieval + rank_and_filter in one call.

    Returns:
        list of dicts: {section_name, item_name, scores, rows}
    """
    raw = _pl.retrieve(job_id=job_id, parsed_jd=parsed, **kwargs)
    return _pl.rank_and_filter(raw)


def combine_retrieved_results(
    retrieved_results: list[tuple[str, str | None, float, dict]],
    top_k_per_section: int | None = None,
) -> dict[str, list[dict] | dict]:
    """
    Combines the flat list returned by the old retrieve() into a
    section-keyed dict ready to be passed into build_starting_edit_state().

    Input (legacy format):
        list of (sec_name, item_name, score, row)

    Output:
        {
            "skills":     [{"content_latex": ..., "score": 0.91}],
            "experience": {"Google": {"content_latex": ..., "score": 0.87}},
        }
    """
    flat_acc:   dict[str, list[tuple[float, dict]]]      = {}
    atomic_acc: dict[str, dict[str, tuple[float, dict]]] = {}

    for sec_name, item_name, score, row in retrieved_results:
        if row is None:
            log.warning("combine | row is None sec=%s item=%s", sec_name, item_name)
            continue

        if item_name is None:
            flat_acc.setdefault(sec_name, []).append((score, row))
        else:
            atomic_acc.setdefault(sec_name, {})
            existing_score, _ = atomic_acc[sec_name].get(item_name, (-1.0, {}))
            if score > existing_score:
                atomic_acc[sec_name][item_name] = (score, row)

    combined: dict[str, list[dict] | dict] = {}

    for sec_name, entries in flat_acc.items():
        sorted_entries = sorted(entries, key=lambda x: x[0], reverse=True)
        if top_k_per_section:
            sorted_entries = sorted_entries[:top_k_per_section]
        combined[sec_name] = [{**row, "score": round(s, 4)} for s, row in sorted_entries]

    for sec_name, item_map in atomic_acc.items():
        sorted_items = sorted(item_map.items(), key=lambda x: x[1][0], reverse=True)
        if top_k_per_section:
            sorted_items = sorted_items[:top_k_per_section]
        combined[sec_name] = {
            item_name: {**row, "score": round(s, 4)}
            for item_name, (s, row) in sorted_items
        }

    log.info("combine_retrieved_results | sections=%s", {k: len(v) for k, v in combined.items()})
    return combined


# ------------------------------------------------------------------
# Stage 4 — Edit state helpers
# ------------------------------------------------------------------

def edit_state_to_dict(state: ResumeEditState) -> Dict:
    """Serialise a ResumeEditState to a plain dict keyed by section name."""
    res: Dict = {}
    for sec in FLAT_SECTIONS:
        sec_state = getattr(state, sec, None)
        if sec_state:
            res[sec] = sec_state.updated_section
    for sec in ATOMIC_SECTIONS:
        item_dict = {}
        for item in (getattr(state, sec, None) or []):
            if item:
                item_dict[item.item_name] = item.updated_section
        res[sec] = item_dict
    return res


def build_starting_edit_state(ranked_items: list[dict]) -> ResumeEditState:
    """Delegate to pipeline.build_starting_edit_state."""
    return _pl.build_starting_edit_state(ranked_items)


# ------------------------------------------------------------------
# Stage 5 — Generation
# ------------------------------------------------------------------

def generation_email_letter_message(
    jd: ParsedJD,
    resume: ParsedResume,
    generation_type: str,
) -> str:
    return _generate_content(resume, jd, generation_type)


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score_resume_against_jd(
    resume_latex: str,
    jd: ParsedJD,
    app_id: int = 0,
    resume_id: int = 0,
) -> dict:
    """Score a raw LaTeX resume against a ParsedJD. Returns ResumeScore as dict."""
    result = _score_resume(
        app_id=app_id,
        resume_id=resume_id,
        resume_latex=resume_latex,
        jd=jd,
    )
    return result.model_dump()
