# pipeline.py

from pathlib import Path

from config.config import get_config_dict
from app.utils import SQLHandler, get_logger

from app.models.jd import ParsedJD
from app.models.resume import ParsedResume, PersonalInfo, ResumeSection, SectionContent, AtomicItem
from app.models.edit import ResumeEditState, ResumeEditCycle, SectionEditState, ItemEditState
from app.models.scoring import ResumeScore
from app.models.resume_builder import ResumeBuilderInput

from app.core.jd_parser import JDParser
from app.core.resume_parser import ResumeParser
from app.core.retriever import ResumeRetriever, FLAT_SECTIONS, ATOMIC_SECTIONS
from app.core.resume_builder import ResumeBuilder
from app.agents.edit_agent import EditAgent
from app.agents.score_agent import score as score_resume_fn
from app.agents.generator_agent import generate as generate_fn

log = get_logger(__name__)
config = get_config_dict()
retriever_config = config['retriever_config']

# ------------------------------------------------------------------
# Lazy singletons — instantiated on first use to avoid import-time side-effects
# ------------------------------------------------------------------

_jd_parser:     JDParser | None      = None
_resume_parser: ResumeParser | None  = None
_retriever:     ResumeRetriever | None = None


def _get_jd_parser() -> JDParser:
    global _jd_parser
    if _jd_parser is None:
        _jd_parser = JDParser()
    return _jd_parser


def _get_resume_parser() -> ResumeParser:
    global _resume_parser
    if _resume_parser is None:
        _resume_parser = ResumeParser()
    return _resume_parser


def _get_retriever() -> ResumeRetriever:
    global _retriever
    if _retriever is None:
        _retriever = ResumeRetriever()
    return _retriever


# ------------------------------------------------------------------
# JD
# ------------------------------------------------------------------

def parse_jd(jd_text: str, user_id: int) -> tuple[int, ParsedJD]:
    """Parse raw JD text, save to DB, embed to ChromaDB."""
    result = _get_jd_parser().run(jd_text=jd_text, user_id=user_id)
    job_id = result["job_id"]
    parsed = ParsedJD(**result["parsed"]) if isinstance(result["parsed"], dict) else result["parsed"]
    return job_id, parsed


# ------------------------------------------------------------------
# Resume
# ------------------------------------------------------------------

def parse_resume(resume_path: str | Path, user_id: int) -> tuple[int, ParsedResume]:
    """Parse raw resume LaTeX file, save to DB, embed to ChromaDB."""
    parsed_resume = _get_resume_parser().run(resume_path=Path(resume_path), user_id=user_id)
    resume_id = int(parsed_resume.resume_id) if parsed_resume.resume_id.isdigit() else 0
    return resume_id, parsed_resume


def get_user_resume_ids(user_id: int) -> list[int]:
    """Return all resume IDs stored in the DB for a given user_id."""
    db   = SQLHandler()
    rows = db.execute_raw(
        "SELECT id FROM resumes WHERE user_id = :uid ORDER BY created_at DESC",
        {"uid": user_id},
    )
    return [int(r["id"]) for r in (rows or [])]


def load_latest_resume_for_user(user_id: int) -> tuple[int, ParsedResume]:
    """Load the most-recently stored resume for a user (used for personal info)."""
    db  = SQLHandler()
    row = db.execute_raw(
        "SELECT id FROM resumes WHERE user_id = :uid ORDER BY created_at DESC LIMIT 1",
        {"uid": user_id},
    )
    if not row:
        raise ValueError(f"No resumes found in DB for user_id={user_id}")
    return load_resume_from_db(int(row[0]["id"]))


def load_resume_from_db(resume_id: int) -> tuple[int, ParsedResume]:
    """
    Reconstruct a ParsedResume entirely from SQLite — no file access needed.
    Used when the user selects an existing resume from the DB instead of uploading.
    """
    db = SQLHandler()

    row = db.fetch_one("resumes", filters={"id": resume_id})
    if row is None:
        raise ValueError(f"No resume found in DB with id={resume_id}")

    personal_info = PersonalInfo(
        name=row.get("name") or "Unknown",
        email=row.get("email") or "",
        phone=row.get("phone") or "",
        github=row.get("github_url") or "",
        linkedin=row.get("linkedin_url") or "",
    )

    sections_df = db.fetch_table_where("resume_sections", filters={"resume_id": resume_id})

    flat: dict[str, SectionContent] = {}
    atomic: dict[str, list[AtomicItem]] = {s: [] for s in ATOMIC_SECTIONS}

    for _, sec_row in sections_df.iterrows():
        sec_name = sec_row["section_name"]
        if sec_name in FLAT_SECTIONS:
            flat[sec_name] = SectionContent(
                content_latex=sec_row.get("content_latex") or None,
                content_text=sec_row.get("content_text") or None,
            )
        elif sec_name in ATOMIC_SECTIONS:
            items = db.execute_raw(
                "SELECT item_name, role_title, content_latex, content_text, item_index "
                "FROM resume_section_items "
                "WHERE section_id = :sid AND is_master = 1 ORDER BY item_index",
                {"sid": int(sec_row["id"])},
            )
            if isinstance(items, list):
                for item in items:
                    atomic[sec_name].append(AtomicItem(
                        name=item.get("item_name"),
                        role=item.get("role_title"),
                        content_latex=item.get("content_latex") or "",
                        content_text=item.get("content_text") or "",
                    ))

    resume_sections = ResumeSection(
        education=flat.get("education"),
        achievements=flat.get("achievements"),
        skills=flat.get("skills"),
        relevant_coursework=flat.get("relevant_coursework"),
        experience=atomic.get("experience", []),
        projects=atomic.get("projects", []),
    )

    return resume_id, ParsedResume(
        resume_id=str(resume_id),
        resume_path=row.get("resume_path") or "",
        personal_info=personal_info,
        resume_sections=resume_sections,
    )


# ------------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------------

def retrieve(
    job_id: int,
    parsed_jd: ParsedJD,
    top_k: int | None = retriever_config['top_k'],
    fetch_mult: int | None = retriever_config['fetch_multiplier'],
    w_skills: float | None = retriever_config['weight_req_skills'],
    w_resp: float | None = retriever_config['weight_resp'],
    w_nth: float | None = retriever_config['weight_nice_to_have'],
    resume_ids: list[int] | None = None,
) -> list[tuple[str, str | None, list[tuple[float, str, dict]]]]:
    """Run retriever, return raw .run() output for rank_and_filter.

    resume_ids — if supplied, only sections from those resumes are searched.
    Pass the result of get_user_resume_ids(user_id) to scope retrieval to one
    user's history while still picking the most JD-relevant version.
    """
    r = _get_retriever()
    r.top_k        = top_k
    r.fetch_mult   = fetch_mult
    r.weight_skill = w_skills
    r.weight_resp  = w_resp
    r.weight_nth   = w_nth
    return r.run(job_id=job_id, jd_obj=parsed_jd, resume_ids=resume_ids)


def rank_and_filter(
    run_results: list[tuple[str, str | None, list[tuple[float, str, dict]]]],
) -> list[dict]:
    """
    Rank atomic section items by relevance and fetch DB rows.

    Returns:
        [
            {"section_name": "skills",     "item_name": None,     "scores": [..], "rows": [dict]},
            {"section_name": "experience", "item_name": "Google", "scores": [..], "rows": [dict, ...]},
        ]
    """
    return _get_retriever().rank_and_filter(run_results)


# ------------------------------------------------------------------
# Edit state construction
# ------------------------------------------------------------------

def build_starting_edit_state(ranked_items: list[dict]) -> ResumeEditState:
    """
    Convert rank_and_filter() output into an initial ResumeEditState.

    ranked_items is a list of dicts, each like:
        {"section_name": str, "item_name": str | None, "scores": [...], "rows": [db_row_dict, ...]}

    Only items passed in are included — caller is responsible for
    filtering to only user-selected items before calling this.
    """
    flat_states:   dict[str, SectionEditState]       = {}
    atomic_states: dict[str, list[ItemEditState]]    = {s: [] for s in ATOMIC_SECTIONS}

    for entry in ranked_items:
        sec_name  = entry["section_name"]
        item_name = entry["item_name"]
        rows      = entry.get("rows", [])
        best_row  = next((r for r in rows if r and r.get("content_latex")), {})
        latex     = best_row.get("content_latex", "") or ""

        if item_name is None:
            # Flat section — last write wins if the same section appears twice
            flat_states[sec_name] = SectionEditState(
                section_name=sec_name,
                section_previous_state="",
                lines_to_change=[],
                suggested_changes=[],
                updated_section=latex,
            )
        elif sec_name in atomic_states:
            atomic_states[sec_name].append(
                ItemEditState(
                    section_name=sec_name,
                    item_name=item_name,
                    section_previous_state="",
                    lines_to_change=[],
                    suggested_changes=[],
                    updated_section=latex,
                )
            )

    state_dict: dict = {}
    for sec in FLAT_SECTIONS:
        if sec in flat_states:
            state_dict[sec] = flat_states[sec]
    for sec in ATOMIC_SECTIONS:
        items = atomic_states.get(sec, [])
        if items:
            state_dict[sec] = items

    return ResumeEditState(**state_dict)


def build_edit_agent(
    edit_state: ResumeEditState,
    parsed_jd: ParsedJD,
    cycle_id: int,
    llm: str = "gemini",
) -> EditAgent:
    """Initialise EditAgent with a fresh ResumeEditCycle."""
    cycle = ResumeEditCycle(cycle_id=cycle_id, jd=parsed_jd)
    return EditAgent(editing_cycle=cycle, resume=edit_state, usellm=llm)


# ------------------------------------------------------------------
# Suggestions
# ------------------------------------------------------------------

def generate_suggestions(edit_agent: EditAgent) -> None:
    """Trigger full resume suggestion pass (all sections in one LLM call)."""
    edit_agent._resume_suggestions()


def generate_suggestions_with_score_feedback(
    edit_agent: EditAgent,
    score_feedback: str,
) -> None:
    """
    Same as generate_suggestions but injects a score summary as a special
    instruction so the LLM focuses on the weakest areas.
    """
    edit_agent.special_instruction = score_feedback
    edit_agent._resume_suggestions()
    edit_agent.special_instruction = ""


# ------------------------------------------------------------------
# Section editing
# ------------------------------------------------------------------

def apply_edit(
    edit_agent: EditAgent,
    section_name: str,
    proposal: str,          # "reject" | "paraphrase" | "another_suggestion"
    item_name: str | None = None,
    special_instruction: str | None = None,
    rejection_reason: str | None = None,
) -> None:
    """Apply a single section edit decision."""
    edit_agent.edit_section(
        section_name=section_name,
        proposal=proposal,
        item_name=item_name,
        special_instruction=special_instruction,
        rejection_reason=rejection_reason,
    )


# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------

def score_resume(
    edit_agent: EditAgent,
    parsed_jd: ParsedJD,
    app_id: int = 0,
    resume_id: int = 0,
) -> ResumeScore:
    """Score the current edit state against the JD across three dimensions."""
    state = edit_agent.editing_cycle.current_state

    latex_parts: list[str] = []
    for sec in FLAT_SECTIONS:
        sec_state = getattr(state, sec, None)
        if sec_state and sec_state.updated_section:
            latex_parts.append(sec_state.updated_section)
    for sec in ATOMIC_SECTIONS:
        for item in (getattr(state, sec, None) or []):
            if item and item.updated_section:
                latex_parts.append(item.updated_section)

    return score_resume_fn(
        app_id=app_id,
        resume_id=resume_id,
        resume_latex="\n\n".join(latex_parts),
        jd=parsed_jd,
    )


def format_score_feedback(score_result: ResumeScore) -> str:
    """
    Convert a ResumeScore into a human-readable special-instruction string
    for inject into generate_suggestions_with_score_feedback.
    """
    lines = [f"Overall score: {score_result.overall_score:.1f}/100\n"]
    for dim in score_result.dimension_scores:
        lines.append(f"{dim.dimension.value}: {dim.score:.1f}/100")
        lines.append(f"  Reasoning: {dim.reasoning}")
        for s in dim.suggestions[:3]:
            lines.append(f"  - {s}")
        lines.append("")
    if score_result.missing_keywords:
        kws = ", ".join(score_result.missing_keywords[:8])
        lines.append(f"Missing required keywords: {kws}")
        lines.append("Naturally incorporate these into the most relevant sections.")
    return "\n".join(lines)


# ------------------------------------------------------------------
# Preview build (on-demand)
# ------------------------------------------------------------------

def build_preview(
    edit_agent:       EditAgent,
    user_id:          int | None   = None,
    section_order:    list[str] | None = None,
    section_space:    float | None = None,
    subsection_space: float | None = None,
    font_size:        int | None   = None,
) -> str:
    """Assemble the current edit state into a compilable LaTeX string."""
    state = edit_agent.editing_cycle.current_state
    builder_input = ResumeBuilderInput(
        education=state.education.updated_section if state.education else "",
        achievements=state.achievements.updated_section if state.achievements else "",
        skills=state.skills.updated_section if state.skills else "",
        relevant_coursework=state.relevant_coursework.updated_section if state.relevant_coursework else "",
        experience=[item.updated_section for item in (state.experience or []) if item],
        projects=[item.updated_section for item in (state.projects or []) if item],
    )
    builder = ResumeBuilder(user_id=user_id, sections=builder_input, font_size=font_size)
    build_kwargs: dict = {}
    if section_order    is not None: build_kwargs["section_order"]    = section_order
    if section_space    is not None: build_kwargs["section_space"]    = section_space
    if subsection_space is not None: build_kwargs["subsection_space"] = subsection_space
    return builder.build(**build_kwargs)


# ------------------------------------------------------------------
# Collapse edit state → ParsedResume (for generation)
# ------------------------------------------------------------------

def collapse_edit_state_to_resume(
    edit_agent: EditAgent,
    personal_info: PersonalInfo,
    resume_id: int = 0,
) -> ParsedResume:
    """Convert the final ResumeEditState into a ParsedResume for generation."""
    state = edit_agent.editing_cycle.current_state

    def _flat(sec_name: str) -> SectionContent | None:
        sec = getattr(state, sec_name, None)
        return SectionContent(content_latex=sec.updated_section, content_text=None) if sec else None

    sections = ResumeSection(
        education=_flat("education"),
        achievements=_flat("achievements"),
        skills=_flat("skills"),
        relevant_coursework=_flat("relevant_coursework"),
        experience=[
            AtomicItem(name=item.item_name, role=None,
                       content_latex=item.updated_section, content_text="")
            for item in (state.experience or []) if item
        ],
        projects=[
            AtomicItem(name=item.item_name, role=None,
                       content_latex=item.updated_section, content_text="")
            for item in (state.projects or []) if item
        ],
    )
    return ParsedResume(
        resume_id=str(resume_id),
        resume_path="",
        personal_info=personal_info,
        resume_sections=sections,
    )


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------

def generate_output(
    parsed_jd: ParsedJD,
    resume_source: ParsedResume,
    generation_type: str,       # "coverletter" | "email" | "outreachmessage"
) -> str:
    """Generate a cover letter, HR email, or outreach message."""
    return generate_fn(resume=resume_source, jd=parsed_jd, type=generation_type)
