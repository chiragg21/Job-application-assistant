# app/graph/state.py
#
# Shared TypedDict state definitions for all LangGraph graphs.
# All values must be JSON-serialisable so the MemorySaver checkpointer
# can persist them between steps.

from __future__ import annotations
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class EditGraphState(TypedDict, total=False):
    # ------------------------------------------------------------------ #
    # Inputs (set once by the caller before the graph starts)             #
    # ------------------------------------------------------------------ #
    user_id:             int
    jd_text:             str          # raw job-description text
    resume_path:         str          # path to the LaTeX resume file (optional if existing_resume_id given)
    existing_resume_id:  int | None   # use a stored DB resume instead of uploading a file

    # ------------------------------------------------------------------ #
    # Stage 1 — JD parsing                                                #
    # ------------------------------------------------------------------ #
    job_id:    int
    parsed_jd: dict           # ParsedJD.model_dump()

    # ------------------------------------------------------------------ #
    # Stage 2 — Resume parsing                                            #
    # ------------------------------------------------------------------ #
    resume_id:     int
    parsed_resume: dict       # ParsedResume.model_dump()

    # ------------------------------------------------------------------ #
    # Stage 3 — Retrieval                                                 #
    # ------------------------------------------------------------------ #
    ranked_items: list[dict]  # output of rank_and_filter()

    # ------------------------------------------------------------------ #
    # Human-in-the-loop 1 — item selection                                #
    # After interrupt the client sends back selected_items.               #
    # ------------------------------------------------------------------ #
    selected_items: list[dict]   # subset of ranked_items chosen by user

    # ------------------------------------------------------------------ #
    # Stage 4 — Edit agent                                                #
    # The ResumeEditCycle is serialised via model_dump() so the           #
    # checkpointer can persist it.                                        #
    # ------------------------------------------------------------------ #
    edit_cycle_dict: dict | None    # ResumeEditCycle.model_dump()
    cycle_id:        int            # monotonically increasing session id

    # ------------------------------------------------------------------ #
    # Section review loop                                                  #
    # sections_pending is the queue of {section, item} pairs still to     #
    # review.  Each interrupt pops one entry, the user replies, and the   #
    # graph loops back until the queue is empty.                          #
    # ------------------------------------------------------------------ #
    sections_pending:  list[dict]   # [{"section": str, "item": str|None}]
    sections_reviewed: list[dict]   # stack of sections already reviewed (for back-navigation)
    current_review:    dict | None  # the entry currently being reviewed

    # ------------------------------------------------------------------ #
    # Scoring                                                              #
    # ------------------------------------------------------------------ #
    score_result:   dict | None    # ResumeScore.model_dump()
    score_feedback: str | None     # formatted string for LLM instruction

    # ------------------------------------------------------------------ #
    # Generation results                                                   #
    # ------------------------------------------------------------------ #
    generation_results: dict       # {"coverletter": str, "email": str, ...}

    # ------------------------------------------------------------------ #
    # Session-level instructions                                           #
    # ------------------------------------------------------------------ #
    custom_instruction: str | None   # applied to all LLM edit calls this session

    # ------------------------------------------------------------------ #
    # Control / diagnostics                                                #
    # ------------------------------------------------------------------ #
    stage:  str           # human-readable current stage label
    error:  str | None    # last error message if a node failed


class GenerationGraphState(TypedDict, total=False):
    # ------------------------------------------------------------------ #
    # Inputs                                                               #
    # ------------------------------------------------------------------ #
    parsed_jd:     dict   # ParsedJD.model_dump()
    parsed_resume: dict   # ParsedResume.model_dump()

    # Which outputs to generate (list of generation_type strings)
    generation_types: list[str]   # e.g. ["coverletter", "email", "outreachmessage"]

    # Optional generation modifiers
    custom_instruction: str | None   # free-text extra instruction injected into every prompt
    no_jd:              bool         # True → cold-outreach mode, JD is ignored

    # ------------------------------------------------------------------ #
    # Outputs                                                              #
    # ------------------------------------------------------------------ #
    results: dict         # {generation_type: rendered_string}
    error:   str | None
