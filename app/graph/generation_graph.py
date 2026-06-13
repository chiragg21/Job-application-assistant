# app/graph/generation_graph.py
#
# Simple LangGraph for batch document generation.
# No human-in-the-loop — runs to completion in one invoke() call.
#
# Input state:
#   parsed_jd       — ParsedJD.model_dump()
#   parsed_resume   — ParsedResume.model_dump()
#   generation_types — list of "coverletter" | "email" | "outreachmessage"
#
# Output state:
#   results  — {generation_type: rendered_string}

from __future__ import annotations

import traceback
import concurrent.futures

from langgraph.graph import StateGraph, END

from app.graph.state import GenerationGraphState
from app.models.jd import ParsedJD
from app.models.resume import ParsedResume
import app.core.pipeline as pl
from app.utils import get_logger

log = get_logger(__name__)

VALID_TYPES = {"coverletter", "email", "outreachmessage"}


# ======================================================================
# Node: generate_all
# ======================================================================

def node_generate_all(state: GenerationGraphState) -> dict:
    """
    Iterate over requested generation_types and call generate_output
    for each.  Failures for individual types are captured in the result
    dict rather than raising so the node never blocks the whole graph.
    """
    parsed_jd     = ParsedJD(**state["parsed_jd"])
    parsed_resume = ParsedResume(**state["parsed_resume"])
    gen_types     = [t.lower().replace("_", "") for t in state.get("generation_types", [])]

    results: dict[str, str] = {}
    errors:  list[str]      = []

    custom_instruction = state.get("custom_instruction") or None
    no_jd              = bool(state.get("no_jd", False))

    valid_types   = [t for t in gen_types if t in VALID_TYPES]
    invalid_types = [t for t in gen_types if t not in VALID_TYPES]
    for t in invalid_types:
        errors.append(f"Unknown generation type: '{t}'")

    def _generate_one(gen_type: str) -> tuple[str, str | None]:
        """Returns (gen_type, result_or_None). Errors are returned as (gen_type, None)."""
        try:
            log.info("[gen_graph] generating %s (no_jd=%s)", gen_type, no_jd)
            return gen_type, pl.generate_output(
                parsed_jd=parsed_jd,
                resume_source=parsed_resume,
                generation_type=gen_type,
                custom_instruction=custom_instruction,
                no_jd=no_jd,
            )
        except Exception:
            log.error("[gen_graph] %s failed", gen_type, exc_info=True)
            return gen_type, None

    # Run all valid generation types in parallel — same number of LLM calls,
    # wall-clock time reduced to the slowest single generation.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(valid_types) or 1) as pool:
        futures = {pool.submit(_generate_one, t): t for t in valid_types}
        for future in concurrent.futures.as_completed(futures):
            gen_type, result = future.result()
            if result is not None:
                results[gen_type] = result
            else:
                errors.append(f"{gen_type}: generation failed (see logs)")

    return {
        "results": results,
        "error":   "\n---\n".join(errors) if errors else None,
    }


# ======================================================================
# Build and compile
# ======================================================================

def build_generation_graph() -> StateGraph:
    g = StateGraph(GenerationGraphState)
    g.add_node("generate_all", node_generate_all)
    g.set_entry_point("generate_all")
    g.add_edge("generate_all", END)
    return g


generation_graph = build_generation_graph().compile()
