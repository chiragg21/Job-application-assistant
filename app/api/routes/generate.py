# app/api/routes/generate.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graph.generation_graph import generation_graph
from app.utils import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/generate", tags=["Generation"])

VALID_TYPES = {"coverletter", "email", "outreachmessage"}


class GenerateRequest(BaseModel):
    parsed_jd:          dict              # ParsedJD.model_dump()
    parsed_resume:      dict              # ParsedResume.model_dump()
    generation_types:   list[str]         # one or more of the VALID_TYPES
    custom_instruction: str | None = None
    no_jd:              bool       = False


class GenerateResponse(BaseModel):
    results: dict        # {generation_type: rendered_string}
    errors:  str | None  # any partial failures


@router.post("/", response_model=GenerateResponse)
def generate_documents(req: GenerateRequest):
    """
    Generate one or more output documents (cover letter, email, outreach
    message) in a single call using the generation LangGraph.

    generation_types accepts: "coverletter", "email", "outreachmessage"
    (case-insensitive, underscores stripped).
    """
    normalised = [t.lower().replace("_", "") for t in req.generation_types]
    unknown    = [t for t in normalised if t not in VALID_TYPES]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown generation type(s): {unknown}. "
                   f"Valid options: {sorted(VALID_TYPES)}",
        )

    try:
        result = generation_graph.invoke({
            "parsed_jd":          req.parsed_jd,
            "parsed_resume":      req.parsed_resume,
            "generation_types":   normalised,
            "custom_instruction": req.custom_instruction,
            "no_jd":              req.no_jd,
        })
        return GenerateResponse(
            results=result.get("results", {}),
            errors=result.get("error"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# -----------------------------------------------------------------------
# Convenience single-type endpoints
# -----------------------------------------------------------------------

class SingleGenerateRequest(BaseModel):
    parsed_jd:     dict
    parsed_resume: dict


@router.post("/cover-letter")
def generate_cover_letter(req: SingleGenerateRequest):
    """Shorthand: generate a cover letter only."""
    return generate_documents(GenerateRequest(
        parsed_jd=req.parsed_jd,
        parsed_resume=req.parsed_resume,
        generation_types=["coverletter"],
    ))


@router.post("/email")
def generate_email(req: SingleGenerateRequest):
    """Shorthand: generate an HR email only."""
    return generate_documents(GenerateRequest(
        parsed_jd=req.parsed_jd,
        parsed_resume=req.parsed_resume,
        generation_types=["email"],
    ))


@router.post("/outreach")
def generate_outreach(req: SingleGenerateRequest):
    """Shorthand: generate an outreach message only."""
    return generate_documents(GenerateRequest(
        parsed_jd=req.parsed_jd,
        parsed_resume=req.parsed_resume,
        generation_types=["outreachmessage"],
    ))
