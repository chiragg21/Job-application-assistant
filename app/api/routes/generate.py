# app/api/routes/generate.py

import hashlib
import json
import os
import re

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.graph.generation_graph import generation_graph
from app.utils import get_logger, SQLHandler

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


# -----------------------------------------------------------------------
# Interview prep
# -----------------------------------------------------------------------

class InterviewPrepRequest(BaseModel):
    resume_id:          int
    job_id:             int
    custom_instruction: str | None = None


@router.post("/interview-prep")
def generate_interview_prep(req: InterviewPrepRequest):
    """
    Generate 5–10 likely interview questions + talking points grounded in the
    user's actual resume experience vs the JD.
    Cached by sha256(resume_id + job_id).
    """
    import google.generativeai as genai

    cache_key = hashlib.sha256(
        f"interview:{req.resume_id}:{req.job_id}".encode()
    ).hexdigest()

    db = SQLHandler()
    cached = db.fetch_one("generated_outputs", {"content_hash": cache_key})
    if cached:
        return {"questions": json.loads(cached["content_text"])}

    # Load resume latex and JD from DB
    resume_row = db.fetch_one("resumes", {"id": req.resume_id})
    job_row    = db.fetch_one("jobs",    {"id": req.job_id})
    if not resume_row or not job_row:
        raise HTTPException(status_code=404, detail="Resume or job not found")

    jd_text      = job_row.get("jd_text", "")
    resume_latex = resume_row.get("resume_path", "")  # content stored in sections

    # Pull all section content
    sections_df = db.fetch_table_where(
        "resume_sections",
        filters={"resume_id": req.resume_id},
    )
    resume_text = "\n\n".join(
        f"{r['section_name']}:\n{r.get('content_latex','')}"
        for _, r in sections_df.iterrows()
    ) if len(sections_df) else ""

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="No LLM API key configured")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))

    instruction = f"\nExtra context: {req.custom_instruction}" if req.custom_instruction else ""
    prompt = f"""You are an expert interview coach.

Job Description:
{jd_text[:3000]}

Candidate Resume:
{resume_text[:3000]}
{instruction}

Generate 8 interview questions the candidate is likely to face, grounded in their
specific resume experience vs this JD.  For each question include:
  - The question itself
  - A talking-point hint (1–2 sentences: what aspect of the candidate's background to highlight)

Return JSON only:
[
  {{"question": "...", "talking_point": "..."}},
  ...
]"""

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        questions = json.loads(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    # Cache result
    try:
        db.add_one("generated_outputs", {
            "application_id": 0,
            "output_type":    "interview_prep",
            "content_hash":   cache_key,
            "content_text":   json.dumps(questions),
            "is_draft":       0,
        })
    except Exception:
        pass  # cache write failure is non-fatal

    return {"questions": questions}


# -----------------------------------------------------------------------
# LinkedIn profile generator
# -----------------------------------------------------------------------

class LinkedInRequest(BaseModel):
    resume_id:   int
    target_role: str
    custom_instruction: str | None = None


@router.post("/linkedin")
def generate_linkedin(req: LinkedInRequest):
    """
    Generate LinkedIn About / Headline / Experience bullets tailored to the
    given target role.
    """
    import google.generativeai as genai

    db = SQLHandler()
    sections_df = db.fetch_table_where("resume_sections", filters={"resume_id": req.resume_id})
    resume_text = "\n\n".join(
        f"{r['section_name']}:\n{r.get('content_latex','')}"
        for _, r in sections_df.iterrows()
    ) if len(sections_df) else ""

    if not resume_text:
        raise HTTPException(status_code=404, detail="Resume sections not found")

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="No LLM API key configured")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))

    instruction = f"\nExtra: {req.custom_instruction}" if req.custom_instruction else ""
    prompt = f"""You are a LinkedIn profile writing expert.

Candidate Resume:
{resume_text[:3000]}

Target Role: {req.target_role}
{instruction}

Write optimised LinkedIn profile sections for the candidate targeting this role.

Return JSON only:
{{
  "headline": "...",
  "about": "...",
  "experience_bullets": [
    {{"company": "...", "bullets": ["...", "..."]}},
    ...
  ]
}}"""

    try:
        resp = model.generate_content(prompt)
        text = resp.text.strip()
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        result = json.loads(text)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}")

    return result
