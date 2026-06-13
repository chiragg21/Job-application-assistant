# app/api/routes/jd.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import app.core.pipeline as pl

router = APIRouter(prefix="/jd", tags=["Job Description"])


class ParseJDRequest(BaseModel):
    jd_text: str
    user_id: int


class ParseJDResponse(BaseModel):
    job_id:    int
    parsed_jd: dict
    duplicate:      bool = False
    duplicate_type: str | None = None


class ExtractFromUrlRequest(BaseModel):
    url:     str
    user_id: int


@router.post("/parse", response_model=ParseJDResponse)
def parse_jd(req: ParseJDRequest):
    """
    Parse a raw job-description string.

    - Deduplicates via raw_hash and parsed_hash.
    - Stores structured output in SQLite and embeds chunks into ChromaDB.
    - Returns the job_id and structured ParsedJD dict.
    """
    try:
        job_id, parsed = pl.parse_jd(jd_text=req.jd_text, user_id=req.user_id)
        return ParseJDResponse(
            job_id=job_id,
            parsed_jd=parsed.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/extract-from-url", response_model=ParseJDResponse)
def extract_jd_from_url(req: ExtractFromUrlRequest):
    """
    Extract a job description from a URL and parse it.

    Fetches the page via Jina Reader (r.jina.ai) — free, no API key, handles
    JS-rendered pages (LinkedIn, Indeed, Glassdoor, Naukri, etc.).

    Equivalent to calling /jd/parse with the extracted text, including
    deduplication and ChromaDB embedding.
    """
    try:
        from app.core.jd_extractor import extract_jd_from_url as _extract
        jd_text = _extract(req.url)
        job_id, parsed = pl.parse_jd(jd_text=jd_text, user_id=req.user_id)
        return ParseJDResponse(job_id=job_id, parsed_jd=parsed.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
