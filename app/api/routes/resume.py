# app/api/routes/resume.py

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

import app.core.pipeline as pl
from app.utils import SQLHandler

router = APIRouter(prefix="/resume", tags=["Resume"])


class UserProfile(BaseModel):
    user_id:          int
    name:             str | None
    email:            str | None
    resume_count:     int
    latest_resume_id: int


class ResumeListItem(BaseModel):
    resume_id:   int
    name:        str | None
    email:       str | None
    resume_path: str
    created_at:  str | None


class ResumeParseResponse(BaseModel):
    resume_id:     int
    parsed_resume: dict


class PersonalInfoRequest(BaseModel):
    resume_path: str


class PersonalInfoResponse(BaseModel):
    name:     str | None
    email:    str | None
    phone:    str | None
    github:   str | None
    linkedin: str | None


class CreateUserRequest(BaseModel):
    name:         str
    email:        str | None = None
    phone:        str | None = None
    github:       str | None = None
    linkedin:     str | None = None
    current_role: str | None = None
    company:      str | None = None


class CreateUserResponse(BaseModel):
    user_id: int


class RetrieveRequest(BaseModel):
    job_id:    int
    parsed_jd: dict   # ParsedJD.model_dump()
    top_k:     int | None = None


class RetrieveResponse(BaseModel):
    ranked_items: list[dict]


@router.get("/users", response_model=list[UserProfile])
def list_users():
    """
    Return all distinct candidate profiles stored in the DB.

    Each entry aggregates all resumes for one user_id, returning the
    name / email from their most-recent resume and a count of how many
    resume versions exist for that user.
    """
    try:
        db   = SQLHandler()
        rows = db.execute_raw(
            """
            SELECT
                r.user_id,
                r.name,
                r.email,
                COUNT(*)          AS resume_count,
                MAX(r.id)         AS latest_resume_id
            FROM resumes r
            GROUP BY r.user_id
            ORDER BY r.user_id
            """
        )
        if not rows:
            return []
        return [
            UserProfile(
                user_id=int(row["user_id"]),
                name=row.get("name") or None,
                email=row.get("email") or None,
                resume_count=int(row.get("resume_count", 1)),
                latest_resume_id=int(row["latest_resume_id"]),
            )
            for row in rows
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/list", response_model=list[ResumeListItem])
def list_resumes(user_id: int = 1):
    """List all previously parsed resumes stored in the database for a user."""
    try:
        db   = SQLHandler()
        rows = db.fetch_table_where("resumes", filters={"user_id": user_id})
        if rows.empty:
            return []
        return [
            ResumeListItem(
                resume_id=int(row["id"]),
                name=row.get("name") or None,
                email=row.get("email") or None,
                resume_path=str(row.get("resume_path") or ""),
                created_at=str(row.get("created_at") or ""),
            )
            for _, row in rows.iterrows()
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/parse", response_model=ResumeParseResponse)
async def parse_resume(
    user_id: int = Form(...),
    file: UploadFile = File(..., description="LaTeX resume file (.tex)"),
):
    """
    Upload a LaTeX resume, parse it, and store it in SQLite + ChromaDB.

    - Accepts multipart/form-data with a .tex file.
    - Returns the resume_id and the structured ParsedResume dict.
    """
    if not file.filename.endswith(".tex"):
        raise HTTPException(status_code=400, detail="Only .tex files are accepted.")

    # Save the upload to a temp file so ResumeParser can open it
    with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        resume_id, parsed = pl.parse_resume(resume_path=tmp_path, user_id=user_id)
        return ResumeParseResponse(
            resume_id=resume_id,
            parsed_resume=parsed.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        tmp_path.unlink(missing_ok=True)


@router.post("/extract-info", response_model=PersonalInfoResponse)
def extract_personal_info(req: PersonalInfoRequest):
    """
    Extract personal info from a local .tex file path without writing to DB.
    Used to pre-populate the new-user form in the UI.
    """
    try:
        info = pl.extract_personal_info(req.resume_path)
        return PersonalInfoResponse(
            name=info.name     if info.name     not in ("", "Unknown") else None,
            email=info.email   if info.email    not in ("", "Unknown") else None,
            phone=info.phone   if info.phone    not in ("", "Unknown") else None,
            github=info.github  or None,
            linkedin=info.linkedin or None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/create-user", response_model=CreateUserResponse)
def create_user(req: CreateUserRequest):
    """Create a new candidate profile in the users table."""
    try:
        user_id = pl.create_user(
            name=req.name,
            email=req.email,
            phone=req.phone,
            github=req.github,
            linkedin=req.linkedin,
            current_role=req.current_role,
            company=req.company,
        )
        return CreateUserResponse(user_id=user_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve_for_job(req: RetrieveRequest):
    """
    Retrieve and rank the most relevant resume sections for a given job.

    - Runs two-stage CombMNZ + weighted CombSUM reranking.
    - Returns ranked_items ready for the item-selection step.
    """
    from app.models.jd import ParsedJD
    try:
        parsed_jd   = ParsedJD(**req.parsed_jd)
        kwargs      = {"top_k": req.top_k} if req.top_k else {}
        raw_results = pl.retrieve(job_id=req.job_id, parsed_jd=parsed_jd, **kwargs)
        ranked      = pl.rank_and_filter(raw_results)
        return RetrieveResponse(ranked_items=ranked)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
