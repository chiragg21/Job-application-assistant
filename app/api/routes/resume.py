# app/api/routes/resume.py

import json
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import Response
from pydantic import BaseModel

import app.core.pipeline as pl
from app.core.resume_builder import TemplateRenderer
from app.utils import SQLHandler

router = APIRouter(prefix="/resume", tags=["Resume"])

_renderer = TemplateRenderer()


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


_ACCEPTED_EXTENSIONS = {".tex", ".pdf", ".docx", ".doc", ".txt"}


@router.post("/parse", response_model=ResumeParseResponse)
async def parse_resume(
    user_id: int = Form(...),
    file: UploadFile = File(..., description="Resume file (.tex, .pdf, .docx, .txt)"),
):
    """
    Upload a resume, parse it, and store it in SQLite + ChromaDB.

    Accepts: .tex (fast regex path), .pdf, .docx, .txt (LLM fallback parser).
    Returns the resume_id and the structured ParsedResume dict.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ACCEPTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Accepted: {sorted(_ACCEPTED_EXTENSIONS)}",
        )

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
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


class ItemHistory(BaseModel):
    id:             int
    section_name:   str
    item_name:      str | None
    content_latex:  str | None
    is_latest:      int
    parent_item_id: int | None
    is_master:      int


class ResumeHistory(BaseModel):
    resume_id: int
    sections:  list[ItemHistory]


@router.get("/{resume_id}/history", response_model=ResumeHistory)
def get_resume_history(resume_id: int):
    """
    Return the full edit history for all items in a resume.

    Each row in resume_section_items is returned — current (is_latest=1) and
    superseded (is_latest=0) versions.  parent_item_id chains to the previous
    version of an item so the caller can reconstruct the full diff chain.
    """
    try:
        db   = SQLHandler()
        rows = db.execute_raw(
            """
            SELECT id, section_name, item_name, content_latex,
                   is_latest, parent_item_id, is_master
            FROM resume_section_items
            WHERE resume_id = :rid
            ORDER BY section_name, item_index, id
            """,
            {"rid": resume_id},
        )
        if not rows:
            return ResumeHistory(resume_id=resume_id, sections=[])
        return ResumeHistory(
            resume_id=resume_id,
            sections=[ItemHistory(**r) for r in rows],
        )
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


# ──────────────────────────────────────────────────────────────────────────────
# Section CRUD
# ──────────────────────────────────────────────────────────────────────────────

class SectionItem(BaseModel):
    id:            int
    section_name:  str
    item_name:     str | None
    content_latex: str | None
    is_latest:     int
    is_master:     int
    item_index:    int | None


class SectionItemUpdate(BaseModel):
    content_latex: str
    item_name:     str | None = None


@router.get("/{resume_id}/sections", response_model=list[SectionItem])
def list_sections(resume_id: int):
    """Return all latest section items for a resume (flat + atomic)."""
    try:
        db = SQLHandler()
        rows = db.execute_raw(
            """
            SELECT id, section_name, item_name, content_latex,
                   is_latest, is_master, item_index
            FROM resume_section_items
            WHERE resume_id = :rid AND is_latest = 1
            ORDER BY section_name, item_index
            """,
            {"rid": resume_id},
        ) or []
        return [SectionItem(**r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/{resume_id}/sections/{item_id}", response_model=SectionItem)
def update_section_item(resume_id: int, item_id: int, req: SectionItemUpdate):
    """
    Update a section item by creating a new versioned row.

    Marks the old row is_latest=0 and inserts a new row with
    is_latest=1 and parent_item_id pointing to the old row.
    """
    try:
        db      = SQLHandler()
        old_row = db.fetch_one("resume_section_items",
                               filters={"id": item_id, "resume_id": resume_id})
        if not old_row:
            raise HTTPException(status_code=404, detail=f"Item {item_id} not found.")
        if not old_row.get("is_latest"):
            raise HTTPException(status_code=409, detail="Item is already superseded (is_latest=0).")

        new_id = db.add_one("resume_section_items", {
            "resume_id":      resume_id,
            "section_id":     old_row.get("section_id"),
            "section_name":   old_row["section_name"],
            "item_name":      req.item_name if req.item_name is not None else old_row.get("item_name"),
            "role_title":     old_row.get("role_title") or "",
            "content_latex":  req.content_latex,
            "content_text":   req.content_latex,
            "item_index":     old_row.get("item_index", 0),
            "is_master":      0,
            "is_latest":      1,
            "parent_item_id": item_id,
        })
        db.update_data("resume_section_items", {"id": item_id}, {"is_latest": 0})

        new_row = db.fetch_one("resume_section_items", filters={"id": new_id})
        return SectionItem(
            id=new_id,
            section_name=new_row["section_name"],
            item_name=new_row.get("item_name"),
            content_latex=new_row.get("content_latex"),
            is_latest=1,
            is_master=0,
            item_index=new_row.get("item_index"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Layout CRUD
# ──────────────────────────────────────────────────────────────────────────────

class LayoutConfig(BaseModel):
    template_id:   str         = "classic"
    section_order: list[str]   | None = None
    font_size:     int         | None = None
    margin:        float       | None = None


class LayoutResponse(BaseModel):
    resume_id:     int
    template_id:   str
    section_order: list[str] | None
    font_size:     int | None
    margin:        float | None


@router.get("/{resume_id}/layout", response_model=LayoutResponse)
def get_layout(resume_id: int):
    """Return the stored layout config for a resume."""
    try:
        db  = SQLHandler()
        row = db.fetch_one("resume_layouts", filters={"resume_id": resume_id})
        if not row:
            raise HTTPException(status_code=404, detail="No layout stored for this resume.")
        order = None
        if row.get("section_order"):
            try:
                order = json.loads(row["section_order"])
            except Exception:
                pass
        return LayoutResponse(
            resume_id=resume_id,
            template_id=row.get("template_id", "classic"),
            section_order=order,
            font_size=row.get("font_size"),
            margin=row.get("margin"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.patch("/{resume_id}/layout", response_model=LayoutResponse)
def patch_layout(resume_id: int, req: LayoutConfig):
    """Create or update the layout config for a resume."""
    try:
        db       = SQLHandler()
        existing = db.fetch_one("resume_layouts", filters={"resume_id": resume_id})
        order_json = json.dumps(req.section_order) if req.section_order else None

        if existing:
            update_vals: dict = {"template_id": req.template_id}
            if order_json is not None:
                update_vals["section_order"] = order_json
            if req.font_size is not None:
                update_vals["font_size"] = req.font_size
            if req.margin is not None:
                update_vals["margin"] = req.margin
            db.update_data("resume_layouts", {"resume_id": resume_id}, update_vals)
        else:
            db.add_one("resume_layouts", {
                "resume_id":    resume_id,
                "template_id":  req.template_id,
                "section_order": order_json,
                "font_size":    req.font_size,
                "margin":       req.margin,
            })

        return LayoutResponse(
            resume_id=resume_id,
            template_id=req.template_id,
            section_order=req.section_order,
            font_size=req.font_size,
            margin=req.margin,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ──────────────────────────────────────────────────────────────────────────────
# Template list + render (template-engine based)
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/templates", response_model=list[str])
def list_templates():
    """Return the IDs of all available resume templates."""
    return _renderer.list_templates()


@router.get("/{resume_id}/render")
def render_resume(
    resume_id:        int,
    template_id:      str   | None = None,
    section_order:    str   | None = None,   # comma-separated
    font_size:        int   | None = None,
    margin:           float | None = None,
    format:           str          = "tex",  # "tex" | "latex"
):
    """
    Render a resume from DB data using the Jinja2 template engine.

    Returns the compiled LaTeX as a downloadable file.
    Use template_id, section_order, font_size, margin to override the stored
    layout (or the defaults if no layout has been saved yet).
    """
    try:
        order_list = section_order.split(",") if section_order else None
        latex = _renderer.render(
            resume_id=resume_id,
            template_id=template_id,
            section_order=order_list,
            font_size=font_size,
            margin=margin,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    short = f"resume_{resume_id}"
    return Response(
        content=latex.encode("utf-8"),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{short}.tex"'},
    )
