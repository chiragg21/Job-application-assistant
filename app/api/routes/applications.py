"""
app/api/routes/applications.py
-------------------------------
Application tracker CRUD.

  GET    /applications?user_id=1           — list all applications for user
  POST   /applications                     — create a new application
  PATCH  /applications/{id}                — update status / notes / dates
  DELETE /applications/{id}                — delete an application
  GET    /applications/{id}                — get single application
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.utils.sqlite_handler import SQLHandler
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/applications", tags=["Application Tracker"])

ApplicationStatus = Literal[
    "applied", "phone_screen", "technical", "final_round",
    "offer", "rejected", "withdrawn",
]

STATUS_ORDER: list[ApplicationStatus] = [
    "applied", "phone_screen", "technical", "final_round",
    "offer", "rejected", "withdrawn",
]


# ── request models ────────────────────────────────────────────────────────────

class CreateApplicationRequest(BaseModel):
    user_id:     int = 1
    company:     str
    role:        str
    status:      ApplicationStatus = "applied"
    job_id:      int | None = None
    resume_id:   int | None = None
    session_id:  int | None = None
    variant_id:  int | None = None
    notes:       str | None = None
    applied_at:  str | None = None
    follow_up_at: str | None = None


class PatchApplicationRequest(BaseModel):
    status:       ApplicationStatus | None = None
    notes:        str | None = None
    follow_up_at: str | None = None
    company:      str | None = None
    role:         str | None = None


# ── helpers ───────────────────────────────────────────────────────────────────

def _enrich(row: dict) -> dict:
    """Add display-friendly fields."""
    row["status_index"] = STATUS_ORDER.index(row["status"]) if row.get("status") in STATUS_ORDER else 0
    return row


# ── endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
def list_applications(user_id: int = Query(1)):
    db = SQLHandler()
    df = db.fetch_table_where("applications_tracker", filters={"user_id": user_id})
    rows = df.sort_values("applied_at", ascending=False).to_dict("records") if len(df) else []
    return [_enrich(r) for r in rows]


@router.get("/{app_id}")
def get_application(app_id: int):
    db = SQLHandler()
    row = db.fetch_one("applications_tracker", {"id": app_id})
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    return _enrich(row)


@router.post("", status_code=201)
def create_application(req: CreateApplicationRequest):
    db = SQLHandler()
    data = req.model_dump(exclude_none=True)
    # apply default applied_at if not provided
    if "applied_at" not in data:
        import time
        data["applied_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    row_id = db.add_one("applications_tracker", data)
    row = db.fetch_one("applications_tracker", {"id": row_id})
    return _enrich(row)


@router.patch("/{app_id}")
def patch_application(app_id: int, req: PatchApplicationRequest):
    db = SQLHandler()
    row = db.fetch_one("applications_tracker", {"id": app_id})
    if not row:
        raise HTTPException(status_code=404, detail="Application not found")
    updates = req.model_dump(exclude_none=True)
    if not updates:
        return _enrich(row)
    import time
    updates["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    db.update_data("applications_tracker", updates, {"id": app_id})
    updated = db.fetch_one("applications_tracker", {"id": app_id})
    return _enrich(updated)


@router.delete("/{app_id}", status_code=204)
def delete_application(app_id: int):
    db = SQLHandler()
    if not db.fetch_one("applications_tracker", {"id": app_id}):
        raise HTTPException(status_code=404, detail="Application not found")
    db.execute_raw("DELETE FROM applications_tracker WHERE id = :id", {"id": app_id})
