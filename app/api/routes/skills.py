"""
app/api/routes/skills.py
-------------------------
Skill gap analysis.

  GET /skills/gaps?user_id=1  — ranked gaps: skills required across all JDs
                                 the user has processed, that they lack.

Algorithm:
  1. Fetch all job_ids linked to edit sessions for this user.
  2. For each job, load required_skills from the jobs table (jd_parsed JSON).
  3. Load the user's resume skills from the latest resume_sections content.
  4. Count frequency of each required skill across all JDs.
  5. Subtract skills the user already has.
  6. Return sorted by frequency desc (most-needed gap first).
"""

from __future__ import annotations

import json
import re
from collections import Counter

from fastapi import APIRouter, Query

from app.utils.sqlite_handler import SQLHandler
from app.utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/skills", tags=["Skills"])


def _extract_text_skills(skills_latex: str) -> set[str]:
    """Pull skill tokens from a LaTeX skills section (best-effort)."""
    # Strip LaTeX macros, grab comma/newline/bullet-separated tokens
    text = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", skills_latex)
    text = re.sub(r"[\\{}]", " ", text)
    tokens: set[str] = set()
    for part in re.split(r"[,\n•·|/]", text):
        t = part.strip().lower()
        if 2 < len(t) < 60:
            tokens.add(t)
    return tokens


@router.get("/gaps")
def skill_gaps(user_id: int = Query(1), top_k: int = Query(30, ge=1, le=100)):
    db = SQLHandler()

    # 1. All jobs the user has processed (via edit_sessions)
    sessions_df = db.fetch_table_where(
        "edit_sessions",
        columns=["job_id"],
        filters={"user_id": user_id},
    )
    job_ids = list({int(r) for r in sessions_df["job_id"].dropna().tolist()}) if len(sessions_df) else []

    if not job_ids:
        return {"gaps": [], "total_jds": 0, "user_skills_count": 0}

    # 2. Load required skills from each job's jd_parsed
    required_counter: Counter = Counter()
    for jid in job_ids:
        row = db.fetch_one("jobs", {"id": jid}, columns=["jd_parsed"])
        if not row or not row.get("jd_parsed"):
            continue
        try:
            parsed = json.loads(row["jd_parsed"])
        except (json.JSONDecodeError, TypeError):
            continue
        # jd_parsed has required_skills as a list
        for skill in parsed.get("required_skills", []):
            required_counter[skill.strip().lower()] += 1
        for skill in parsed.get("nice_to_have_skills", []):
            required_counter[skill.strip().lower()] += 0.5   # half-weight

    # 3. User's current skills from latest resume
    user_skills: set[str] = set()
    resumes_df = db.fetch_table_where(
        "resumes",
        columns=["id"],
        filters={"user_id": user_id},
    )
    if len(resumes_df):
        latest_rid = int(resumes_df.iloc[-1]["id"])
        skills_row = db.fetch_one("resume_sections", {
            "resume_id": latest_rid, "section_name": "skills",
        })
        if skills_row and skills_row.get("content_latex"):
            user_skills = _extract_text_skills(skills_row["content_latex"])

    # 4. Compute gaps
    gaps = []
    for skill, freq in required_counter.most_common():
        if skill not in user_skills:
            gaps.append({
                "skill":     skill,
                "frequency": round(float(freq)),
                "jd_count":  sum(1 for jid in job_ids
                                 if _job_requires_skill(db, jid, skill)),
            })
        if len(gaps) >= top_k:
            break

    return {
        "gaps":              gaps,
        "total_jds":         len(job_ids),
        "user_skills_count": len(user_skills),
    }


def _job_requires_skill(db: SQLHandler, job_id: int, skill: str) -> bool:
    row = db.fetch_one("jobs", {"id": job_id}, columns=["jd_parsed"])
    if not row or not row.get("jd_parsed"):
        return False
    try:
        parsed = json.loads(row["jd_parsed"])
    except (json.JSONDecodeError, TypeError):
        return False
    all_skills = [s.strip().lower() for s in parsed.get("required_skills", [])]
    return skill in all_skills
