# app/api/routes/score.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.agents.score_agent import score as _score
from app.models.jd import ParsedJD
from app.utils import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/score", tags=["Scoring"])


class ScoreRequest(BaseModel):
    app_id:        int = 0
    resume_id:     int = 0
    resume_latex:  str          # raw LaTeX string
    parsed_jd:     dict         # ParsedJD.model_dump()
    weights:       dict | None = None   # override default dimension weights


class ScoreResponse(BaseModel):
    app_id:                int
    resume_id:             int
    overall_score:         float
    ats_friendliness_score: float
    keyword_match_score:   float
    resume_quality_score:  float
    dimension_scores:      list[dict]
    overall_feedback:      str
    missing_keywords:      list[str]
    weights:               dict


@router.post("/", response_model=ScoreResponse)
def score_resume(req: ScoreRequest):
    """
    Score a LaTeX resume against a parsed job description.

    Returns a weighted composite score across three dimensions:
      - keyword_match    (40% by default) — embedding cosine similarity
      - ats_friendliness (30%)            — LLM evaluation of structure/format
      - resume_quality   (30%)            — LLM evaluation of bullet quality

    Pass `weights` to override the defaults.
    """
    try:
        parsed_jd = ParsedJD(**req.parsed_jd)
        result    = _score(
            app_id=req.app_id,
            resume_id=req.resume_id,
            resume_latex=req.resume_latex,
            jd=parsed_jd,
            weights=req.weights,
        )
        return ScoreResponse(**result.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
