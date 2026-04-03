# data_models/score_dm.py

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class ScoreDimension(str, Enum):
    ATS_FRIENDLINESS = "ats_friendliness"
    KEYWORD_MATCH    = "keyword_match"
    RESUME_QUALITY   = "resume_quality"


class DimensionScore(BaseModel):
    dimension: ScoreDimension
    score: float = Field(..., ge=0, le=100, description="score out of 100")
    reasoning: str
    suggestions: List[str] = Field(default_factory=list)


class LLMScoreOutput(BaseModel):
    ats_friendliness: DimensionScore
    resume_quality: DimensionScore
    overall_feedback: str


class ResumeScore(BaseModel):
    app_id: int
    resume_id: int

    # Individual dimensions
    ats_friendliness_score: float
    keyword_match_score: float         # from embeddings
    resume_quality_score: float        # from LLM

    # Weighted composite
    overall_score: float

    # Detail
    dimension_scores: List[DimensionScore]
    overall_feedback: str
    missing_keywords: List[str] = Field(default_factory=list)

    # Weights used (from config)
    weights: dict = Field(default_factory=dict)