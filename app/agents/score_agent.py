# src/agents/score_agent.py
import re
import numpy as np
from typing import Optional
from infrakit.llm import LLMClient
from chromadb.utils import embedding_functions
from app.utils.llm import llm as _llm, Prompt
from app.utils.logger import get_logger
from app.models.scoring import (
    ResumeScore, LLMScoreOutput, DimensionScore, ScoreDimension
)
from app.models.jd import ParsedJD
from config.config import get_config_dict

logger = get_logger(__name__)
config = get_config_dict()

_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.get('rag_config', {}).get('embedding_model', 'all-mpnet-base-v2')
)

# Weights for composite score — override in config.ini if needed
DEFAULT_WEIGHTS = config.get('score_weights', {
    'ats_friendliness': 0.35,
    'keyword_match':    0.20,
    'resume_quality':   0.45,
})


def _latex_to_text(latex: str) -> str:
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', latex)  # \cmd{content} → content
    text = re.sub(r'\\[a-zA-Z]+', ' ', text)                 # remaining commands
    text = re.sub(r'[{}]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _get_embedding(text: str) -> np.ndarray:
    return np.array(_ef([text])[0])


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def _skill_found(skill: str, resume_lower: str) -> bool:
    """
    Flexible skill matching:
    - Direct substring match (e.g. "python" in resume)
    - For multi-word skills, accept if ANY significant word (>3 chars) matches
      so "machine learning" matches even if resume says "ML" or "deep learning"
      — the semantic pass will handle degree of match
    """
    skill = skill.lower().strip()
    if skill in resume_lower:
        return True
    words = [w for w in skill.split() if len(w) > 3]
    return bool(words) and any(w in resume_lower for w in words)


def _score_keyword_match(resume_text: str, jd: ParsedJD) -> tuple[float, list[str]]:
    """
    Hybrid keyword score:
    - 60 % coverage  : fraction of required_skills found in resume (flexible match)
    - 40 % semantic  : embedding cosine-similarity of resume vs full JD text

    Much more reliable than cosine-similarity alone, which tends to produce
    uniformly low scores regardless of actual keyword presence.
    """
    required     = [kw for kw in (jd.required_skills or []) if kw.strip()]
    resume_lower = resume_text.lower()

    if required:
        missing  = [kw for kw in required if not _skill_found(kw, resume_lower)]
        found    = len(required) - len(missing)
        coverage = found / len(required) * 100
    else:
        missing  = []
        coverage = 80.0   # no required skills listed → neutral

    jd_text = " ".join([
        " ".join(jd.required_skills or []),
        " ".join(jd.responsibilities or []),
        " ".join(jd.nice_to_have_skills or []),
    ])
    resume_emb    = _get_embedding(resume_text)
    jd_emb        = _get_embedding(jd_text)
    semantic      = _cosine_similarity(resume_emb, jd_emb) * 100

    score = round(0.6 * coverage + 0.4 * semantic, 2)

    logger.info(
        f"[ScoreAgent] Keyword match: coverage={coverage:.1f} "
        f"semantic={semantic:.1f} final={score} missing={missing}"
    )
    return score, missing


def _build_llm_score_prompt(resume_latex: str, jd: ParsedJD) -> tuple[str, str]:
    system_prompt = f"""
You are an expert ATS analyst and senior technical recruiter.
You will be given a resume in LaTeX format and a job description.
Score the resume on two dimensions, each out of 100:

1. ats_friendliness — evaluate:
   - Keyword density relative to the JD
   - Use of standard section names (Experience, Education, Skills etc.)
   - Absence of tables, graphics, multi-column layouts that confuse ATS parsers
   - Consistent formatting and date formats
   - Appropriate use of bold for keywords

2. resume_quality — evaluate independently of the JD:
   - Quantified achievements (numbers, percentages, impact)
   - Action verb strength
   - Clarity and conciseness of bullet points
   - No redundancy or filler phrases
   - Appropriate length and density

For each dimension provide a score, clear reasoning, and 2-3 specific improvement suggestions.
Also provide a short overall_feedback paragraph.

## JD CONTEXT
Required Skills: {jd.required_skills}
Responsibilities: {jd.responsibilities}
"""

    prompt = f"""
## RESUME (LaTeX):
{resume_latex}

## OUTPUT FORMAT (JSON only, no markdown):
{LLMScoreOutput.model_json_schema()}
"""
    return system_prompt, prompt


def _score_llm_dimensions(llm: LLMClient, resume_latex: str, jd: ParsedJD) -> LLMScoreOutput:
    system_prompt, prompt = _build_llm_score_prompt(resume_latex, jd)
    response = llm.generate(
        Prompt(system=system_prompt, user=prompt),
        provider="gemini",
        response_model=LLMScoreOutput,
    )
    if not response.schema_matched or response.parsed is None:
        raise ValueError("LLM response did not match LLMScoreOutput schema")
    result: LLMScoreOutput = response.parsed
    logger.info(
        f"[ScoreAgent] LLM scores — "
        f"ATS={result.ats_friendliness.score}, "
        f"Quality={result.resume_quality.score}"
    )
    return result


def _weighted_score(ats: float, keyword: float, quality: float, weights: dict) -> float:
    return round(
        ats     * weights['ats_friendliness'] +
        keyword * weights['keyword_match']    +
        quality * weights['resume_quality'],
        2,
    )


def score(
    app_id: int,
    resume_id: int,
    resume_latex: str,
    jd: ParsedJD,
    llm: Optional[LLMClient] = None,
    weights: Optional[dict] = None,
) -> ResumeScore:
    """
    Scores a resume across three dimensions:

    1. keyword_match    — embedding cosine similarity (resume text vs JD text)
    2. ats_friendliness — LLM judges formatting, keyword density, structure
    3. resume_quality   — LLM judges impact, clarity, quantification (JD-agnostic)

    Produces a ResumeScore with per-dimension breakdowns and actionable suggestions.
    """
    if llm is None:
        llm = _llm
    if weights is None:
        weights = DEFAULT_WEIGHTS

    resume_text = _latex_to_text(resume_latex)

    keyword_score, missing_keywords = _score_keyword_match(resume_text, jd)
    llm_output = _score_llm_dimensions(llm, resume_latex, jd)

    overall = _weighted_score(
        ats     = llm_output.ats_friendliness.score,
        keyword = keyword_score,
        quality = llm_output.resume_quality.score,
        weights = weights,
    )

    result = ResumeScore(
        app_id                 = app_id,
        resume_id              = resume_id,
        ats_friendliness_score = llm_output.ats_friendliness.score,
        keyword_match_score    = keyword_score,
        resume_quality_score   = llm_output.resume_quality.score,
        overall_score          = overall,
        dimension_scores       = [
            llm_output.ats_friendliness,
            llm_output.resume_quality,
            DimensionScore(
                dimension  = ScoreDimension.KEYWORD_MATCH,
                score      = keyword_score,
                reasoning  = f"Embedding cosine similarity with {len(missing_keywords)} missing required keywords.",
                suggestions= [f"Add missing keyword: '{kw}'" for kw in missing_keywords[:3]],
            ),
        ],
        overall_feedback  = llm_output.overall_feedback,
        missing_keywords  = missing_keywords,
        weights           = weights,
    )

    logger.info(f"[ScoreAgent] Overall score={overall} | app_id={app_id}")
    return result
