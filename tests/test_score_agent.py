"""Tests for app/agents/score_agent.py"""
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.agents.score_agent import (
    _latex_to_text,
    _cosine_similarity,
    _weighted_score,
    _build_llm_score_prompt,
    _get_embedding,
    _score_keyword_match,
    _score_llm_dimensions,
    score,
)
from app.models.scoring import (
    DimensionScore, LLMScoreOutput, ResumeScore, ScoreDimension,
)


# ── Pure helpers ──────────────────────────────────────────────────────────────

class TestLatexToText:
    def test_strips_command_with_arg(self):
        assert _latex_to_text(r"\textbf{hello}") == "hello"

    def test_strips_bare_command(self):
        text = _latex_to_text(r"\item foo")
        assert "item" not in text
        assert "foo" in text

    def test_strips_braces(self):
        assert "{" not in _latex_to_text(r"\section{Skills}")

    def test_collapses_whitespace(self):
        result = _latex_to_text("a   b\t\tc")
        assert "  " not in result

    def test_empty_string(self):
        assert _latex_to_text("") == ""

    def test_plain_text_unchanged(self):
        assert _latex_to_text("Python Docker SQL") == "Python Docker SQL"


class TestCosineSimilarity:
    def test_identical_vectors_is_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_is_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_is_minus_one(self):
        v = np.array([1.0, 0.0])
        assert _cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_returns_zero(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 2.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


class TestWeightedScore:
    def test_equal_weights_averages_scores(self):
        w = {"ats_friendliness": 1/3, "keyword_match": 1/3, "resume_quality": 1/3}
        result = _weighted_score(90.0, 90.0, 90.0, w)
        assert result == pytest.approx(90.0, abs=0.1)

    def test_keyword_heavy_weights(self):
        w = {"ats_friendliness": 0.0, "keyword_match": 1.0, "resume_quality": 0.0}
        assert _weighted_score(0.0, 75.0, 0.0, w) == pytest.approx(75.0, abs=0.01)

    def test_result_is_rounded_to_two_decimals(self):
        w = {"ats_friendliness": 0.30, "keyword_match": 0.40, "resume_quality": 0.30}
        result = _weighted_score(80.0, 70.0, 60.0, w)
        assert result == round(result, 2)


class TestBuildLLMScorePrompt:
    def test_returns_two_strings(self, sample_jd):
        system_prompt, prompt = _build_llm_score_prompt("\\latex{}", sample_jd)
        assert isinstance(system_prompt, str)
        assert isinstance(prompt, str)

    def test_system_prompt_mentions_ats(self, sample_jd):
        system_prompt, _ = _build_llm_score_prompt("\\latex{}", sample_jd)
        assert "ats" in system_prompt.lower()

    def test_prompt_contains_required_skills(self, sample_jd):
        system_prompt, _ = _build_llm_score_prompt("\\latex{}", sample_jd)
        assert "Python" in system_prompt


# ── Functions that depend on the embedding function ──────────────────────────

class TestGetEmbedding:
    def test_returns_ndarray(self):
        fake_vector = [0.1, 0.2, 0.3]
        _get_embedding.cache_clear()
        with patch("app.agents.score_agent.get_ef") as mock_get_ef:
            mock_ef = MagicMock(return_value=[fake_vector])
            mock_get_ef.return_value = mock_ef
            result = _get_embedding("some text")
        # _get_embedding returns a tuple (LRU-cache friendly); convert for comparison
        result_arr = np.array(result)
        np.testing.assert_array_almost_equal(result_arr, np.array(fake_vector))

    def test_calls_ef_with_list(self):
        _get_embedding.cache_clear()
        with patch("app.agents.score_agent.get_ef") as mock_get_ef:
            mock_ef = MagicMock(return_value=[[0.0]])
            mock_get_ef.return_value = mock_ef
            _get_embedding("hello")
        mock_ef.assert_called_once_with(["hello"])


class TestScoreKeywordMatch:
    def _make_embedding_patch(self, resume_vec, jd_vec):
        call_count = {"n": 0}
        vecs = [resume_vec, jd_vec]

        def fake_embed(text):
            v = vecs[call_count["n"]]
            call_count["n"] += 1
            return v

        return fake_embed

    def test_all_skills_present_no_penalty(self, sample_jd):
        resume_text = "Python PyTorch Docker"   # all required skills present
        v = np.array([1.0, 0.0])

        with patch("app.agents.score_agent._get_embedding", return_value=v):
            score_val, missing = _score_keyword_match(resume_text, sample_jd)

        assert missing == []
        assert score_val >= 0

    def test_missing_skills_penalised(self, sample_jd):
        resume_text = "JavaScript Node.js"  # none of the required skills
        v = np.array([1.0, 0.0])

        with patch("app.agents.score_agent._get_embedding", return_value=v):
            score_val, missing = _score_keyword_match(resume_text, sample_jd)

        assert len(missing) == len(sample_jd.required_skills)
        assert score_val < 100.0

    def test_score_bounded_above_zero(self, sample_jd):
        v = np.array([0.0, 1.0])  # orthogonal → similarity = 0
        with patch("app.agents.score_agent._get_embedding", return_value=v):
            score_val, _ = _score_keyword_match("no match text", sample_jd)
        assert score_val >= 0.0


# ── LLM-dependent scoring ─────────────────────────────────────────────────────

class TestScoreLLMDimensions:
    def _make_llm_score_output(self):
        ats = DimensionScore(
            dimension=ScoreDimension.ATS_FRIENDLINESS,
            score=80,
            reasoning="Good keyword density.",
            suggestions=["Add more keywords"],
        )
        quality = DimensionScore(
            dimension=ScoreDimension.RESUME_QUALITY,
            score=75,
            reasoning="Decent impact verbs.",
            suggestions=["Quantify achievements"],
        )
        return LLMScoreOutput(
            ats_friendliness=ats,
            resume_quality=quality,
            overall_feedback="Solid resume overall.",
        )

    def test_returns_llm_score_output(self, sample_jd):
        parsed = self._make_llm_score_output()
        with patch("app.agents.score_agent.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=parsed)
            result = _score_llm_dimensions("\\latex{}", sample_jd)

        assert isinstance(result, LLMScoreOutput)
        assert result.ats_friendliness.score == 80
        assert result.resume_quality.score == 75

    def test_schema_mismatch_raises(self, sample_jd):
        with patch("app.agents.score_agent.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(schema_matched=False, parsed=None)
            result = _score_llm_dimensions("\\latex{}", sample_jd)
        assert result is None


# ── Full pipeline ─────────────────────────────────────────────────────────────

class TestScore:
    def _make_llm_score_output(self):
        ats = DimensionScore(
            dimension=ScoreDimension.ATS_FRIENDLINESS,
            score=80,
            reasoning="Good.",
            suggestions=[],
        )
        quality = DimensionScore(
            dimension=ScoreDimension.RESUME_QUALITY,
            score=70,
            reasoning="OK.",
            suggestions=[],
        )
        return LLMScoreOutput(
            ats_friendliness=ats,
            resume_quality=quality,
            overall_feedback="Good resume.",
        )

    def test_returns_resume_score(self, sample_jd):
        llm_out = self._make_llm_score_output()
        embedding = np.array([1.0, 0.0])
        with patch("app.agents.score_agent._get_embedding", return_value=embedding), \
             patch("app.agents.score_agent._score_llm_dimensions", return_value=llm_out), \
             patch("app.agents.score_agent._score_cache", {}):
            result = score(
                app_id=1,
                resume_id=1,
                resume_latex=r"\textbf{Python} Docker",
                jd=sample_jd,
            )

        assert isinstance(result, ResumeScore)
        assert result.app_id == 1
        assert result.resume_id == 1
        assert 0.0 <= result.overall_score <= 100.0

    def test_uses_default_llm_when_none(self, sample_jd):
        llm_out = self._make_llm_score_output()
        embedding = np.array([1.0, 0.0])

        with patch("app.agents.score_agent._get_embedding", return_value=embedding), \
             patch("app.agents.score_agent._score_llm_dimensions", return_value=llm_out) as mock_score, \
             patch("app.agents.score_agent._score_cache", {}):
            result = score(app_id=2, resume_id=2, resume_latex=r"\item Python", jd=sample_jd)

        assert isinstance(result, ResumeScore)
        mock_score.assert_called_once()

    def test_custom_weights_applied(self, sample_jd):
        """With keyword weight=1.0 and others=0.0, overall must equal keyword score."""
        llm_out = self._make_llm_score_output()
        custom_weights = {"ats_friendliness": 0.0, "keyword_match": 1.0, "resume_quality": 0.0}
        embedding = np.array([1.0, 0.0])

        with patch("app.agents.score_agent._get_embedding", return_value=embedding), \
             patch("app.agents.score_agent._score_llm_dimensions", return_value=llm_out), \
             patch("app.agents.score_agent._score_cache", {}):
            result = score(
                app_id=1, resume_id=1,
                resume_latex=r"\textbf{Python} Docker",
                jd=sample_jd,
                weights=custom_weights,
            )

        assert result.overall_score == pytest.approx(result.keyword_match_score, abs=0.01)
        assert result.weights == custom_weights

    def test_missing_keywords_populate_dimension_scores(self, sample_jd):
        """Required skills absent from resume appear in missing_keywords and in the KEYWORD_MATCH dimension suggestions."""
        llm_out = self._make_llm_score_output()
        embedding = np.array([1.0, 0.0])
        # resume text contains none of the required skills
        with patch("app.agents.score_agent._get_embedding", return_value=embedding), \
             patch("app.agents.score_agent._score_llm_dimensions", return_value=llm_out), \
             patch("app.agents.score_agent._score_cache", {}):
            result = score(
                app_id=1, resume_id=1,
                resume_latex=r"JavaScript Node.js Ruby",
                jd=sample_jd,
            )

        assert len(result.missing_keywords) == len(sample_jd.required_skills)
        keyword_dim = next(
            d for d in result.dimension_scores
            if d.dimension == ScoreDimension.KEYWORD_MATCH
        )
        assert len(keyword_dim.suggestions) > 0
