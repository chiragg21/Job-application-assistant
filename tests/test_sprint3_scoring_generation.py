"""
Sprint 3 -- Scoring, generation, and session history tests.

Covers:
  - ScoreAgent: caching, keyword_match, score() contract
  - GeneratorAgent: render helpers, prompt builders, generate() for each type
  - pipeline.format_score_feedback
  - /edit/{tid}/diff endpoint
  - /edit/sessions/{user_id} listing endpoint
  - /edit/{tid}/state endpoint
  - /edit/{tid}/preview endpoint

All LLM, DB, and ChromaDB calls are mocked.
"""
import json
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def sample_resume():
    from app.models.resume import ParsedResume, PersonalInfo, ResumeSection, SectionContent
    return ParsedResume(
        resume_id="1",
        resume_path="data/resumes/resume_v1.tex",
        personal_info=PersonalInfo(
            name="Chirag Garg", email="c@x.com", phone="+91-123",
            github="https://github.com/cg", linkedin="https://linkedin.com/in/cg",
        ),
        resume_sections=ResumeSection(
            skills=SectionContent(
                content_latex=r"\item Python \item FastAPI",
                content_text="Python FastAPI",
            ),
        ),
    )


@pytest.fixture
def sample_jd_fixture():
    from app.models.jd import ParsedJD
    return ParsedJD(
        company="DeepMind", role="ML Engineer", seniority_level="mid",
        location="London", is_remote=True,
        about_company="World-class AI lab.",
        about_job="Build training infra.",
        responsibilities=["Design ML pipelines", "Deploy models"],
        required_skills=["Python", "PyTorch"],
        nice_to_have_skills=["LangChain"],
        perks="Equity + health", others="", raw_text="",
    )


@pytest.fixture
def edit_client():
    from app.api.routes.edit import router
    app = FastAPI()
    app.include_router(router)

    # Build mock editing cycle for diff route
    mock_section = MagicMock()
    mock_section.updated_section = ""
    mock_resume_state = MagicMock()
    mock_resume_state.get_section.return_value = mock_section
    mock_resume_state.experience = []
    mock_resume_state.projects = []

    mock_cycle_node = MagicMock()
    mock_cycle_node.resume_state = mock_resume_state

    mock_cycle = MagicMock()
    mock_cycle.nodes = {0: mock_cycle_node}
    mock_cycle.current_state = mock_resume_state

    mock_graph_state = MagicMock()
    mock_graph_state.values = {
        "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
        "user_id": 1, "resume_id": 1,
        "parsed_jd": {
            "company": "X", "role": "SWE", "seniority_level": "mid",
            "location": "", "is_remote": False, "about_company": "",
            "about_job": "", "responsibilities": [],
            "required_skills": [], "nice_to_have_skills": [], "perks": "", "others": "",
        },
        "stage": "scored", "error": None,
        "score_result": None, "generation_results": None,
    }
    mock_graph_state.tasks = []

    mock_graph = MagicMock()
    mock_graph.get_state.return_value = mock_graph_state

    mock_agent = MagicMock()
    mock_agent.editing_cycle = mock_cycle

    mock_pl = MagicMock()
    mock_pl.build_preview.return_value = r"\documentclass{article}"

    with patch("app.api.routes.edit.edit_graph", mock_graph), \
         patch("app.api.routes.edit._rebuild_agent", return_value=mock_agent), \
         patch("app.api.routes.edit.pl", mock_pl), \
         patch("app.api.routes.edit.SQLHandler") as mock_sql, \
         patch("app.api.routes.edit.register_thread"):
        yield TestClient(app), mock_graph, mock_agent, mock_pl, mock_sql


class TestGeneratorRenderHelpers:
    def test_render_email_returns_string(self):
        from app.agents.generator_agent import render_email
        from app.models.generation import Email
        email = Email(subject="Application", greeting="Hi,", body=["Para 1"], closing="Best,", signature="CG")
        assert isinstance(render_email(email), str)

    def test_render_email_contains_subject(self):
        from app.agents.generator_agent import render_email
        from app.models.generation import Email
        email = Email(subject="ML Engineer Role", greeting="Dear HM,", body=["Applying."], closing="Regards,", signature="CG")
        assert "ML Engineer Role" in render_email(email)

    def test_render_email_contains_body(self):
        from app.agents.generator_agent import render_email
        from app.models.generation import Email
        email = Email(subject="Sub", greeting="Hi,", body=["Unique body paragraph XYZ."], closing="Best,", signature="CG")
        assert "Unique body paragraph XYZ" in render_email(email)

    def test_render_email_from_dict(self):
        from app.agents.generator_agent import render_email
        d = {"subject": "App", "greeting": "Hi,", "body": ["Hello."], "closing": "Best,", "signature": "CG"}
        assert isinstance(render_email(d), str)

    def test_render_cover_letter_returns_string(self):
        from app.agents.generator_agent import render_cover_letter
        from app.models.generation import CoverLetter
        cl = CoverLetter(greeting="Dear HM,", opening_paragraph="Excited.", body_paragraphs=["BG."], closing_paragraph="LF.", closing="Sincerely,", signature="CG")
        assert isinstance(render_cover_letter(cl), str)

    def test_render_cover_letter_contains_opening(self):
        from app.agents.generator_agent import render_cover_letter
        from app.models.generation import CoverLetter
        cl = CoverLetter(greeting="Hi,", opening_paragraph="UniqueOpener123", body_paragraphs=[], closing_paragraph="TY.", closing="Best,", signature="CG")
        assert "UniqueOpener123" in render_cover_letter(cl)

    def test_render_cover_letter_contains_greeting(self):
        from app.agents.generator_agent import render_cover_letter
        from app.models.generation import CoverLetter
        cl = CoverLetter(greeting="Dear Hiring Manager,", opening_paragraph="Excited.", body_paragraphs=[], closing_paragraph="TY.", closing="Best,", signature="CG")
        assert "Dear Hiring Manager" in render_cover_letter(cl)

    def test_render_outreach_returns_full_message(self):
        from app.agents.generator_agent import render_outreach_message
        from app.models.generation import OutreachMessage
        om = OutreachMessage(intro="Hi, I am CG.", highlight="Built RAG.", call_to_action="Connect.", full_message="Hi, I am CG. Built RAG. Connect.")
        result = render_outreach_message(om)
        assert "Hi, I am CG" in result and "Connect" in result

    def test_render_outreach_returns_string(self):
        from app.agents.generator_agent import render_outreach_message
        from app.models.generation import OutreachMessage
        om = OutreachMessage(intro="Intro", highlight="Highlight", call_to_action="UniqueCallToAction999", full_message="Intro Highlight UniqueCallToAction999")
        assert isinstance(render_outreach_message(om), str)


class TestGeneratorPromptBuilders:
    def _jd_dict(self, jd):
        return jd.model_dump()

    def _resume_dict(self, resume):
        d = resume.model_dump()
        return {k: d[k] for k in ["personal_info", "resume_sections"]}

    def test_cover_letter_prompt_returns_two_strings(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate_cover_letter_prompt
        system, user = generate_cover_letter_prompt(self._jd_dict(sample_jd_fixture), self._resume_dict(sample_resume))
        assert isinstance(system, str) and isinstance(user, str)

    def test_cover_letter_prompt_mentions_company(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate_cover_letter_prompt
        system, user = generate_cover_letter_prompt(self._jd_dict(sample_jd_fixture), self._resume_dict(sample_resume))
        assert "DeepMind" in system or "DeepMind" in user

    def test_email_prompt_returns_two_strings(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate_email_prompt
        system, user = generate_email_prompt(self._jd_dict(sample_jd_fixture), self._resume_dict(sample_resume))
        assert isinstance(system, str) and isinstance(user, str)

    def test_outreach_prompt_returns_two_strings(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate_outreach_message_prompt
        system, user = generate_outreach_message_prompt(self._jd_dict(sample_jd_fixture), self._resume_dict(sample_resume), no_jd=False)
        assert isinstance(system, str) and isinstance(user, str)

    def test_outreach_no_jd_mode(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate_outreach_message_prompt
        system, user = generate_outreach_message_prompt(self._jd_dict(sample_jd_fixture), self._resume_dict(sample_resume), no_jd=True)
        assert isinstance(system, str) and isinstance(user, str)


class TestGeneratorGenerateFunction:
    def _email_parsed(self):
        from app.models.generation import Email
        return Email(subject="App", greeting="Hi,", body=["Applying."], closing="Best,", signature="CG")

    def _cl_parsed(self):
        from app.models.generation import CoverLetter
        return CoverLetter(greeting="Hi,", opening_paragraph="App.", body_paragraphs=["BG."], closing_paragraph="TY.", closing="Best,", signature="CG")

    def _outreach_parsed(self):
        from app.models.generation import OutreachMessage
        return OutreachMessage(intro="Hi.", highlight="ML.", call_to_action="Connect.", full_message="Hi. ML. Connect.")

    def test_generate_email_returns_string(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task", return_value=MagicMock(schema_matched=True, parsed=self._email_parsed())), \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = None
            mc.make_key.return_value = "k"
            assert isinstance(generate(sample_resume, sample_jd_fixture, type="email"), str)

    def test_generate_coverletter_returns_string(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task", return_value=MagicMock(schema_matched=True, parsed=self._cl_parsed())), \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = None
            mc.make_key.return_value = "k"
            assert isinstance(generate(sample_resume, sample_jd_fixture, type="coverletter"), str)

    def test_generate_outreach_returns_string(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task", return_value=MagicMock(schema_matched=True, parsed=self._outreach_parsed())), \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = None
            mc.make_key.return_value = "k"
            assert isinstance(generate(sample_resume, sample_jd_fixture, type="outreachmessage"), str)

    def test_cache_hit_skips_llm(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task") as mock_llm, \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = "cached email text"
            mc.make_key.return_value = "hit"
            result = generate(sample_resume, sample_jd_fixture, type="email")
        mock_llm.assert_not_called()
        assert result == "cached email text"

    def test_cache_miss_calls_llm(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task", return_value=MagicMock(schema_matched=True, parsed=self._email_parsed())) as mock_llm, \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = None
            mc.make_key.return_value = "miss"
            generate(sample_resume, sample_jd_fixture, type="email")
        mock_llm.assert_called_once()

    def test_schema_mismatch_raises(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task", return_value=MagicMock(schema_matched=False, parsed=None)), \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = None
            mc.make_key.return_value = "bad"
            with pytest.raises(Exception):
                generate(sample_resume, sample_jd_fixture, type="email")

    def test_unknown_type_raises(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with pytest.raises(Exception):
            generate(sample_resume, sample_jd_fixture, type="presentation")

    def test_result_stored_in_cache(self, sample_resume, sample_jd_fixture):
        from app.agents.generator_agent import generate
        with patch("app.agents.generator_agent.generate_for_task", return_value=MagicMock(schema_matched=True, parsed=self._email_parsed())), \
             patch("app.core.gen_cache.generation_cache") as mc:
            mc.get.return_value = None
            mc.make_key.return_value = "store"
            generate(sample_resume, sample_jd_fixture, type="email")
        mc.set.assert_called_once()


class TestFormatScoreFeedback:
    def _make_score(self, overall=72.0, ats=70.0, kw=75.0, quality=71.0, missing=None):
        from app.models.scoring import ResumeScore, DimensionScore, ScoreDimension
        ds = DimensionScore(dimension=ScoreDimension.ATS_FRIENDLINESS, score=int(ats), reasoning="ok", suggestions=["Add keywords"])
        return ResumeScore(
            app_id=0, resume_id=1, overall_score=overall,
            ats_friendliness_score=ats, keyword_match_score=kw, resume_quality_score=quality,
            dimension_scores=[ds],
            missing_keywords=missing or ["Docker", "Kubernetes"],
            overall_feedback="Good resume.",
            weights={"ats_friendliness": 0.3, "keyword_match": 0.4, "resume_quality": 0.3},
        )

    def test_returns_string(self):
        import app.core.pipeline as pl
        assert isinstance(pl.format_score_feedback(self._make_score()), str)

    def test_contains_overall_score(self):
        import app.core.pipeline as pl
        assert "72" in pl.format_score_feedback(self._make_score(overall=72.5))

    def test_contains_missing_keywords(self):
        import app.core.pipeline as pl
        result = pl.format_score_feedback(self._make_score(missing=["Kubernetes", "Terraform"]))
        assert "Kubernetes" in result or "Terraform" in result

    def test_non_empty(self):
        import app.core.pipeline as pl
        assert len(pl.format_score_feedback(self._make_score()).strip()) > 0

    def test_no_keywords_still_returns_string(self):
        import app.core.pipeline as pl
        assert isinstance(pl.format_score_feedback(self._make_score(missing=[])), str)


class TestGetStateEndpoint:
    def test_returns_200(self, edit_client):
        client, *_ = edit_client
        assert client.get("/edit/abc123/state").status_code == 200

    def test_returns_stage_field(self, edit_client):
        client, *_ = edit_client
        assert "stage" in client.get("/edit/abc123/state").json()

    def test_returns_error_field(self, edit_client):
        client, *_ = edit_client
        assert "error" in client.get("/edit/abc123/state").json()

    def test_error_is_null_when_no_error(self, edit_client):
        client, *_ = edit_client
        assert client.get("/edit/abc123/state").json()["error"] is None


class TestSessionListEndpoint:
    def _rows(self):
        return [{"id": 1, "user_id": 1, "thread_id": "tid-1", "status": "completed",
                 "created_at": "2026-06-01T10:00:00Z", "finished_at": "2026-06-01T11:00:00Z",
                 "role": "ML Engineer", "company": "DeepMind"}]

    def test_returns_200(self, edit_client):
        client, _, _, _, mock_sql = edit_client
        mock_sql.return_value.execute_raw.return_value = self._rows()
        assert client.get("/edit/sessions/1").status_code == 200

    def test_returns_list(self, edit_client):
        client, _, _, _, mock_sql = edit_client
        mock_sql.return_value.execute_raw.return_value = self._rows()
        assert isinstance(client.get("/edit/sessions/1").json(), list)

    def test_empty_returns_empty_list(self, edit_client):
        client, _, _, _, mock_sql = edit_client
        mock_sql.return_value.execute_raw.return_value = []
        resp = client.get("/edit/sessions/99")
        assert resp.status_code == 200 and resp.json() == []


class TestDiffEndpoint:
    def _diff_body(self):
        return {"has_changes": True, "flat_sections": {"skills": {"original": "old", "final": "new", "changed": True}}, "atomic_sections": {}}

    def test_returns_200(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_diff.return_value = self._diff_body()
        assert client.get("/edit/abc123/diff").status_code == 200

    def test_has_has_changes(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_diff.return_value = self._diff_body()
        assert "has_changes" in client.get("/edit/abc123/diff").json()

    def test_has_flat_sections(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_diff.return_value = self._diff_body()
        assert "flat_sections" in client.get("/edit/abc123/diff").json()

    def test_has_atomic_sections(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_diff.return_value = self._diff_body()
        assert "atomic_sections" in client.get("/edit/abc123/diff").json()


class TestPreviewEndpoint:
    def test_returns_200(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_preview.return_value = r"\documentclass{article}"
        assert client.get("/edit/abc123/preview").status_code == 200

    def test_returns_latex_field(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_preview.return_value = r"\documentclass{article}"
        assert "latex" in client.get("/edit/abc123/preview").json()

    def test_latex_is_string(self, edit_client):
        client, _, _, mock_pl, _ = edit_client
        mock_pl.build_preview.return_value = r"\documentclass[11pt]{article}"
        assert isinstance(client.get("/edit/abc123/preview").json()["latex"], str)


class TestScoreAgentCaching:
    def _llm_out(self):
        from app.models.scoring import DimensionScore, LLMScoreOutput, ScoreDimension
        ats = DimensionScore(dimension=ScoreDimension.ATS_FRIENDLINESS, score=75, reasoning="ok", suggestions=[])
        quality = DimensionScore(dimension=ScoreDimension.RESUME_QUALITY, score=70, reasoning="ok", suggestions=[])
        return LLMScoreOutput(ats_friendliness=ats, resume_quality=quality, overall_feedback="Good.")

    def test_identical_inputs_hit_cache(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        llm_out = self._llm_out()
        call_count = {"n": 0}

        def _counting(*a, **kw):
            call_count["n"] += 1
            return llm_out

        latex = r"\item Python Docker"
        with patch("app.agents.score_agent._score_llm_dimensions", side_effect=_counting), \
             patch("app.agents.score_agent._score_keyword_match", return_value=(0.75, [])), \
             patch.dict("app.agents.score_agent._score_cache", {}, clear=True):
            sa.score(app_id=1, resume_id=1, resume_latex=latex, jd=sample_jd_fixture)
            sa.score(app_id=1, resume_id=1, resume_latex=latex, jd=sample_jd_fixture)
        assert call_count["n"] == 1

    def test_different_resume_bypasses_cache(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        llm_out = self._llm_out()
        call_count = {"n": 0}

        def _counting(*a, **kw):
            call_count["n"] += 1
            return llm_out

        with patch("app.agents.score_agent._score_llm_dimensions", side_effect=_counting), \
             patch("app.agents.score_agent._score_keyword_match", return_value=(0.75, [])), \
             patch.dict("app.agents.score_agent._score_cache", {}, clear=True):
            sa.score(app_id=1, resume_id=1, resume_latex=r"\item Python", jd=sample_jd_fixture)
            sa.score(app_id=1, resume_id=1, resume_latex=r"\item Go Rust", jd=sample_jd_fixture)
        assert call_count["n"] == 2

    def test_score_returns_resume_score(self, sample_jd_fixture):
        from app.models.scoring import ResumeScore
        import app.agents.score_agent as sa
        with patch("app.agents.score_agent._score_llm_dimensions", return_value=self._llm_out()), \
             patch("app.agents.score_agent._score_keyword_match", return_value=(0.75, ["Docker"])), \
             patch.dict("app.agents.score_agent._score_cache", {}, clear=True):
            result = sa.score(app_id=1, resume_id=1, resume_latex=r"\item Python", jd=sample_jd_fixture)
        assert isinstance(result, ResumeScore)

    def test_score_has_overall_score_in_range(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        with patch("app.agents.score_agent._score_llm_dimensions", return_value=self._llm_out()), \
             patch("app.agents.score_agent._score_keyword_match", return_value=(0.80, [])), \
             patch.dict("app.agents.score_agent._score_cache", {}, clear=True):
            result = sa.score(app_id=1, resume_id=1, resume_latex=r"\item Python", jd=sample_jd_fixture)
        assert 0.0 <= result.overall_score <= 100.0

    def test_none_llm_output_raises(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        with patch("app.agents.score_agent._score_llm_dimensions", return_value=None), \
             patch("app.agents.score_agent._score_keyword_match", return_value=(0.5, [])), \
             patch.dict("app.agents.score_agent._score_cache", {}, clear=True):
            with pytest.raises(ValueError):
                sa.score(app_id=1, resume_id=1, resume_latex=r"\item Python", jd=sample_jd_fixture)


class TestScoreKeywordMatch:
    def test_returns_tuple(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        import numpy as np
        emb = np.array([1.0, 0.0, 0.0])
        with patch("app.agents.score_agent._get_embedding", return_value=emb):
            score, missing = sa._score_keyword_match("Python FastAPI", sample_jd_fixture)
        assert isinstance(score, float) and isinstance(missing, list)

    def test_score_in_valid_range(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        import numpy as np
        emb = np.array([0.7, 0.3, 0.5])
        with patch("app.agents.score_agent._get_embedding", return_value=emb):
            score, _ = sa._score_keyword_match("Python FastAPI", sample_jd_fixture)
        assert 0.0 <= score <= 100.0

    def test_missing_keywords_are_strings(self, sample_jd_fixture):
        import app.agents.score_agent as sa
        import numpy as np
        emb = np.array([1.0, 0.0])
        with patch("app.agents.score_agent._get_embedding", return_value=emb):
            _, missing = sa._score_keyword_match("JavaScript", sample_jd_fixture)
        assert all(isinstance(k, str) for k in missing)
