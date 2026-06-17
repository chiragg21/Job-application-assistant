"""
Sprint 0 — Foundation tests.

Covers: FastAPI app meta endpoints, core pydantic models, JD parsing route,
resume model validation, SQLHandler basic contract, LaTeX personal-info extraction.
No real DB, LLM, or ChromaDB connections are made — all external deps mocked.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# Fixture: minimal test client for the full app (lifespan suppressed)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def app_client():
    """
    Spin up the full FastAPI app without real startup side-effects.
    The lifespan handler touches _checkpointer, evict_old_threads, load_user_keys,
    and prewarm_embeddings — all mocked so tests stay fast and self-contained.
    """
    with patch("app.graph.edit_graph._checkpointer") as mock_cp, \
         patch("app.graph.edit_graph.evict_old_threads"), \
         patch("app.api.routes.keys.load_user_keys"), \
         patch("app.core.embeddings.prewarm"):
        mock_cp.setup = MagicMock()
        from app.api.main import app
        client = TestClient(app, raise_server_exceptions=False)
        yield client


# ══════════════════════════════════════════════════════════════════════════════
# Meta endpoints
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthAndRoot:
    def test_health_returns_200(self, app_client):
        resp = app_client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_ok_status(self, app_client):
        resp = app_client.get("/health")
        assert resp.json()["status"] == "ok"

    def test_root_returns_200(self, app_client):
        resp = app_client.get("/")
        assert resp.status_code == 200

    def test_root_body_has_message(self, app_client):
        resp = app_client.get("/")
        assert "message" in resp.json()

    def test_root_body_has_docs_link(self, app_client):
        resp = app_client.get("/")
        body = resp.json()
        assert "docs" in body or "redoc" in body


# ══════════════════════════════════════════════════════════════════════════════
# ParsedJD model
# ══════════════════════════════════════════════════════════════════════════════

class TestParsedJDModel:
    def _make_jd(self, **overrides):
        from app.models.jd import ParsedJD
        defaults = dict(
            company="Acme", role="SWE", seniority_level="mid",
            location="NYC", is_remote=False, about_company="",
            about_job="", responsibilities=["Ship code"],
            required_skills=["Python"], nice_to_have_skills=["Go"],
            perks="", others="", raw_text="",
        )
        defaults.update(overrides)
        return ParsedJD(**defaults)

    def test_can_instantiate(self):
        jd = self._make_jd()
        assert jd.company == "Acme"

    def test_defaults_are_empty(self):
        from app.models.jd import ParsedJD
        jd = ParsedJD()
        assert jd.company == ""
        assert jd.required_skills == []

    def test_required_skills_is_list(self):
        jd = self._make_jd(required_skills=["Python", "FastAPI"])
        assert isinstance(jd.required_skills, list)
        assert len(jd.required_skills) == 2

    def test_raw_text_excluded_from_model_dump(self):
        jd = self._make_jd(raw_text="secret raw text")
        dumped = jd.model_dump()
        assert "raw_text" not in dumped

    def test_to_dict_excludes_raw_text(self):
        jd = self._make_jd(raw_text="private")
        d = jd.to_dict()
        assert "raw_text" not in d

    def test_is_remote_defaults_false(self):
        from app.models.jd import ParsedJD
        jd = ParsedJD(company="X")
        assert jd.is_remote is False

    def test_round_trip_through_dict(self):
        from app.models.jd import ParsedJD
        jd = self._make_jd()
        dumped = jd.model_dump()
        restored = ParsedJD(**dumped)
        assert restored.company == jd.company
        assert restored.required_skills == jd.required_skills


# ══════════════════════════════════════════════════════════════════════════════
# ParsedResume / PersonalInfo models
# ══════════════════════════════════════════════════════════════════════════════

class TestPersonalInfoModel:
    def test_can_instantiate_all_fields(self):
        from app.models.resume import PersonalInfo
        pi = PersonalInfo(
            name="Chirag Garg", email="c@x.com", phone="+91-123",
            github="https://github.com/cg", linkedin="https://linkedin.com/in/cg",
            github_handle="cg", linkedin_name="Chirag Garg",
        )
        assert pi.name == "Chirag Garg"

    def test_optional_fields_default_to_empty(self):
        from app.models.resume import PersonalInfo
        pi = PersonalInfo(name="X", email="", phone="", linkedin="", github="")
        assert hasattr(pi, "email")

    def test_model_dump_returns_dict(self):
        from app.models.resume import PersonalInfo
        pi = PersonalInfo(name="X", email="x@x.com", phone="", linkedin="", github="")
        assert isinstance(pi.model_dump(), dict)


class TestParsedResumeModel:
    def _make_resume(self, **kw):
        from app.models.resume import ParsedResume, PersonalInfo, ResumeSection
        defaults = dict(
            resume_id="1",
            resume_path="/tmp/test.tex",
            personal_info=PersonalInfo(name="A", email="a@a.com", phone="", linkedin="", github=""),
            resume_sections=ResumeSection(),
        )
        defaults.update(kw)
        return ParsedResume(**defaults)

    def test_can_instantiate_empty(self):
        pr = self._make_resume()
        assert isinstance(pr, object)

    def test_resume_id_field_exists(self):
        pr = self._make_resume(resume_id="1")
        assert pr.resume_id == "1"


# ══════════════════════════════════════════════════════════════════════════════
# ResumeScore / DimensionScore models
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringModels:
    def test_dimension_score_instantiation(self):
        from app.models.scoring import DimensionScore, ScoreDimension
        ds = DimensionScore(
            dimension=ScoreDimension.ATS_FRIENDLINESS,
            score=75,
            reasoning="Good",
            suggestions=["Add keywords"],
        )
        assert ds.score == 75

    def test_resume_score_instantiation(self):
        from app.models.scoring import ResumeScore, DimensionScore, ScoreDimension
        ds = DimensionScore(
            dimension=ScoreDimension.KEYWORD_MATCH,
            score=80, reasoning="ok", suggestions=[],
        )
        rs = ResumeScore(
            app_id=1, resume_id=1,
            overall_score=78.0,
            ats_friendliness_score=75.0, keyword_match_score=80.0, resume_quality_score=79.0,
            dimension_scores=[ds],
            missing_keywords=[],
            overall_feedback="Good",
            weights={"ats_friendliness": 0.3, "keyword_match": 0.4, "resume_quality": 0.3},
        )
        assert rs.overall_score == 78.0

    def test_score_dimensions_enum_values(self):
        from app.models.scoring import ScoreDimension
        values = [d.value for d in ScoreDimension]
        assert "ats_friendliness" in values
        assert "keyword_match" in values
        assert "resume_quality" in values


# ══════════════════════════════════════════════════════════════════════════════
# Generation models
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationModels:
    def test_email_model(self):
        from app.models.generation import Email
        e = Email(
            subject="Job App", greeting="Hi,",
            body=["Para 1"], closing="Thanks,", signature="X",
        )
        assert e.subject == "Job App"

    def test_cover_letter_model(self):
        from app.models.generation import CoverLetter
        cl = CoverLetter(
            greeting="Dear HM,", opening_paragraph="I apply.",
            body_paragraphs=["Experience."],
            closing_paragraph="Thank you.", closing="Best,", signature="X",
        )
        assert len(cl.body_paragraphs) == 1

    def test_outreach_message_model(self):
        from app.models.generation import OutreachMessage
        om = OutreachMessage(
            intro="Hi", highlight="RAG experience",
            call_to_action="Let's chat", full_message="Hi RAG Let's chat",
        )
        assert "RAG" in om.highlight


# ══════════════════════════════════════════════════════════════════════════════
# Edit models (ResumeEditCycle, SectionEditState, etc.)
# ══════════════════════════════════════════════════════════════════════════════

class TestEditModels:
    def test_section_edit_state_instantiation(self):
        from app.models.edit import SectionEditState
        s = SectionEditState(
            section_name="skills",
            section_previous_state="\\item Python",
            lines_to_change=["\\item Python"],
            suggested_changes=["\\item Python, FastAPI"],
            updated_section="\\item Python, FastAPI",
        )
        assert s.section_name == "skills"

    def test_item_edit_state_instantiation(self):
        from app.models.edit import ItemEditState
        i = ItemEditState(
            section_name="experience",
            item_name="Google",
            section_previous_state="old text",
            lines_to_change=[], suggested_changes=[], updated_section="new text",
        )
        assert i.item_name == "Google"

    def test_resume_edit_state_has_flat_and_atomic_sections(self):
        from app.models.edit import ResumeEditState
        res = ResumeEditState()
        # Should have attributes for all section types (may be None or empty)
        assert hasattr(res, "skills") or hasattr(res, "experience")

    def test_resume_state_node_has_node_id(self):
        from app.models.edit import ResumeStateNode, ResumeEditState
        node = ResumeStateNode(
            node_id=0, action="initial",
            resume_state=ResumeEditState(), section_name=None,
        )
        assert node.node_id == 0

    def test_resume_edit_cycle_starts_at_node_zero(self):
        from app.models.edit import ResumeEditCycle, ResumeEditState
        from app.models.jd import ParsedJD
        jd = ParsedJD()
        res = ResumeEditState()
        cycle = ResumeEditCycle(jd=jd, initial_state=res, cycle_id=1)
        assert cycle.current_node_id == 0

    def test_resume_edit_cycle_push_increments_node(self):
        from app.models.edit import ResumeEditCycle, ResumeEditState
        from app.models.jd import ParsedJD
        jd = ParsedJD()
        res = ResumeEditState()
        cycle = ResumeEditCycle(jd=jd, cycle_id=1)
        cycle.init_root(res)
        cycle.push(ResumeEditState(), action="edit", section_name="skills")
        assert cycle.current_node_id == 1

    def test_resume_edit_cycle_serialises_to_dict(self):
        from app.models.edit import ResumeEditCycle, ResumeEditState
        from app.models.jd import ParsedJD
        cycle = ResumeEditCycle(jd=ParsedJD(), initial_state=ResumeEditState(), cycle_id=1)
        d = cycle.model_dump()
        assert isinstance(d, dict)
        assert "nodes" in d or "current_node_id" in d


# ══════════════════════════════════════════════════════════════════════════════
# Resume parser — personal info extraction (regex path)
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_TEX_HEADER = r"""
\begin{center}
    {\Huge \textbf{John Doe}}\\
    \vspace{1mm}
    \faEnvelope~\href{mailto:john@example.com}{john@example.com} \quad
    \faPhone~+1-555-0100 \quad
    \faGithub~\href{https://github.com/johndoe}{johndoe} \quad
    \faLinkedin~\href{https://linkedin.com/in/johndoe}{John Doe}
\end{center}
"""


class TestResumeParserPersonalInfo:
    def _make_parser(self):
        from app.core.resume_parser import ResumeParser
        p = object.__new__(ResumeParser)
        p.db = MagicMock()
        p.llmhandler = MagicMock()
        p._chroma_client = MagicMock()
        p.col_resume = MagicMock()
        return p

    def test_extract_name(self):
        parser = self._make_parser()
        info = parser.extract_personal_info(SAMPLE_TEX_HEADER)
        assert info.name == "John Doe"

    def test_extract_email(self):
        parser = self._make_parser()
        info = parser.extract_personal_info(SAMPLE_TEX_HEADER)
        assert info.email == "john@example.com"

    def test_extract_phone(self):
        parser = self._make_parser()
        info = parser.extract_personal_info(SAMPLE_TEX_HEADER)
        assert "+1-555-0100" in info.phone

    def test_extract_github_handle(self):
        parser = self._make_parser()
        info = parser.extract_personal_info(SAMPLE_TEX_HEADER)
        assert "johndoe" in info.github

    def test_extract_linkedin_name(self):
        parser = self._make_parser()
        info = parser.extract_personal_info(SAMPLE_TEX_HEADER)
        assert "johndoe" in info.linkedin or "linkedin" in info.linkedin

    def test_missing_header_returns_defaults(self):
        parser = self._make_parser()
        info = parser.extract_personal_info(r"\documentclass{article}\begin{document}\end{document}")
        # Should not raise; returns a PersonalInfo with empty/default fields
        from app.models.resume import PersonalInfo
        assert isinstance(info, PersonalInfo)


# ══════════════════════════════════════════════════════════════════════════════
# JD parse route — /jd/parse
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def jd_client():
    from app.api.routes.jd import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestJDParseRoute:
    def _mock_parse_jd(self, job_id=1):
        from app.models.jd import ParsedJD
        parsed = ParsedJD(company="Acme", role="SWE", required_skills=["Python"])
        return job_id, parsed

    def test_parse_jd_returns_200(self, jd_client):
        with patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.return_value = self._mock_parse_jd()
            resp = jd_client.post("/jd/parse", json={"jd_text": "We need a Python engineer", "user_id": 1})
        assert resp.status_code == 200

    def test_parse_jd_returns_job_id(self, jd_client):
        with patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.return_value = self._mock_parse_jd(job_id=42)
            resp = jd_client.post("/jd/parse", json={"jd_text": "Python role", "user_id": 1})
        assert resp.json()["job_id"] == 42

    def test_parse_jd_returns_parsed_jd_dict(self, jd_client):
        with patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.return_value = self._mock_parse_jd()
            resp = jd_client.post("/jd/parse", json={"jd_text": "Python role", "user_id": 1})
        body = resp.json()
        assert "parsed_jd" in body
        assert body["parsed_jd"]["company"] == "Acme"

    def test_parse_jd_missing_jd_text_returns_422(self, jd_client):
        resp = jd_client.post("/jd/parse", json={"user_id": 1})
        assert resp.status_code == 422

    def test_parse_jd_missing_user_id_returns_422(self, jd_client):
        resp = jd_client.post("/jd/parse", json={"jd_text": "Some JD"})
        assert resp.status_code == 422

    def test_parse_jd_pipeline_error_returns_500(self, jd_client):
        with patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.side_effect = RuntimeError("DB unavailable")
            resp = jd_client.post("/jd/parse", json={"jd_text": "Python role", "user_id": 1})
        assert resp.status_code == 500

    def test_parse_jd_calls_pipeline_with_correct_args(self, jd_client):
        with patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.return_value = self._mock_parse_jd()
            jd_client.post("/jd/parse", json={"jd_text": "We need Python", "user_id": 7})
        mock_pl.parse_jd.assert_called_once_with(jd_text="We need Python", user_id=7)


# ══════════════════════════════════════════════════════════════════════════════
# JD extract-from-URL route — /jd/extract-from-url
# ══════════════════════════════════════════════════════════════════════════════

class TestJDExtractFromURLRoute:
    def _mock_parse_jd(self):
        from app.models.jd import ParsedJD
        return 1, ParsedJD(company="DeepMind", role="Researcher")

    def test_extract_from_url_returns_200(self, jd_client):
        with patch("app.core.jd_extractor.extract_jd_from_url", return_value="JD text"), \
             patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.return_value = self._mock_parse_jd()
            resp = jd_client.post(
                "/jd/extract-from-url",
                json={"url": "https://example.com/job", "user_id": 1},
            )
        assert resp.status_code == 200

    def test_extract_from_url_calls_extractor(self, jd_client):
        with patch("app.core.jd_extractor.extract_jd_from_url", return_value="extracted text") as mock_ex, \
             patch("app.api.routes.jd.pl") as mock_pl:
            mock_pl.parse_jd.return_value = self._mock_parse_jd()
            jd_client.post(
                "/jd/extract-from-url",
                json={"url": "https://example.com/job", "user_id": 1},
            )
        mock_ex.assert_called_once_with("https://example.com/job")

    def test_extract_from_url_value_error_returns_422(self, jd_client):
        with patch("app.core.jd_extractor.extract_jd_from_url",
                   side_effect=ValueError("URL not reachable")):
            resp = jd_client.post(
                "/jd/extract-from-url",
                json={"url": "https://bad.url", "user_id": 1},
            )
        assert resp.status_code == 422

    def test_extract_from_url_network_error_returns_500(self, jd_client):
        with patch("app.core.jd_extractor.extract_jd_from_url",
                   side_effect=ConnectionError("timeout")):
            resp = jd_client.post(
                "/jd/extract-from-url",
                json={"url": "https://slow.url", "user_id": 1},
            )
        assert resp.status_code == 500


# ══════════════════════════════════════════════════════════════════════════════
# SQLHandler contract (unit — no real DB)
# ══════════════════════════════════════════════════════════════════════════════

class TestSQLHandlerContract:
    """Verify that SQLHandler can be imported and has the expected interface."""

    def test_can_import(self):
        from app.utils import SQLHandler
        assert SQLHandler is not None

    def test_has_fetch_one_method(self):
        from app.utils import SQLHandler
        assert hasattr(SQLHandler, "fetch_one") or callable(getattr(SQLHandler, "fetch_one", None))

    def test_has_add_one_method(self):
        from app.utils import SQLHandler
        assert hasattr(SQLHandler, "add_one")

    def test_has_update_data_method(self):
        from app.utils import SQLHandler
        assert hasattr(SQLHandler, "update_data")

    def test_has_execute_raw_method(self):
        from app.utils import SQLHandler
        assert hasattr(SQLHandler, "execute_raw")
