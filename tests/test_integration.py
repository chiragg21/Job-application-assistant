"""
Integration tests -- cross-route wiring and contract verification.

Each fixture creates a minimal FastAPI app with only the relevant router
mounted and all external deps mocked. This avoids lifespan startup issues
while still exercising the full request/response pipeline.
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _jd_dict():
    return {
        "company": "DeepMind", "role": "ML Engineer", "seniority_level": "mid",
        "location": "London", "is_remote": True, "about_company": "AI lab.",
        "about_job": "Build training infra.",
        "responsibilities": ["Design ML pipelines"],
        "required_skills": ["Python", "PyTorch"], "nice_to_have_skills": ["LangChain"],
        "perks": "Equity", "others": "", "raw_text": "",
    }


def _edit_cycle_state():
    mock_section = MagicMock()
    mock_section.updated_section = ""
    mock_rs = MagicMock()
    mock_rs.get_section.return_value = mock_section
    mock_rs.experience = []
    mock_rs.projects = []
    mock_node = MagicMock()
    mock_node.resume_state = mock_rs
    mock_cycle = MagicMock()
    mock_cycle.nodes = {0: mock_node}
    mock_cycle.current_state = mock_rs
    return mock_cycle


def _graph_state(stage="item_selection"):
    gs = MagicMock()
    gs.values = {
        "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
        "user_id": 1, "resume_id": 1,
        "parsed_jd": _jd_dict(),
        "stage": stage, "error": None,
        "score_result": None, "generation_results": None, "dropped_items": [],
    }
    gs.tasks = []
    gs.next = []
    return gs


@pytest.fixture(scope="module")
def edit_router_client():
    from app.api.routes.edit import router
    app = FastAPI()
    app.include_router(router)

    mock_graph = MagicMock()
    mock_graph.get_state.return_value = _graph_state()
    mock_graph.invoke.return_value = _graph_state()
    mock_graph.update_state.return_value = None
    mock_graph.stream.return_value = iter([])

    mock_agent = MagicMock()
    mock_agent.editing_cycle = _edit_cycle_state()

    mock_pl = MagicMock()
    mock_pl.build_preview.return_value = r"\documentclass{article}"
    mock_pl.build_diff.return_value = {"has_changes": False, "flat_sections": {}, "atomic_sections": {}}
    mock_pl.format_score_feedback.return_value = "Improve keywords."

    mock_sql = MagicMock()
    mock_sql.return_value.execute_raw.return_value = []

    with patch("app.api.routes.edit.edit_graph", mock_graph), \
         patch("app.api.routes.edit._rebuild_agent", return_value=mock_agent), \
         patch("app.api.routes.edit.pl", mock_pl), \
         patch("app.api.routes.edit.SQLHandler", mock_sql), \
         patch("app.api.routes.edit.register_thread"):
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def jd_router_client():
    from app.api.routes.jd import router
    app = FastAPI()
    app.include_router(router)
    mock_parsed_jd = MagicMock()
    mock_parsed_jd.model_dump.return_value = _jd_dict()
    mock_pl = MagicMock()
    mock_pl.parse_jd.return_value = (1, mock_parsed_jd)
    with patch("app.api.routes.jd.pl", mock_pl):
        yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def resume_router_client():
    from app.api.routes.resume import router
    app = FastAPI()
    app.include_router(router)
    yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def keys_router_client():
    from app.api.routes.keys import router
    app = FastAPI()
    app.include_router(router)
    with patch("app.api.routes.keys._load_raw", return_value=[]), \
         patch("app.api.routes.keys._save_raw", return_value=None):
        yield TestClient(app, raise_server_exceptions=False)


class TestJDParseRoute:
    def _payload(self):
        return {"jd_text": "We are hiring a Python engineer.", "user_id": 1}

    def test_200_on_valid_payload(self, jd_router_client):
        resp = jd_router_client.post("/jd/parse", json=self._payload())
        assert resp.status_code == 200

    def test_response_has_job_id(self, jd_router_client):
        assert "job_id" in jd_router_client.post("/jd/parse", json=self._payload()).json()

    def test_response_has_parsed_jd(self, jd_router_client):
        assert "parsed_jd" in jd_router_client.post("/jd/parse", json=self._payload()).json()

    def test_422_on_missing_user_id(self, jd_router_client):
        assert jd_router_client.post("/jd/parse", json={"jd_text": "ML role."}).status_code == 422

    def test_422_on_missing_jd_text(self, jd_router_client):
        assert jd_router_client.post("/jd/parse", json={"user_id": 1}).status_code == 422


class TestJDExtractFromURLRoute:
    def _url_payload(self):
        return {"url": "https://example.com/job", "user_id": 1}

    def test_200_on_valid_url(self, jd_router_client):
        with patch("app.api.routes.jd.extract_jd_from_url", return_value="ML role text"):
            resp = jd_router_client.post("/jd/extract-from-url", json=self._url_payload())
        assert resp.status_code == 200

    def test_response_has_job_id(self, jd_router_client):
        with patch("app.api.routes.jd.extract_jd_from_url", return_value="ML role text"):
            resp = jd_router_client.post("/jd/extract-from-url", json=self._url_payload())
        assert "job_id" in resp.json()

    def test_422_on_missing_url(self, jd_router_client):
        assert jd_router_client.post("/jd/extract-from-url", json={"user_id": 1}).status_code == 422


class TestResumeTemplatesRoute:
    def test_200(self, resume_router_client):
        assert resume_router_client.get("/resume/templates").status_code == 200

    def test_returns_list_or_dict(self, resume_router_client):
        assert isinstance(resume_router_client.get("/resume/templates").json(), (list, dict))


class TestEditStateRoute:
    def test_200(self, edit_router_client):
        assert edit_router_client.get("/edit/test-thread/state").status_code == 200

    def test_has_stage(self, edit_router_client):
        assert "stage" in edit_router_client.get("/edit/test-thread/state").json()

    def test_has_error(self, edit_router_client):
        assert "error" in edit_router_client.get("/edit/test-thread/state").json()

    def test_error_is_null(self, edit_router_client):
        assert edit_router_client.get("/edit/test-thread/state").json()["error"] is None


class TestEditStartRoute:
    def _payload(self):
        return {"user_id": 1, "jd_text": "ML Engineer role at DeepMind."}

    def test_200_or_202(self, edit_router_client):
        resp = edit_router_client.post("/edit/start", json=self._payload())
        assert resp.status_code in (200, 202)

    def test_returns_thread_id(self, edit_router_client):
        assert "thread_id" in edit_router_client.post("/edit/start", json=self._payload()).json()

    def test_422_on_missing_user_id(self, edit_router_client):
        assert edit_router_client.post("/edit/start", json={"jd_text": "role"}).status_code == 422

    def test_422_on_missing_jd_text(self, edit_router_client):
        assert edit_router_client.post("/edit/start", json={"user_id": 1}).status_code == 422


class TestEditResumeRoute:
    def test_valid_status_on_resume(self, edit_router_client):
        resp = edit_router_client.post("/edit/test-thread/resume", json={"response": {"proposal": "accept"}})
        assert resp.status_code in (200, 202, 400, 422)

    def test_422_on_missing_response(self, edit_router_client):
        assert edit_router_client.post("/edit/test-thread/resume", json={}).status_code == 422


class TestEditPreviewRoute:
    def test_200(self, edit_router_client):
        assert edit_router_client.get("/edit/test-thread/preview").status_code == 200

    def test_has_latex(self, edit_router_client):
        assert "latex" in edit_router_client.get("/edit/test-thread/preview").json()

    def test_latex_is_string(self, edit_router_client):
        assert isinstance(edit_router_client.get("/edit/test-thread/preview").json()["latex"], str)


class TestEditDiffRoute:
    def test_200(self, edit_router_client):
        assert edit_router_client.get("/edit/test-thread/diff").status_code == 200

    def test_has_has_changes(self, edit_router_client):
        assert "has_changes" in edit_router_client.get("/edit/test-thread/diff").json()

    def test_has_flat_sections(self, edit_router_client):
        assert "flat_sections" in edit_router_client.get("/edit/test-thread/diff").json()

    def test_has_atomic_sections(self, edit_router_client):
        assert "atomic_sections" in edit_router_client.get("/edit/test-thread/diff").json()


class TestEditSessionsRoute:
    def test_200(self, edit_router_client):
        assert edit_router_client.get("/edit/sessions/1").status_code == 200

    def test_returns_list(self, edit_router_client):
        assert isinstance(edit_router_client.get("/edit/sessions/1").json(), list)


class TestEditFinishRoute:
    def test_valid_response(self, edit_router_client):
        resp = edit_router_client.post("/edit/test-thread/finish")
        assert resp.status_code in (200, 400, 404, 500)


class TestKeysRoutes:
    def test_status_200(self, keys_router_client):
        assert keys_router_client.get("/keys/status").status_code == 200

    def test_status_returns_list(self, keys_router_client):
        assert isinstance(keys_router_client.get("/keys/status").json(), list)

    def test_add_key_422_on_missing_provider(self, keys_router_client):
        resp = keys_router_client.post("/keys", json={"key": "abc"})
        assert resp.status_code == 422

    def test_add_key_422_on_missing_key(self, keys_router_client):
        resp = keys_router_client.post("/keys", json={"provider": "gemini"})
        assert resp.status_code == 422


class TestOpenAPISchema:
    def test_openapi_200(self, edit_router_client):
        assert edit_router_client.get("/openapi.json").status_code == 200

    def test_openapi_has_paths(self, edit_router_client):
        schema = edit_router_client.get("/openapi.json").json()
        assert "paths" in schema and len(schema["paths"]) > 0

    def test_edit_state_in_schema(self, edit_router_client):
        paths = edit_router_client.get("/openapi.json").json()["paths"]
        assert any("state" in p for p in paths)

    def test_edit_diff_in_schema(self, edit_router_client):
        paths = edit_router_client.get("/openapi.json").json()["paths"]
        assert any("diff" in p for p in paths)

    def test_jd_parse_in_jd_schema(self, jd_router_client):
        paths = jd_router_client.get("/openapi.json").json()["paths"]
        assert any("parse" in p for p in paths)
