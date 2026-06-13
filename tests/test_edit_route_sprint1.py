"""Tests for Sprint 1 edit route additions: custom_instruction and export endpoint."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_LATEX = r"\documentclass{article}\begin{document}Hello\end{document}"

DUMMY_STATE = {
    "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
    "user_id": 1,
    "parsed_jd": {"required_skills": [], "responsibilities": []},
}


@pytest.fixture
def export_client():
    """Minimal FastAPI app with only the edit router, all heavy deps mocked."""
    from app.api.routes.edit import router

    app = FastAPI()
    app.include_router(router)

    mock_graph_state = MagicMock()
    mock_graph_state.values = DUMMY_STATE.copy()

    mock_graph = MagicMock()
    mock_graph.get_state.return_value = mock_graph_state

    mock_agent = MagicMock()

    with patch("app.api.routes.edit.edit_graph", mock_graph), \
         patch("app.api.routes.edit._rebuild_agent", return_value=mock_agent), \
         patch("app.api.routes.edit.pl") as mock_pl, \
         patch("app.api.routes.edit.register_thread"):
        mock_pl.build_preview.return_value = SAMPLE_LATEX
        yield TestClient(app), mock_graph, mock_agent, mock_pl


# ── StartSessionRequest model validation ──────────────────────────────────────

class TestStartSessionRequestModel:

    def test_custom_instruction_defaults_to_none(self):
        from app.api.routes.edit import StartSessionRequest
        req = StartSessionRequest(user_id=1, jd_text="some JD")
        assert req.custom_instruction is None

    def test_custom_instruction_accepted_as_string(self):
        from app.api.routes.edit import StartSessionRequest
        req = StartSessionRequest(
            user_id=1,
            jd_text="some JD",
            custom_instruction="Use quantified metrics",
        )
        assert req.custom_instruction == "Use quantified metrics"

    def test_custom_instruction_none_explicit(self):
        from app.api.routes.edit import StartSessionRequest
        req = StartSessionRequest(user_id=1, jd_text="some JD", custom_instruction=None)
        assert req.custom_instruction is None

    def test_user_id_required(self):
        from app.api.routes.edit import StartSessionRequest
        with pytest.raises(ValidationError):
            StartSessionRequest(jd_text="some JD")

    def test_jd_text_required(self):
        from app.api.routes.edit import StartSessionRequest
        with pytest.raises(ValidationError):
            StartSessionRequest(user_id=1)


# ── Export endpoint — .tex format ─────────────────────────────────────────────

class TestExportTex:

    def test_returns_200(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abc123/export?format=tex")
        assert resp.status_code == 200

    def test_content_type_is_text_plain(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abc123/export?format=tex")
        assert "text/plain" in resp.headers["content-type"]

    def test_content_disposition_filename_tex(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abc123/export?format=tex")
        disposition = resp.headers.get("content-disposition", "")
        assert ".tex" in disposition

    def test_body_contains_latex(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abc123/export?format=tex")
        assert SAMPLE_LATEX.encode() in resp.content

    def test_filename_uses_thread_id_prefix(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abcdef12-xxxx/export?format=tex")
        disposition = resp.headers.get("content-disposition", "")
        assert "abcdef12" in disposition


# ── Export endpoint — .pdf format ─────────────────────────────────────────────

class TestExportPdf:

    def test_pdflatex_not_installed_returns_501(self, export_client):
        client, mock_graph, mock_agent, mock_pl = export_client
        with patch("app.api.routes.edit.subprocess.run", side_effect=FileNotFoundError):
            resp = client.get("/edit/abc123/export?format=pdf")
        assert resp.status_code == 501
        assert "pdflatex" in resp.json()["detail"].lower()

    def test_successful_pdf_returns_200(self, export_client, tmp_path):
        client, mock_graph, mock_agent, mock_pl = export_client
        fake_pdf = b"%PDF-1.4 fake content"

        def _fake_subprocess(cmd, **kwargs):
            # Write a fake PDF so the endpoint finds the file
            import shutil
            outdir = cmd[cmd.index("-output-directory") + 1]
            (type("p", (), {"__truediv__": lambda s, n: tmp_path / n})() / "resume.pdf")
            pdf_path = __import__("pathlib").Path(outdir) / "resume.pdf"
            pdf_path.write_bytes(fake_pdf)
            return MagicMock(returncode=0, stderr=b"")

        with patch("app.api.routes.edit.subprocess.run", side_effect=_fake_subprocess):
            resp = client.get("/edit/abc123/export?format=pdf")
        assert resp.status_code == 200

    def test_successful_pdf_content_type(self, export_client, tmp_path):
        client, mock_graph, mock_agent, mock_pl = export_client
        fake_pdf = b"%PDF-1.4 fake content"

        def _fake_subprocess(cmd, **kwargs):
            outdir = cmd[cmd.index("-output-directory") + 1]
            pdf_path = __import__("pathlib").Path(outdir) / "resume.pdf"
            pdf_path.write_bytes(fake_pdf)
            return MagicMock(returncode=0, stderr=b"")

        with patch("app.api.routes.edit.subprocess.run", side_effect=_fake_subprocess):
            resp = client.get("/edit/abc123/export?format=pdf")
        assert "application/pdf" in resp.headers["content-type"]

    def test_pdflatex_failure_returns_500(self, export_client, tmp_path):
        """pdflatex runs but produces no PDF → 500."""
        client, mock_graph, mock_agent, mock_pl = export_client

        def _fake_subprocess_no_output(cmd, **kwargs):
            # Do NOT write resume.pdf — simulate pdflatex failure
            return MagicMock(returncode=1, stderr=b"LaTeX error: missing package")

        with patch("app.api.routes.edit.subprocess.run", side_effect=_fake_subprocess_no_output):
            resp = client.get("/edit/abc123/export?format=pdf")
        assert resp.status_code == 500


# ── Export endpoint — unknown format ─────────────────────────────────────────

class TestExportUnknownFormat:

    def test_unknown_format_returns_422(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abc123/export?format=docx")
        assert resp.status_code == 422

    def test_unknown_format_error_mentions_format(self, export_client):
        client, *_ = export_client
        resp = client.get("/edit/abc123/export?format=html")
        detail = resp.json().get("detail", "")
        assert "html" in detail.lower() or "format" in detail.lower()


# ── Export endpoint — edit state not ready ───────────────────────────────────

class TestExportStateNotReady:

    def test_missing_edit_cycle_dict_returns_400(self):
        from app.api.routes.edit import router

        app = FastAPI()
        app.include_router(router)

        not_ready_state = MagicMock()
        not_ready_state.values = {"user_id": 1}  # no edit_cycle_dict

        mock_graph = MagicMock()
        mock_graph.get_state.return_value = not_ready_state

        with patch("app.api.routes.edit.edit_graph", mock_graph), \
             patch("app.api.routes.edit._rebuild_agent"), \
             patch("app.api.routes.edit.pl"):
            client = TestClient(app)
            resp = client.get("/edit/abc123/export")

        assert resp.status_code == 400

    def test_edit_cycle_dict_none_returns_400(self):
        from app.api.routes.edit import router

        app = FastAPI()
        app.include_router(router)

        not_ready_state = MagicMock()
        not_ready_state.values = {"user_id": 1, "edit_cycle_dict": None}

        mock_graph = MagicMock()
        mock_graph.get_state.return_value = not_ready_state

        with patch("app.api.routes.edit.edit_graph", mock_graph), \
             patch("app.api.routes.edit._rebuild_agent"), \
             patch("app.api.routes.edit.pl"):
            client = TestClient(app)
            resp = client.get("/edit/abc123/export")

        assert resp.status_code == 400


# ── custom_instruction forwarded through start_session ────────────────────────

class TestCustomInstructionInStartSession:

    def test_custom_instruction_added_to_initial_state(self):
        from app.api.routes.edit import router

        app = FastAPI()
        app.include_router(router)

        captured_state: dict = {}

        def _capture_invoke(initial_state, config):
            captured_state.update(initial_state)

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = _capture_invoke
        mock_state = MagicMock()
        mock_state.tasks = []
        mock_graph.get_state.return_value = mock_state

        with patch("app.api.routes.edit.edit_graph", mock_graph), \
             patch("app.api.routes.edit.register_thread"):
            client = TestClient(app)
            client.post("/edit/start", json={
                "user_id": 1,
                "jd_text": "Looking for a Python engineer",
                "custom_instruction": "Mirror the JD language exactly",
            })

        assert captured_state.get("custom_instruction") == "Mirror the JD language exactly"

    def test_whitespace_only_instruction_not_added(self):
        from app.api.routes.edit import router

        app = FastAPI()
        app.include_router(router)

        captured_state: dict = {}

        def _capture_invoke(initial_state, config):
            captured_state.update(initial_state)

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = _capture_invoke
        mock_state = MagicMock()
        mock_state.tasks = []
        mock_graph.get_state.return_value = mock_state

        with patch("app.api.routes.edit.edit_graph", mock_graph), \
             patch("app.api.routes.edit.register_thread"):
            client = TestClient(app)
            client.post("/edit/start", json={
                "user_id": 1,
                "jd_text": "Looking for a Python engineer",
                "custom_instruction": "   ",
            })

        assert "custom_instruction" not in captured_state
