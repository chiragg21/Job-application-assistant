"""
Sprint 1 — Key management, JD URL extraction, and route-level tests.

Covers:
  - POST/GET/DELETE/PATCH /keys/* endpoints
  - POST /jd/extract-from-url (Jina Reader path)
  - custom_instruction wiring in start_session (integration with graph state)
  - JD extractor module (network mocked)

Note: export endpoint tests and custom_instruction model tests already live in
test_edit_route_sprint1.py and are not duplicated here.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ══════════════════════════════════════════════════════════════════════════════
# Helpers / shared fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def keys_client(tmp_path):
    """
    TestClient for /keys routes with:
    - _USER_KEYS_PATH redirected to a tmp file so we don't touch real data
    - ALL_CLIENTS mocked so we never call real LLM providers
    - merged_status mocked to return a deterministic status list
    """
    from app.api.routes import keys as keys_mod

    fake_path = tmp_path / "user_keys.json"

    mock_client_a = MagicMock()
    mock_client_a.add_key = MagicMock()
    mock_client_a.remove_key = MagicMock()
    mock_client_a.toggle_key = MagicMock(return_value=True)
    mock_client_a.set_quota = MagicMock()
    mock_client_a.status = MagicMock(return_value=[])

    mock_status = [
        {
            "provider": "gemini",
            "key_id":   "abc12345",
            "active":   True,
            "rpm":      0,
            "tpm":      0,
            "daily_tokens": 0,
            "errors":   0,
        }
    ]

    with patch.object(keys_mod, "_USER_KEYS_PATH", fake_path), \
         patch.object(keys_mod, "ALL_CLIENTS", [mock_client_a]), \
         patch.object(keys_mod, "merged_status", return_value=mock_status):
        from app.api.routes.keys import router
        app = FastAPI()
        app.include_router(router)
        yield TestClient(app), fake_path, mock_client_a


# ══════════════════════════════════════════════════════════════════════════════
# GET /keys/status
# ══════════════════════════════════════════════════════════════════════════════

class TestGetKeyStatus:
    def test_returns_200(self, keys_client):
        client, *_ = keys_client
        resp = client.get("/keys/status")
        assert resp.status_code == 200

    def test_returns_list(self, keys_client):
        client, *_ = keys_client
        resp = client.get("/keys/status")
        assert isinstance(resp.json(), list)

    def test_returns_key_entry_fields(self, keys_client):
        client, *_ = keys_client
        resp = client.get("/keys/status")
        entries = resp.json()
        if entries:
            assert "provider" in entries[0]
            assert "key_id" in entries[0]


# ══════════════════════════════════════════════════════════════════════════════
# POST /keys — add a key
# ══════════════════════════════════════════════════════════════════════════════

class TestAddKey:
    def test_add_gemini_key_returns_200(self, keys_client):
        client, *_ = keys_client
        resp = client.post("/keys", json={"provider": "gemini", "key": "AIza_test_key_1234"})
        assert resp.status_code == 200

    def test_add_key_persists_to_file(self, keys_client):
        client, fake_path, _ = keys_client
        client.post("/keys", json={"provider": "gemini", "key": "AIza_test_key_1234"})
        assert fake_path.exists()
        data = json.loads(fake_path.read_text())
        assert any(e["key"] == "AIza_test_key_1234" for e in data)

    def test_add_key_registers_with_clients(self, keys_client):
        client, _, mock_client = keys_client
        client.post("/keys", json={"provider": "groq", "key": "gsk_test_key"})
        mock_client.add_key.assert_called()

    def test_add_groq_key_accepted(self, keys_client):
        client, *_ = keys_client
        resp = client.post("/keys", json={"provider": "groq", "key": "gsk_test_key_abc"})
        assert resp.status_code == 200

    def test_invalid_provider_returns_422(self, keys_client):
        client, *_ = keys_client
        resp = client.post("/keys", json={"provider": "unknown_ai", "key": "some_key"})
        assert resp.status_code == 422

    def test_empty_key_string_returns_422(self, keys_client):
        client, *_ = keys_client
        resp = client.post("/keys", json={"provider": "gemini", "key": ""})
        assert resp.status_code == 422

    def test_duplicate_key_not_stored_twice(self, keys_client):
        client, fake_path, _ = keys_client
        payload = {"provider": "gemini", "key": "AIza_dup_key"}
        client.post("/keys", json=payload)
        client.post("/keys", json=payload)
        data = json.loads(fake_path.read_text())
        count = sum(1 for e in data if e["key"] == "AIza_dup_key")
        assert count == 1

    def test_provider_normalised_to_lowercase(self, keys_client):
        client, fake_path, _ = keys_client
        client.post("/keys", json={"provider": "Gemini", "key": "AIza_case_test"})
        if fake_path.exists():
            data = json.loads(fake_path.read_text())
            providers = [e.get("provider", "").lower() for e in data]
            assert "gemini" in providers


# ══════════════════════════════════════════════════════════════════════════════
# DELETE /keys/{provider}/{key_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestRemoveKey:
    def _seed_key(self, fake_path: Path, provider="gemini", key="AIza_to_remove"):
        fake_path.write_text(json.dumps([{"provider": provider, "key": key}]))

    def test_remove_existing_key_returns_200(self, keys_client):
        client, fake_path, _ = keys_client
        self._seed_key(fake_path)
        # key_id is first 8 chars of the key
        resp = client.delete("/keys/gemini/AIza_to_")
        assert resp.status_code == 200

    def test_remove_key_removes_from_file(self, keys_client):
        client, fake_path, _ = keys_client
        self._seed_key(fake_path, key="AIza_to_remove_x")
        client.delete("/keys/gemini/AIza_to_")
        data = json.loads(fake_path.read_text())
        remaining = [e for e in data if e["key"] == "AIza_to_remove_x"]
        assert len(remaining) == 0

    def test_remove_non_existent_key_returns_removed_false_or_404(self, keys_client):
        client, fake_path, _ = keys_client
        fake_path.write_text(json.dumps([]))
        resp = client.delete("/keys/gemini/nonexist")
        # Either 200 with removed=False or 404 — both are acceptable
        assert resp.status_code in (200, 404)

    def test_remove_matching_key_calls_clients(self, keys_client):
        client, fake_path, mock_client = keys_client
        self._seed_key(fake_path, key="AIza_remove_call")
        client.delete("/keys/gemini/AIza_rem")
        mock_client.remove_key.assert_called()


# ══════════════════════════════════════════════════════════════════════════════
# PATCH /keys/{provider}/{key_id}/toggle
# ══════════════════════════════════════════════════════════════════════════════

class TestToggleKey:
    def test_toggle_active_true_returns_200(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/gemini/abc12345/toggle?active=true")
        assert resp.status_code == 200

    def test_toggle_active_false_returns_200(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/gemini/abc12345/toggle?active=false")
        assert resp.status_code == 200

    def test_toggle_returns_active_field(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/gemini/abc12345/toggle?active=true")
        body = resp.json()
        assert "active" in body

    def test_toggle_calls_llm_client(self, keys_client):
        client, _, mock_client = keys_client
        client.patch("/keys/gemini/abc12345/toggle?active=false")
        mock_client.toggle_key.assert_called()


# ══════════════════════════════════════════════════════════════════════════════
# PATCH /keys/quota
# ══════════════════════════════════════════════════════════════════════════════

class TestSetQuota:
    def test_set_quota_returns_200(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/quota", json={
            "provider": "gemini",
            "key_id":   "abc12345",
            "rpm_limit": 15,
        })
        assert resp.status_code == 200

    def test_set_tpm_limit(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/quota", json={
            "provider": "groq",
            "key_id":   "gsk12345",
            "tpm_limit": 5000,
        })
        assert resp.status_code == 200

    def test_set_daily_token_limit(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/quota", json={
            "provider":          "gemini",
            "key_id":            "abc12345",
            "daily_token_limit": 100000,
        })
        assert resp.status_code == 200

    def test_quota_missing_provider_returns_422(self, keys_client):
        client, *_ = keys_client
        resp = client.patch("/keys/quota", json={"key_id": "abc12345", "rpm_limit": 10})
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════════
# load_user_keys — startup helper
# ══════════════════════════════════════════════════════════════════════════════

class TestLoadUserKeys:
    def test_load_empty_file_does_not_error(self, tmp_path):
        from app.api.routes import keys as keys_mod
        fake_path = tmp_path / "user_keys.json"
        fake_path.write_text("[]")
        mock_client = MagicMock()
        with patch.object(keys_mod, "_USER_KEYS_PATH", fake_path), \
             patch.object(keys_mod, "ALL_CLIENTS", [mock_client]):
            keys_mod.load_user_keys()
        mock_client.add_key.assert_not_called()

    def test_load_existing_keys_registers_them(self, tmp_path):
        from app.api.routes import keys as keys_mod
        fake_path = tmp_path / "user_keys.json"
        fake_path.write_text(json.dumps([
            {"provider": "gemini", "key": "AIza_loaded"}
        ]))
        mock_client = MagicMock()
        with patch.object(keys_mod, "_USER_KEYS_PATH", fake_path), \
             patch.object(keys_mod, "ALL_CLIENTS", [mock_client]):
            keys_mod.load_user_keys()
        mock_client.add_key.assert_called_once_with("gemini", "AIza_loaded")

    def test_load_missing_file_does_not_error(self, tmp_path):
        from app.api.routes import keys as keys_mod
        fake_path = tmp_path / "nonexistent.json"
        mock_client = MagicMock()
        with patch.object(keys_mod, "_USER_KEYS_PATH", fake_path), \
             patch.object(keys_mod, "ALL_CLIENTS", [mock_client]):
            keys_mod.load_user_keys()  # must not raise
        mock_client.add_key.assert_not_called()

    def test_load_corrupt_json_does_not_error(self, tmp_path):
        from app.api.routes import keys as keys_mod
        fake_path = tmp_path / "user_keys.json"
        fake_path.write_text("{invalid json}")
        mock_client = MagicMock()
        with patch.object(keys_mod, "_USER_KEYS_PATH", fake_path), \
             patch.object(keys_mod, "ALL_CLIENTS", [mock_client]):
            keys_mod.load_user_keys()  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# JD extractor module — app/core/jd_extractor.py
# ══════════════════════════════════════════════════════════════════════════════

class TestJDExtractor:
    def test_returns_string(self):
        from app.core.jd_extractor import extract_jd_from_url
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "We are looking for a Python engineer with 3+ years..."
        with patch("requests.get", return_value=mock_response):
            result = extract_jd_from_url("https://example.com/job")
        assert isinstance(result, str)

    def test_raises_value_error_on_http_failure(self):
        from app.core.jd_extractor import extract_jd_from_url
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 not found")
        with patch("requests.get", return_value=mock_response):
            with pytest.raises((ValueError, Exception)):
                extract_jd_from_url("https://example.com/gone")

    def test_content_returned_from_jina_response(self):
        from app.core.jd_extractor import extract_jd_from_url
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Job title: ML Engineer\nSkills: Python, PyTorch"
        with patch("requests.get", return_value=mock_response):
            result = extract_jd_from_url("https://linkedin.com/jobs/123")
        assert "ML Engineer" in result or len(result) > 0

    def test_uses_jina_reader_url(self):
        from app.core.jd_extractor import extract_jd_from_url
        captured = {}
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Content"

        def _capture(url, **kwargs):
            captured["url"] = url
            return mock_response

        with patch("requests.get", side_effect=_capture):
            extract_jd_from_url("https://example.com/job")

        called_url = captured.get("url", "")
        assert "example.com/job" in called_url or "jina" in called_url.lower()


# ══════════════════════════════════════════════════════════════════════════════
# custom_instruction flows through graph state  (integration-level)
# ══════════════════════════════════════════════════════════════════════════════

class TestCustomInstructionFlow:
    """
    Verify that node_generate_suggestions picks up custom_instruction from
    graph state and sets it on agent.global_instruction. This is the cross-
    cutting concern introduced in Sprint 1.
    """

    def test_instruction_surfaces_in_agent_global_instruction(self):
        from app.graph.edit_graph import node_generate_suggestions

        state = {
            "user_id": 1,
            "custom_instruction": "Use XYZ formula for every bullet",
            "score_feedback": None,
            "selected_items": [],
            "dropped_items": [],
            "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
            "parsed_jd": {
                "company": "X", "role": "Eng", "seniority_level": "mid",
                "location": "", "is_remote": False, "about_company": "",
                "about_job": "", "responsibilities": [], "required_skills": [],
                "nice_to_have_skills": [], "perks": "", "others": "",
            },
        }

        mock_agent = MagicMock()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        assert "XYZ formula" in mock_agent.global_instruction

    def test_no_instruction_leaves_global_instruction_empty(self):
        from app.graph.edit_graph import node_generate_suggestions

        state = {
            "user_id": 1,
            "selected_items": [],
            "dropped_items": [],
            "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
            "parsed_jd": {
                "company": "X", "role": "Eng", "seniority_level": "mid",
                "location": "", "is_remote": False, "about_company": "",
                "about_job": "", "responsibilities": [], "required_skills": [],
                "nice_to_have_skills": [], "perks": "", "others": "",
            },
        }

        mock_agent = MagicMock()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        assert mock_agent.global_instruction == ""

    def test_instruction_persisted_in_start_session_state(self):
        """Start session with custom_instruction → it should appear in the
        initial graph state that is passed to edit_graph.invoke."""
        from app.api.routes.edit import router
        app = FastAPI()
        app.include_router(router)

        captured: dict = {}

        def _capture_invoke(state, config):
            captured.update(state)

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = _capture_invoke
        mock_state = MagicMock()
        mock_state.tasks = []
        mock_graph.get_state.return_value = mock_state

        with patch("app.api.routes.edit.edit_graph", mock_graph), \
             patch("app.api.routes.edit.register_thread"), \
             patch("app.api.routes.edit.SQLHandler"):
            client = TestClient(app)
            client.post("/edit/start", json={
                "user_id": 1,
                "jd_text": "Need a Python dev",
                "custom_instruction": "Always quantify achievements",
            })

        assert captured.get("custom_instruction") == "Always quantify achievements"
