"""Tests for app/api/routes/keys.py — key management endpoints."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    m = MagicMock()
    m.status.return_value = [
        {
            "provider":    "gemini",
            "key_id":      "AIza1234",
            "status":      "active",
            "rpm_limit":   15,
            "current_rpm": 2,
            "models":      [],
            "recent_meta": [],
        }
    ]
    return m


@pytest.fixture
def keys_client(mock_llm, tmp_path):
    """Minimal FastAPI app with only the keys router + all heavy deps mocked."""
    from app.api.routes.keys import router
    app = FastAPI()
    app.include_router(router)
    user_keys_path = tmp_path / "user_keys.json"
    with patch("app.api.routes.keys.llm", mock_llm), \
         patch("app.api.routes.keys._USER_KEYS_PATH", user_keys_path):
        yield TestClient(app), mock_llm, user_keys_path


# ── GET /keys/status ──────────────────────────────────────────────────────────

class TestGetKeyStatus:

    def test_returns_200(self, keys_client):
        client, _, _ = keys_client
        resp = client.get("/keys/status")
        assert resp.status_code == 200

    def test_returns_list(self, keys_client):
        client, _, _ = keys_client
        resp = client.get("/keys/status")
        assert isinstance(resp.json(), list)

    def test_calls_llm_status_no_filter(self, keys_client):
        client, mock_llm, _ = keys_client
        client.get("/keys/status")
        mock_llm.status.assert_called_once_with(provider=None)

    def test_provider_filter_forwarded_to_status(self, keys_client):
        client, mock_llm, _ = keys_client
        mock_llm.status.return_value = []
        client.get("/keys/status?provider=gemini")
        mock_llm.status.assert_called_once_with(provider="gemini")

    def test_status_error_returns_500(self, keys_client):
        client, mock_llm, _ = keys_client
        mock_llm.status.side_effect = RuntimeError("internal error")
        resp = client.get("/keys/status")
        assert resp.status_code == 500


# ── PATCH /keys/quota ─────────────────────────────────────────────────────────

class TestPatchQuota:

    def test_valid_quota_returns_updated_true(self, keys_client):
        client, _, _ = keys_client
        resp = client.patch("/keys/quota", json={
            "provider": "gemini", "key_id": "AIza1234",
            "rpm_limit": 10, "daily_token_limit": 1_000_000,
        })
        assert resp.status_code == 200
        assert resp.json()["updated"] is True

    def test_calls_set_quota_once(self, keys_client):
        client, mock_llm, _ = keys_client
        client.patch("/keys/quota", json={
            "provider": "gemini", "key_id": "AIza1234", "rpm_limit": 10,
        })
        mock_llm.set_quota.assert_called_once()

    def test_model_scope_forwarded(self, keys_client):
        import sys
        client, mock_llm, _ = keys_client
        # infrakit.llm.QuotaConfig is a MagicMock (stubbed in conftest).
        # Verify it was constructed with the right keyword args.
        mock_quota_cls = sys.modules["infrakit.llm"].QuotaConfig
        mock_quota_cls.reset_mock()
        client.patch("/keys/quota", json={
            "provider": "gemini", "key_id": "AIza1234",
            "model": "gemini-2.0-flash", "daily_token_limit": 500_000,
        })
        _, ctor_kwargs = mock_quota_cls.call_args
        assert ctor_kwargs.get("model") == "gemini-2.0-flash"
        assert ctor_kwargs.get("daily_token_limit") == 500_000

    def test_set_quota_error_returns_400(self, keys_client):
        client, mock_llm, _ = keys_client
        mock_llm.set_quota.side_effect = ValueError("key not found")
        resp = client.patch("/keys/quota", json={
            "provider": "gemini", "key_id": "DOESNOTEXIST", "rpm_limit": 5,
        })
        assert resp.status_code == 400


# ── POST /keys ────────────────────────────────────────────────────────────────

class TestAddKey:

    def test_valid_gemini_key_returns_added_true(self, keys_client):
        client, _, _ = keys_client
        resp = client.post("/keys", json={"provider": "gemini", "key": "AIzaTestKey1234"})
        assert resp.status_code == 200
        assert resp.json()["added"] is True

    def test_valid_groq_key_accepted(self, keys_client):
        client, _, _ = keys_client
        resp = client.post("/keys", json={"provider": "groq", "key": "gsk_testkey9876"})
        assert resp.status_code == 200

    def test_calls_llm_add_key(self, keys_client):
        client, mock_llm, _ = keys_client
        client.post("/keys", json={"provider": "gemini", "key": "AIzaTestKey1234"})
        mock_llm.add_key.assert_called_once_with("gemini", "AIzaTestKey1234")

    def test_key_persisted_to_json_file(self, keys_client):
        client, _, user_keys_path = keys_client
        client.post("/keys", json={"provider": "groq", "key": "gsk_persisted123"})
        data = json.loads(user_keys_path.read_text())
        assert any(e["provider"] == "groq" and e["key"] == "gsk_persisted123" for e in data)

    def test_invalid_provider_returns_422(self, keys_client):
        client, _, _ = keys_client
        resp = client.post("/keys", json={"provider": "anthropic", "key": "sk-ant-123"})
        assert resp.status_code == 422

    def test_empty_key_returns_422(self, keys_client):
        client, _, _ = keys_client
        resp = client.post("/keys", json={"provider": "gemini", "key": "   "})
        assert resp.status_code == 422

    def test_duplicate_key_not_stored_twice(self, keys_client):
        client, _, user_keys_path = keys_client
        payload = {"provider": "gemini", "key": "AIzaDuplicateKey"}
        client.post("/keys", json=payload)
        client.post("/keys", json=payload)
        data = json.loads(user_keys_path.read_text())
        matches = [e for e in data if e["key"] == "AIzaDuplicateKey"]
        assert len(matches) == 1

    def test_provider_normalised_to_lowercase(self, keys_client):
        client, mock_llm, _ = keys_client
        client.post("/keys", json={"provider": "Gemini", "key": "AIzaMixedCase"})
        mock_llm.add_key.assert_called_once_with("gemini", "AIzaMixedCase")


# ── DELETE /keys/{provider}/{key_id} ─────────────────────────────────────────

class TestRemoveKey:

    def test_remove_returns_removed_true(self, keys_client):
        client, _, _ = keys_client
        resp = client.delete("/keys/gemini/AIza1234")
        assert resp.status_code == 200
        assert resp.json()["removed"] is True

    def test_calls_llm_remove_key(self, keys_client):
        client, mock_llm, _ = keys_client
        client.delete("/keys/gemini/AIza1234")
        mock_llm.remove_key.assert_called_once_with("gemini", "AIza1234")

    def test_matching_key_removed_from_file(self, keys_client):
        client, _, user_keys_path = keys_client
        user_keys_path.write_text(json.dumps([
            {"provider": "gemini", "key": "AIza1234abcdef"},
            {"provider": "groq",   "key": "gsk_other"},
        ]))
        client.delete("/keys/gemini/AIza1234")
        data = json.loads(user_keys_path.read_text())
        assert not any(e["key"].startswith("AIza1234") and e["provider"] == "gemini" for e in data)

    def test_non_matching_key_preserved_in_file(self, keys_client):
        client, _, user_keys_path = keys_client
        user_keys_path.write_text(json.dumps([
            {"provider": "groq", "key": "gsk_other"},
        ]))
        client.delete("/keys/gemini/AIza1234")
        data = json.loads(user_keys_path.read_text())
        assert any(e["key"] == "gsk_other" for e in data)


# ── load_user_keys ────────────────────────────────────────────────────────────

class TestLoadUserKeys:

    def test_loads_all_persisted_keys(self, tmp_path):
        mock_llm = MagicMock()
        user_keys_path = tmp_path / "user_keys.json"
        user_keys_path.write_text(json.dumps([
            {"provider": "gemini", "key": "AIzaPersistedKey1"},
            {"provider": "groq",   "key": "gsk_PersistedKey2"},
        ]))
        with patch("app.api.routes.keys.llm", mock_llm), \
             patch("app.api.routes.keys._USER_KEYS_PATH", user_keys_path):
            from app.api.routes.keys import load_user_keys
            load_user_keys()
        assert mock_llm.add_key.call_count == 2
        mock_llm.add_key.assert_any_call("gemini", "AIzaPersistedKey1")
        mock_llm.add_key.assert_any_call("groq",   "gsk_PersistedKey2")

    def test_missing_file_does_not_raise(self, tmp_path):
        mock_llm = MagicMock()
        user_keys_path = tmp_path / "nonexistent.json"
        with patch("app.api.routes.keys.llm", mock_llm), \
             patch("app.api.routes.keys._USER_KEYS_PATH", user_keys_path):
            from app.api.routes.keys import load_user_keys
            load_user_keys()   # must not raise
        mock_llm.add_key.assert_not_called()

    def test_failed_add_key_does_not_crash(self, tmp_path):
        mock_llm = MagicMock()
        mock_llm.add_key.side_effect = RuntimeError("bad key")
        user_keys_path = tmp_path / "user_keys.json"
        user_keys_path.write_text(json.dumps([
            {"provider": "gemini", "key": "AIzaBadKey"},
        ]))
        with patch("app.api.routes.keys.llm", mock_llm), \
             patch("app.api.routes.keys._USER_KEYS_PATH", user_keys_path):
            from app.api.routes.keys import load_user_keys
            load_user_keys()   # must not raise even if add_key fails
