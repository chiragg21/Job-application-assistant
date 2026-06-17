"""
Sprint 6 tests — auth middleware, JWT tokens, application tracker, skill gap analysis.

Covers:
  - AuthUser model
  - make_access_token (creates verifiable JWT)
  - get_current_user in DEV_AUTH mode
  - get_current_user with valid / expired / invalid JWT
  - GET /auth/me in DEV_AUTH mode
  - Applications _enrich (status_index)
  - Applications CRUD via TestClient (list, get, create, patch, delete, 404)
  - _extract_text_skills (LaTeX skill tokenizer)
  - GET /skills/gaps with empty sessions
  - GET /skills/gaps with mocked sessions and job data
"""

import json
import time
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch


# ── helpers ────────────────────────────────────────────────────────────────────

def _df(records):
    return pd.DataFrame(records) if records else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 1. AuthUser model
# ══════════════════════════════════════════════════════════════════════════════

class TestAuthUser:
    def test_requires_id(self):
        from app.api.middleware import AuthUser
        u = AuthUser(id=5)
        assert u.id == 5

    def test_defaults_all_optional_fields(self):
        from app.api.middleware import AuthUser
        u = AuthUser(id=1)
        assert u.email is None
        assert u.name is None
        assert u.avatar_url is None
        assert u.email_verified is False

    def test_sets_all_fields(self):
        from app.api.middleware import AuthUser
        u = AuthUser(id=2, email="a@b.com", name="Alice", avatar_url="http://x", email_verified=True)
        assert u.email == "a@b.com"
        assert u.email_verified is True


# ══════════════════════════════════════════════════════════════════════════════
# 2. make_access_token
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeAccessToken:
    def test_returns_string(self):
        from app.api.middleware import make_access_token
        token = make_access_token(1, "a@b.com", "Alice")
        assert isinstance(token, str)
        assert len(token) > 10

    def test_token_has_three_parts(self):
        from app.api.middleware import make_access_token
        token = make_access_token(1, "a@b.com", "Alice")
        assert len(token.split(".")) == 3   # header.payload.signature

    def test_decodes_to_correct_sub(self):
        import jwt
        from app.api.middleware import make_access_token, JWT_SECRET, JWT_ALG
        token = make_access_token(7, "user@x.com", "Bob")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        assert payload["sub"] == "7"

    def test_decodes_to_correct_email(self):
        import jwt
        from app.api.middleware import make_access_token, JWT_SECRET, JWT_ALG
        token = make_access_token(1, "test@test.com", "Test")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        assert payload["email"] == "test@test.com"

    def test_token_expires_in_30_days(self):
        import jwt
        from app.api.middleware import make_access_token, JWT_SECRET, JWT_ALG
        token = make_access_token(1, "a@b.com", "A")
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        expected = int(time.time()) + 60 * 60 * 24 * 30
        assert abs(payload["exp"] - expected) < 5     # within 5 seconds


# ══════════════════════════════════════════════════════════════════════════════
# 3. get_current_user
# ══════════════════════════════════════════════════════════════════════════════

class TestGetCurrentUser:
    def test_dev_auth_returns_user_id_1(self):
        with patch("app.api.middleware.DEV_AUTH", True):
            from app.api.middleware import get_current_user
            user = get_current_user(access_token=None)
        assert user.id == 1
        assert user.email == "dev@local"

    def test_dev_auth_ignores_token(self):
        with patch("app.api.middleware.DEV_AUTH", True):
            from app.api.middleware import get_current_user
            user = get_current_user(access_token="some-garbage-token")
        assert user.id == 1

    def test_no_token_raises_401_when_not_dev(self):
        from fastapi import HTTPException
        with patch("app.api.middleware.DEV_AUTH", False):
            from app.api.middleware import get_current_user
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(access_token=None)
        assert exc_info.value.status_code == 401

    def test_valid_token_returns_correct_user(self):
        import jwt
        from app.api.middleware import get_current_user, JWT_SECRET, JWT_ALG
        payload = {
            "sub": "5",
            "email": "jane@co.com",
            "name": "Jane",
            "picture": "",
            "email_verified": True,
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
        with patch("app.api.middleware.DEV_AUTH", False):
            user = get_current_user(access_token=token)
        assert user.id == 5
        assert user.email == "jane@co.com"

    def test_expired_token_raises_401(self):
        import jwt
        from fastapi import HTTPException
        from app.api.middleware import get_current_user, JWT_SECRET, JWT_ALG
        payload = {
            "sub": "1", "email": "x@y.com", "name": "X",
            "iat": int(time.time()) - 7200,
            "exp": int(time.time()) - 3600,   # already expired
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
        with patch("app.api.middleware.DEV_AUTH", False):
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(access_token=token)
        assert exc_info.value.status_code == 401
        assert "expired" in exc_info.value.detail.lower()

    def test_garbage_token_raises_401(self):
        from fastapi import HTTPException
        with patch("app.api.middleware.DEV_AUTH", False):
            from app.api.middleware import get_current_user
            with pytest.raises(HTTPException) as exc_info:
                get_current_user(access_token="not.a.jwt")
        assert exc_info.value.status_code == 401


# ══════════════════════════════════════════════════════════════════════════════
# 4. GET /auth/me (DEV_AUTH mode)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def auth_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes.auth import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app, raise_server_exceptions=False)


class TestAuthMeEndpoint:
    def test_me_returns_200_in_dev_auth(self, auth_client):
        with patch("app.api.middleware.DEV_AUTH", True):
            resp = auth_client.get("/auth/me")
        assert resp.status_code == 200

    def test_me_returns_dev_user(self, auth_client):
        with patch("app.api.middleware.DEV_AUTH", True):
            resp = auth_client.get("/auth/me")
        data = resp.json()
        assert data["id"] == 1
        assert data["email"] == "dev@local"

    def test_logout_clears_cookie(self, auth_client):
        resp = auth_client.post("/auth/logout")
        assert resp.status_code == 200
        assert resp.json()["logged_out"] is True


# ══════════════════════════════════════════════════════════════════════════════
# 5. Applications _enrich
# ══════════════════════════════════════════════════════════════════════════════

class TestApplicationsEnrich:
    def _enrich(self, row):
        from app.api.routes.applications import _enrich
        return _enrich(row)

    def test_applied_has_status_index_0(self):
        row = {"id": 1, "status": "applied"}
        assert self._enrich(row)["status_index"] == 0

    def test_offer_has_highest_valid_index(self):
        row = {"id": 2, "status": "offer"}
        assert self._enrich(row)["status_index"] == 4

    def test_rejected_status_index(self):
        row = {"id": 3, "status": "rejected"}
        assert self._enrich(row)["status_index"] == 5

    def test_unknown_status_defaults_to_0(self):
        row = {"id": 4, "status": "mystery"}
        assert self._enrich(row)["status_index"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. Applications CRUD via TestClient
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def apps_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes.applications import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_apps_db():
    db = MagicMock()
    db.fetch_table_where.return_value = _df([])
    db.fetch_one.return_value = None
    db.add_one.return_value = 1
    db.update_data.return_value = 1
    return db


class TestApplicationsCRUD:
    def _app_row(self, **kwargs):
        base = {
            "id": 1, "user_id": 1, "company": "Google", "role": "SWE",
            "status": "applied", "notes": None, "applied_at": "2024-01-01T00:00:00Z",
            "follow_up_at": None, "job_id": None, "resume_id": None,
            "session_id": None, "variant_id": None, "updated_at": "2024-01-01T00:00:00Z",
        }
        return {**base, **kwargs}

    def test_list_empty(self, apps_client, mock_apps_db):
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.get("/applications?user_id=1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_returns_items(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_table_where.return_value = _df([self._app_row()])
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.get("/applications?user_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["company"] == "Google"

    def test_create_returns_201(self, apps_client, mock_apps_db):
        mock_apps_db.add_one.return_value = 1
        mock_apps_db.fetch_one.return_value = self._app_row()
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.post("/applications", json={"company": "Google", "role": "SWE"})
        assert resp.status_code == 201

    def test_create_returns_application_with_status_index(self, apps_client, mock_apps_db):
        mock_apps_db.add_one.return_value = 1
        mock_apps_db.fetch_one.return_value = self._app_row()
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.post("/applications", json={"company": "Apple", "role": "iOS"})
        data = resp.json()
        assert "status_index" in data
        assert data["status_index"] == 0   # applied = index 0

    def test_create_requires_company_and_role(self, apps_client, mock_apps_db):
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.post("/applications", json={"company": "Google"})
        assert resp.status_code == 422   # pydantic validation error

    def test_get_application_200(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = self._app_row()
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.get("/applications/1")
        assert resp.status_code == 200
        assert resp.json()["company"] == "Google"

    def test_get_application_404(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = None
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.get("/applications/999")
        assert resp.status_code == 404

    def test_patch_application_updates_status(self, apps_client, mock_apps_db):
        original = self._app_row(status="applied")
        updated  = self._app_row(status="phone_screen")
        mock_apps_db.fetch_one.side_effect = [original, updated]
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.patch("/applications/1", json={"status": "phone_screen"})
        assert resp.status_code == 200
        assert resp.json()["status"] == "phone_screen"

    def test_patch_application_404(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = None
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.patch("/applications/999", json={"status": "offer"})
        assert resp.status_code == 404

    def test_patch_invalid_status_422(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = self._app_row()
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.patch("/applications/1", json={"status": "INVALID_STATUS"})
        assert resp.status_code == 422

    def test_delete_application_204(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = self._app_row()
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.delete("/applications/1")
        assert resp.status_code == 204

    def test_delete_application_404(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = None
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.delete("/applications/999")
        assert resp.status_code == 404

    def test_delete_calls_execute_raw(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_one.return_value = self._app_row()
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            apps_client.delete("/applications/1")
        mock_apps_db.execute_raw.assert_called_once()

    def test_status_index_in_list_response(self, apps_client, mock_apps_db):
        mock_apps_db.fetch_table_where.return_value = _df([
            self._app_row(status="offer"),
        ])
        with patch("app.api.routes.applications.SQLHandler", return_value=mock_apps_db):
            resp = apps_client.get("/applications")
        assert resp.json()[0]["status_index"] == 4   # offer


# ══════════════════════════════════════════════════════════════════════════════
# 7. _extract_text_skills
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractTextSkills:
    def _extract(self, text):
        from app.api.routes.skills import _extract_text_skills
        return _extract_text_skills(text)

    def test_extracts_comma_separated_skills(self):
        skills = self._extract("Python, JavaScript, TypeScript")
        assert "python" in skills
        assert "javascript" in skills

    def test_strips_latex_macros(self):
        skills = self._extract(r"\textbf{Python}, \textit{Docker}")
        assert "python" in skills
        assert "docker" in skills

    def test_excludes_short_tokens(self):
        skills = self._extract("Go, C, Python")
        assert "c" not in skills
        # "go" has 2 chars, should be excluded (len > 2 rule)
        assert "python" in skills

    def test_excludes_very_long_tokens(self):
        long_skill = "a" * 61
        skills = self._extract(long_skill)
        assert long_skill not in skills

    def test_pipe_separated_skills(self):
        skills = self._extract("React | Node.js | PostgreSQL")
        assert "react" in skills
        assert "postgresql" in skills

    def test_newline_separated_skills(self):
        skills = self._extract("Python\nDocker\nKubernetes")
        assert "docker" in skills
        assert "kubernetes" in skills

    def test_returns_lowercase(self):
        skills = self._extract("PYTHON, JavaScript")
        assert "python" in skills
        assert "javascript" in skills

    def test_returns_set(self):
        result = self._extract("Python, Python, Python")
        assert isinstance(result, set)
        assert len(result) == 1


# ══════════════════════════════════════════════════════════════════════════════
# 8. GET /skills/gaps
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def skills_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes.skills import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_skills_db():
    db = MagicMock()
    db.fetch_table_where.return_value = _df([])
    db.fetch_one.return_value = None
    return db


class TestSkillGapsEndpoint:
    def test_empty_sessions_returns_no_gaps(self, skills_client, mock_skills_db):
        with patch("app.api.routes.skills.SQLHandler", return_value=mock_skills_db):
            resp = skills_client.get("/skills/gaps?user_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["gaps"] == []
        assert data["total_jds"] == 0
        assert data["user_skills_count"] == 0

    def test_response_structure(self, skills_client, mock_skills_db):
        with patch("app.api.routes.skills.SQLHandler", return_value=mock_skills_db):
            resp = skills_client.get("/skills/gaps?user_id=1")
        data = resp.json()
        assert "gaps" in data
        assert "total_jds" in data
        assert "user_skills_count" in data

    def test_gaps_returned_when_sessions_exist(self, skills_client, mock_skills_db):
        jd_parsed = json.dumps({
            "required_skills": ["Kubernetes", "Terraform", "Rust"],
            "nice_to_have_skills": [],
        })
        mock_skills_db.fetch_table_where.side_effect = [
            _df([{"job_id": 10}]),    # edit_sessions
            _df([{"id": 1}]),          # resumes for user
        ]
        mock_skills_db.fetch_one.side_effect = [
            {"jd_parsed": jd_parsed},  # jobs row for job_id=10
            None,                       # resume_sections for skills (no match)
            {"jd_parsed": jd_parsed},  # _job_requires_skill check × 3
            {"jd_parsed": jd_parsed},
            {"jd_parsed": jd_parsed},
        ]
        with patch("app.api.routes.skills.SQLHandler", return_value=mock_skills_db):
            resp = skills_client.get("/skills/gaps?user_id=1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_jds"] == 1
        assert len(data["gaps"]) > 0

    def test_gaps_sorted_by_frequency_desc(self, skills_client, mock_skills_db):
        jd1 = json.dumps({"required_skills": ["Python", "Docker"], "nice_to_have_skills": []})
        jd2 = json.dumps({"required_skills": ["Python", "Rust"],   "nice_to_have_skills": []})
        mock_skills_db.fetch_table_where.side_effect = [
            _df([{"job_id": 1}, {"job_id": 2}]),  # 2 sessions
            _df([{"id": 1}]),                       # resumes
        ]
        # fetch_one: 2 job rows + resume_sections (no skills) + _job_requires_skill lookups
        mock_skills_db.fetch_one.side_effect = [
            {"jd_parsed": jd1},  # job 1
            {"jd_parsed": jd2},  # job 2
            None,                 # resume_sections (no user skills)
            # _job_requires_skill for each unique skill × 2 jobs each = many calls
        ] + [{"jd_parsed": jd1}, {"jd_parsed": jd2}] * 10
        with patch("app.api.routes.skills.SQLHandler", return_value=mock_skills_db):
            resp = skills_client.get("/skills/gaps?user_id=1")
        data = resp.json()
        if len(data["gaps"]) > 1:
            freqs = [g["frequency"] for g in data["gaps"]]
            assert freqs == sorted(freqs, reverse=True)

    def test_top_k_limits_results(self, skills_client, mock_skills_db):
        many_skills = [f"skill{i}" for i in range(50)]
        jd_parsed = json.dumps({"required_skills": many_skills, "nice_to_have_skills": []})
        mock_skills_db.fetch_table_where.side_effect = [
            _df([{"job_id": 1}]),
            _df([{"id": 1}]),
        ]
        mock_skills_db.fetch_one.side_effect = [
            {"jd_parsed": jd_parsed},
            None,
        ] + [{"jd_parsed": jd_parsed}] * 100
        with patch("app.api.routes.skills.SQLHandler", return_value=mock_skills_db):
            resp = skills_client.get("/skills/gaps?user_id=1&top_k=5")
        assert resp.status_code == 200
        assert len(resp.json()["gaps"]) <= 5
