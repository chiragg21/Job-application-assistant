"""
Sprint 5 tests — resume variant library, optimizer, quick-resume routing.

Covers:
  - _slugify helper
  - VariantStore._enrich (JSON column parsing)
  - VariantStore.save  (filesystem + DB mock)
  - VariantStore.list / get / delete (DB mock)
  - VariantStore.search_by_jd fallback (ChromaDB failure → all variants)
  - route_after_build_edit_state (quick vs full)
  - Library API routes via TestClient (list, get, save, delete, 404 paths)
  - Optimizer: _remove_item_from_latex (dict-based item), _compile_and_count
"""

import json
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_df(records):
    import pandas as pd
    return pd.DataFrame(records) if records else pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# 1. _slugify
# ══════════════════════════════════════════════════════════════════════════════

class TestSlugify:
    def _fn(self, text):
        from app.core.variant_store import _slugify
        return _slugify(text)

    def test_spaces_to_underscores(self):
        assert self._fn("My Resume") == "my_resume"

    def test_special_chars_removed(self):
        result = self._fn("résumé v2!")
        assert all(c in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in result)

    def test_truncated_to_40(self):
        assert len(self._fn("a" * 100)) <= 40

    def test_no_leading_underscore(self):
        assert not self._fn("  hello world  ").startswith("_")

    def test_empty_string(self):
        assert self._fn("") == ""

    def test_already_valid(self):
        assert self._fn("my_cv") == "my_cv"


# ══════════════════════════════════════════════════════════════════════════════
# 2. VariantStore._enrich
# ══════════════════════════════════════════════════════════════════════════════

class TestVariantStoreEnrich:
    def _enrich(self, row):
        from app.core.variant_store import VariantStore
        return VariantStore._enrich(row)

    def test_parses_tags_json_string(self):
        row = {"tags": '["python", "ml"]', "job_ids": "[]", "session_ids": "[]"}
        assert self._enrich(row)["tags"] == ["python", "ml"]

    def test_parses_job_ids(self):
        row = {"tags": "[]", "job_ids": "[1, 2, 3]", "session_ids": "[]"}
        assert self._enrich(row)["job_ids"] == [1, 2, 3]

    def test_invalid_json_becomes_empty_list(self):
        row = {"tags": "bad json{", "job_ids": "[]", "session_ids": "[]"}
        assert self._enrich(row)["tags"] == []

    def test_already_list_is_left_alone(self):
        row = {"tags": ["already", "a", "list"], "job_ids": "[]", "session_ids": "[]"}
        assert self._enrich(row)["tags"] == ["already", "a", "list"]

    def test_missing_field_not_crashed(self):
        self._enrich({"tags": "[]"})   # no raise


# ══════════════════════════════════════════════════════════════════════════════
# 3. VariantStore.save  (uses a real temp dir; DB mocked)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def base_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store_ctx(base_dir):
    """
    Yields (VariantStore, mock_db).
    Patches app.core.variant_store.cfg with a config that points at base_dir,
    mocks the DB, and mocks _compile_latex.  Patches remain active for the
    entire test via yield-inside-with.
    """
    cfg = {
        "path_dir": {
            "data_dir": str(base_dir),   # VariantStore.__init__: self.base_dir = data_dir/"resumes"
            "base_dir": str(base_dir),   # VariantStore.save: rel_path relative to base_dir
        },
        "chromadb": {"collection1": "jd_chunks"},
        "sqlite":   {"sql_path": "/tmp/test.db"},
    }
    mock_db = MagicMock()
    mock_db.add_one.return_value = 42

    with patch("app.core.variant_store.cfg", cfg), \
         patch("app.core.variant_store.SQLHandler", return_value=mock_db), \
         patch("app.core.variant_store._compile_latex", return_value=(None, 1)):
        from app.core.variant_store import VariantStore
        s = VariantStore()
        yield s, mock_db


class TestVariantStoreSave:
    """Files land in base_dir/resumes/user_{id}/{row_id}__{slug}/."""

    def test_returns_dict_with_id(self, store_ctx):
        s, _ = store_ctx
        meta = s.save(user_id=1, name="Test Resume", latex="\\documentclass{article}")
        assert meta["id"] == 42

    def test_creates_tex_file(self, store_ctx, base_dir):
        s, _ = store_ctx
        s.save(user_id=1, name="My CV", latex="\\documentclass{article}")
        tex_files = list((base_dir / "resumes").glob("user_1/**/*.tex"))
        assert len(tex_files) == 1

    def test_creates_meta_json(self, store_ctx, base_dir):
        s, _ = store_ctx
        s.save(user_id=1, name="My CV", latex="\\documentclass{article}")
        meta_files = list((base_dir / "resumes").glob("user_1/**/meta.json"))
        assert len(meta_files) == 1
        meta = json.loads(meta_files[0].read_text())
        assert meta["name"] == "My CV"

    def test_db_insert_called_with_placeholder(self, store_ctx):
        s, mock_db = store_ctx
        s.save(user_id=1, name="CV", latex="x")
        call_data = mock_db.add_one.call_args[0][1]
        assert call_data["folder_path"] == "__placeholder__"

    def test_db_update_called_with_real_path(self, store_ctx):
        s, mock_db = store_ctx
        s.save(user_id=1, name="CV", latex="x")
        mock_db.update_data.assert_called_once()
        update_data = mock_db.update_data.call_args[0][1]
        assert "folder_path" in update_data
        assert update_data["folder_path"] != "__placeholder__"

    def test_folder_named_with_id_and_slug(self, store_ctx, base_dir):
        s, _ = store_ctx
        s.save(user_id=1, name="Data Science Resume", latex="x")
        user_dir = base_dir / "resumes" / "user_1"
        folders = [f.name for f in user_dir.iterdir() if f.is_dir()]
        assert any(f.startswith("42__") for f in folders)

    def test_tags_and_job_ids_in_meta(self, store_ctx, base_dir):
        s, _ = store_ctx
        s.save(user_id=1, name="T", latex="x", tags=["ml"], job_ids=[7])
        meta_files = list((base_dir / "resumes").glob("user_1/**/meta.json"))
        meta = json.loads(meta_files[0].read_text())
        assert meta["tags"] == ["ml"]
        assert meta["job_ids"] == [7]

    def test_page_count_in_meta(self, store_ctx, base_dir):
        s, _ = store_ctx
        # _compile_latex is mocked to return (None, 1)
        meta = s.save(user_id=1, name="PC", latex="x")
        assert meta["page_count"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# 4. VariantStore.list / get / delete  (DB mocked; no real files for list/get)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def read_store_ctx(base_dir):
    cfg = {
        "path_dir": {"data_dir": str(base_dir), "base_dir": str(base_dir)},
        "chromadb": {"collection1": "jd_chunks"},
        "sqlite":   {"sql_path": "/tmp/test.db"},
    }
    mock_db = MagicMock()
    with patch("app.core.variant_store.cfg", cfg), \
         patch("app.core.variant_store.SQLHandler", return_value=mock_db):
        from app.core.variant_store import VariantStore
        s = VariantStore()
        yield s, mock_db


class TestVariantStoreRead:
    def test_list_returns_enriched_rows(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_table_where.return_value = _make_df([
            {"id": 1, "user_id": 1, "name": "A", "tags": '["a"]', "job_ids": "[]", "session_ids": "[]"},
        ])
        result = s.list(user_id=1)
        assert len(result) == 1
        assert result[0]["tags"] == ["a"]

    def test_list_empty(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_table_where.return_value = _make_df([])
        assert s.list(user_id=1) == []

    def test_get_returns_none_when_not_found(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_one.return_value = None
        assert s.get(999) is None

    def test_get_returns_enriched_row(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_one.return_value = {
            "id": 5, "name": "B", "tags": '["x"]', "job_ids": "[1]", "session_ids": "[]"
        }
        result = s.get(5)
        assert result["id"] == 5
        assert result["tags"] == ["x"]


class TestVariantStoreDelete:
    def test_delete_returns_false_when_not_found(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_one.return_value = None
        assert s.delete(999) is False

    def test_delete_calls_execute_raw(self, read_store_ctx, base_dir):
        s, mock_db = read_store_ctx
        # Use a relative path that resolves under base_dir
        mock_db.fetch_one.return_value = {"id": 5, "folder_path": "resumes/user_1/5__test"}
        variant_dir = base_dir / "resumes" / "user_1" / "5__test"
        variant_dir.mkdir(parents=True)
        result = s.delete(5)
        assert result is True
        mock_db.execute_raw.assert_called_once()

    def test_delete_removes_folder(self, read_store_ctx, base_dir):
        s, mock_db = read_store_ctx
        mock_db.fetch_one.return_value = {"id": 5, "folder_path": "resumes/user_1/5__test"}
        variant_dir = base_dir / "resumes" / "user_1" / "5__test"
        variant_dir.mkdir(parents=True)
        s.delete(5)
        assert not variant_dir.exists()


# ══════════════════════════════════════════════════════════════════════════════
# 5. VariantStore.search_by_jd — ChromaDB failure fallback
# ══════════════════════════════════════════════════════════════════════════════

class TestVariantStoreSearchByJd:
    def test_fallback_returns_list_on_chroma_error(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_table_where.return_value = _make_df([
            {"id": 1, "name": "A", "tags": "[]", "job_ids": "[]", "session_ids": "[]"},
            {"id": 2, "name": "B", "tags": "[]", "job_ids": "[]", "session_ids": "[]"},
        ])
        # Patch chroma_client at source so the local import inside search_by_jd throws
        with patch("app.core.chroma_client.get_chroma_client", side_effect=Exception("no chroma")):
            result = s.search_by_jd("ML engineer", user_id=1, top_k=10)
        assert isinstance(result, list)

    def test_fallback_respects_top_k(self, read_store_ctx):
        s, mock_db = read_store_ctx
        variants = [
            {"id": i, "name": f"V{i}", "tags": "[]", "job_ids": "[]", "session_ids": "[]"}
            for i in range(20)
        ]
        mock_db.fetch_table_where.return_value = _make_df(variants)
        with patch("app.core.chroma_client.get_chroma_client", side_effect=Exception("no chroma")):
            result = s.search_by_jd("anything", user_id=1, top_k=5)
        assert len(result) <= 5

    def test_empty_library_returns_empty(self, read_store_ctx):
        s, mock_db = read_store_ctx
        mock_db.fetch_table_where.return_value = _make_df([])
        with patch("app.core.chroma_client.get_chroma_client", side_effect=Exception("no chroma")):
            result = s.search_by_jd("test", user_id=1, top_k=10)
        assert result == []


# ══════════════════════════════════════════════════════════════════════════════
# 6. Quick-resume routing (edit_graph)
# ══════════════════════════════════════════════════════════════════════════════

class TestQuickResumeRouting:
    def _route(self, state):
        from app.graph.edit_graph import route_after_build_edit_state
        return route_after_build_edit_state(state)

    def test_quick_true_routes_to_quick(self):
        assert self._route({"quick_resume": True}) == "quick"

    def test_quick_false_routes_to_full(self):
        assert self._route({"quick_resume": False}) == "full"

    def test_quick_missing_routes_to_full(self):
        assert self._route({}) == "full"

    def test_quick_none_routes_to_full(self):
        assert self._route({"quick_resume": None}) == "full"

    def test_truthy_value_routes_to_quick(self):
        assert self._route({"quick_resume": 1}) == "quick"


# ══════════════════════════════════════════════════════════════════════════════
# 7. Library API routes
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def library_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from app.api.routes.library import router
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.list.return_value = []
    store.get.return_value = None
    store.save.return_value = {
        "id": 1, "name": "Test", "variant_type": "custom",
        "folder_path": "user_1/1__test", "page_count": 1,
        "tags": [], "job_ids": [], "session_ids": [],
    }
    store.delete.return_value = True
    store.search_by_jd.return_value = []
    store.get_latex.return_value = None
    store.get_pdf_path.return_value = None
    return store


class TestLibraryRoutes:
    def test_list_variants_200(self, library_client, mock_store):
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants?user_id=1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_variants_returns_items(self, library_client, mock_store):
        mock_store.list.return_value = [{"id": 1, "name": "My CV", "tags": [], "job_ids": [], "session_ids": []}]
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants?user_id=1")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_get_variant_404(self, library_client, mock_store):
        mock_store.get.return_value = None
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants/999")
        assert resp.status_code == 404

    def test_get_variant_200(self, library_client, mock_store):
        mock_store.get.return_value = {"id": 5, "name": "CV", "tags": [], "job_ids": [], "session_ids": []}
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants/5")
        assert resp.status_code == 200
        assert resp.json()["id"] == 5

    def test_save_variant_201(self, library_client, mock_store):
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.post("/library/variants", json={
                "user_id": 1, "name": "New CV", "latex": "\\documentclass{article}"
            })
        assert resp.status_code == 201

    def test_save_variant_passes_tags(self, library_client, mock_store):
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            library_client.post("/library/variants", json={
                "user_id": 1, "name": "New CV", "latex": "x",
                "tags": ["backend"], "job_ids": [3],
            })
        call_kwargs = mock_store.save.call_args[1]
        assert call_kwargs["tags"] == ["backend"]
        assert call_kwargs["job_ids"] == [3]

    def test_delete_variant_204(self, library_client, mock_store):
        mock_store.delete.return_value = True
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.delete("/library/variants/1")
        assert resp.status_code == 204

    def test_delete_variant_404(self, library_client, mock_store):
        mock_store.delete.return_value = False
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.delete("/library/variants/999")
        assert resp.status_code == 404

    def test_search_endpoint_200(self, library_client, mock_store):
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/search?jd_text=ML+engineer&user_id=1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_download_tex_404_when_no_latex(self, library_client, mock_store):
        mock_store.get_latex.return_value = None
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants/1/download?format=tex")
        assert resp.status_code == 404

    def test_download_tex_returns_content(self, library_client, mock_store):
        mock_store.get_latex.return_value = "\\documentclass{article}"
        mock_store.get.return_value = {"id": 1, "name": "My CV", "tags": [], "job_ids": [], "session_ids": []}
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants/1/download?format=tex")
        assert resp.status_code == 200
        assert "documentclass" in resp.text

    def test_download_pdf_404_when_no_pdf(self, library_client, mock_store):
        mock_store.get_pdf_path.return_value = None
        with patch("app.api.routes.library.VariantStore", return_value=mock_store):
            resp = library_client.get("/library/variants/1/download?format=pdf")
        assert resp.status_code == 404


# ══════════════════════════════════════════════════════════════════════════════
# 8. Optimizer graph helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestOptimizerHelpers:
    def _item(self, content_latex: str) -> dict:
        """Build a ranked_item dict as the optimizer uses it."""
        return {"rows": [{"content_latex": content_latex}]}

    def test_remove_item_deletes_matching_block(self):
        from app.graph.optimizer_graph import _remove_item_from_latex
        block = r"\resumeItem{Project X}{Built a thing}"
        latex = f"{block}\n\\resumeItem{{Project Y}}{{Another thing}}"
        result = _remove_item_from_latex(latex, self._item(block))
        assert "Project X" not in result
        assert "Project Y" in result

    def test_remove_item_noop_when_not_found(self):
        from app.graph.optimizer_graph import _remove_item_from_latex
        latex = r"\resumeItem{Project Y}{Another thing}"
        item = self._item(r"\resumeItem{NONEXISTENT}{x}")
        assert _remove_item_from_latex(latex, item) == latex

    def test_remove_item_noop_when_no_rows(self):
        from app.graph.optimizer_graph import _remove_item_from_latex
        latex = r"\resumeItem{Project Y}{Another thing}"
        result = _remove_item_from_latex(latex, {"rows": []})
        assert result == latex

    def test_compile_and_count_returns_tuple(self):
        from app.graph.optimizer_graph import _compile_and_count
        with patch("app.graph.optimizer_graph.subprocess.run") as mock_run:
            mock_run.return_value.stdout = "Output written on resume.pdf (2 pages, 1234 bytes)"
            mock_run.return_value.returncode = 0
            _, page_count = _compile_and_count("\\documentclass{article}")
        assert isinstance(page_count, int)
        assert page_count >= 1

    def test_compile_and_count_defaults_to_1_on_missing_pdflatex(self):
        from app.graph.optimizer_graph import _compile_and_count
        with patch("app.graph.optimizer_graph.subprocess.run", side_effect=FileNotFoundError("no pdflatex")):
            _, page_count = _compile_and_count("\\documentclass{article}")
        assert page_count == 1

    def test_compile_and_count_defaults_to_1_on_timeout(self):
        import subprocess
        from app.graph.optimizer_graph import _compile_and_count
        with patch("app.graph.optimizer_graph.subprocess.run",
                   side_effect=subprocess.TimeoutExpired("pdflatex", 45)):
            _, page_count = _compile_and_count("\\documentclass{article}")
        assert page_count == 1
