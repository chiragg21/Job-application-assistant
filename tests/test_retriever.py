"""Tests for app/core/retriever.py"""
import json
import pytest
from unittest.mock import MagicMock, patch

from app.core.retriever import ResumeRetriever
from app.models.jd import ParsedJD


# ── Fixture: retriever with __init__ bypassed ─────────────────────────────────

@pytest.fixture
def retriever():
    """
    Create a ResumeRetriever skipping __init__ so no DB/Chroma connections
    are made. Weights and top_k are set manually.
    """
    r = ResumeRetriever.__new__(ResumeRetriever)
    r._n_resp     = 0
    r.top_k       = 5
    r.fetch_mult  = 3
    r.weight_skill = 0.5
    r.weight_resp  = 0.3
    r.weight_nth   = 0.2
    r.db           = MagicMock()
    r.col_resume   = MagicMock()
    r.col_jd       = MagicMock()
    r._chroma_client = MagicMock()
    return r


# ── _rerank ───────────────────────────────────────────────────────────────────

class TestRerank:
    def _make_result(self, ids, distances, metadatas):
        """Helper: build the dict that ChromaDB returns for multiple queries."""
        return {
            "ids":       ids,
            "distances": distances,
            "metadatas": metadatas,
        }

    def test_returns_list_of_tuples(self, retriever):
        chroma_result = self._make_result(
            ids=[["doc_A"], ["doc_A"]],
            distances=[[0.1], [0.2]],
            metadatas=[[{"section": "experience"}], [{"section": "experience"}]],
        )
        result = retriever._rerank(chroma_result, n_responsibilities=1, top_k=5)
        assert isinstance(result, list)
        assert all(len(t) == 3 for t in result)

    def test_higher_required_match_ranks_first(self, retriever):
        """
        doc_A appears in required_skills (high-weight group).
        doc_B appears only in nice_to_have (low-weight group).
        doc_A should rank higher.
        """
        chroma_result = self._make_result(
            ids=[
                ["doc_A"],       # required_skills query
                ["doc_A"],       # responsibility query
                ["doc_B"],       # nice_to_have query
            ],
            distances=[
                [0.1],           # doc_A sim=0.9
                [0.1],           # doc_A sim=0.9
                [0.1],           # doc_B sim=0.9
            ],
            metadatas=[
                [{"section": "experience"}],
                [{"section": "experience"}],
                [{"section": "projects"}],
            ],
        )
        result = retriever._rerank(chroma_result, n_responsibilities=1, top_k=5)
        ids_in_order = [cid for _, cid, _ in result]
        assert ids_in_order[0] == "doc_A"

    def test_top_k_limits_results(self, retriever):
        n = 10
        chroma_result = self._make_result(
            ids=[[f"doc_{i}" for i in range(n)]],
            distances=[[0.1 * i for i in range(n)]],
            metadatas=[[{"section": "experience"}] * n],
        )
        result = retriever._rerank(chroma_result, n_responsibilities=0, top_k=3)
        assert len(result) <= 3

    def test_empty_chroma_result_returns_empty(self, retriever):
        chroma_result = self._make_result(ids=[], distances=[], metadatas=[])
        result = retriever._rerank(chroma_result, n_responsibilities=0, top_k=5)
        assert result == []

    def test_scores_are_non_negative(self, retriever):
        chroma_result = self._make_result(
            ids=[["doc_A", "doc_B"]],
            distances=[[0.3, 0.7]],
            metadatas=[[{"section": "s"}, {"section": "s"}]],
        )
        result = retriever._rerank(chroma_result, n_responsibilities=0, top_k=5)
        assert all(score >= 0.0 for score, _, _ in result)

    def test_result_sorted_descending(self, retriever):
        chroma_result = self._make_result(
            ids=[["doc_A", "doc_B", "doc_C"]],
            distances=[[0.5, 0.2, 0.8]],   # sims: 0.5, 0.8, 0.2
            metadatas=[[{"s": "e"}, {"s": "e"}, {"s": "e"}]],
        )
        result = retriever._rerank(chroma_result, n_responsibilities=0, top_k=5)
        scores = [s for s, _, _ in result]
        assert scores == sorted(scores, reverse=True)

    def test_metadata_preserved_in_result(self, retriever):
        meta = {"section": "experience", "item_name": "Google"}
        chroma_result = self._make_result(
            ids=[["doc_A"]],
            distances=[[0.1]],
            metadatas=[[meta]],
        )
        result = retriever._rerank(chroma_result, n_responsibilities=0, top_k=5)
        assert result[0][2] == meta


# ── _fetch_jd ─────────────────────────────────────────────────────────────────

class TestFetchJD:
    def test_returns_none_when_not_found(self, retriever):
        retriever.db.fetch_one.return_value = None
        assert retriever._fetch_jd(job_id=99) is None

    def test_returns_parsed_jd_when_found(self, retriever, sample_jd):
        retriever.db.fetch_one.return_value = {
            "jd_parsed": json.dumps(sample_jd.to_dict())
        }
        result = retriever._fetch_jd(job_id=1)
        assert isinstance(result, ParsedJD)
        assert result.company == sample_jd.company

    def test_fetches_from_jobs_table(self, retriever):
        retriever.db.fetch_one.return_value = None
        retriever._fetch_jd(job_id=5)
        retriever.db.fetch_one.assert_called_once_with(
            table_name="jobs", filters={"id": 5}
        )


# ── retrieve_top_results (via mocked collection) ──────────────────────────────

class TestRetrieveTopResults:
    def _chroma_query_response(self, doc_ids):
        return {
            "ids":       [doc_ids] * 3,  # 3 queries (req, resp, nth)
            "distances": [[0.1] * len(doc_ids)] * 3,
            "metadatas": [[{"section": "experience", "item_name": "Google",
                             "sql_id": str(i)} for i in range(len(doc_ids))]] * 3,
        }

    def test_calls_chroma_query(self, retriever):
        retriever.col_resume.query.return_value = self._chroma_query_response(["doc_1"])
        retriever.retrieve_top_results(
            text=["Python", "ML pipelines", "LangChain"],
            section_name="experience",
            item_name="Google",
        )
        retriever.col_resume.query.assert_called_once()

    def test_returns_list(self, retriever):
        retriever.col_resume.query.return_value = self._chroma_query_response(["doc_1", "doc_2"])
        result = retriever.retrieve_top_results(
            text=["Python", "ML pipelines", "LangChain"],
            section_name="experience",
            item_name=None,
        )
        assert isinstance(result, list)


# ── _apply_rerank ─────────────────────────────────────────────────────────────

class TestApplyRerank:
    def test_empty_raw_returns_empty_list(self, retriever):
        assert retriever._apply_rerank({}, n_responsibilities=0, top_k=5) == []

    def test_empty_ids_returns_empty_list(self, retriever):
        raw = {"ids": [[], []], "distances": [[], []], "metadatas": [[], []]}
        assert retriever._apply_rerank(raw, n_responsibilities=1, top_k=5) == []

    def test_delegates_to_rerank(self, retriever):
        raw = {
            "ids":       [["doc_A"]],
            "distances": [[0.2]],
            "metadatas": [[{"section": "skills"}]],
        }
        result = retriever._apply_rerank(raw, n_responsibilities=0, top_k=5)
        assert len(result) == 1
        assert result[0][1] == "doc_A"


# ── retrieve ──────────────────────────────────────────────────────────────────

class TestRetrieve:
    def _mock_df(self, item_names):
        """Return a DataFrame-like mock whose ['item_name'].unique().tolist() yields item_names."""
        mock_df = MagicMock()
        mock_df.__getitem__.return_value.unique.return_value.tolist.return_value = item_names
        return mock_df

    def test_returns_entry_per_flat_section(self, retriever):
        from app.core.retriever import FLAT_SECTIONS, ATOMIC_SECTIONS
        retriever.retrieve_top_results = MagicMock(return_value=[])
        retriever.db.fetch_table_where = MagicMock(return_value=self._mock_df([]))

        results = retriever.retrieve(text=["Python"])
        flat_names = [sec for sec, _, _ in results if sec in FLAT_SECTIONS]
        assert len(flat_names) == len(FLAT_SECTIONS)

    def test_returns_entry_per_atomic_item(self, retriever):
        from app.core.retriever import ATOMIC_SECTIONS
        retriever.retrieve_top_results = MagicMock(return_value=[])
        # Each atomic section has 2 items
        retriever.db.fetch_table_where = MagicMock(return_value=self._mock_df(["Google", "Meta"]))

        results = retriever.retrieve(text=["Python"])
        atomic_entries = [(sec, item) for sec, item, _ in results if sec in ATOMIC_SECTIONS]
        # 2 atomic sections × 2 items each = 4
        assert len(atomic_entries) == len(ATOMIC_SECTIONS) * 2

    def test_result_tuples_have_three_elements(self, retriever):
        retriever.retrieve_top_results = MagicMock(return_value=[])
        retriever.db.fetch_table_where = MagicMock(return_value=self._mock_df([]))
        results = retriever.retrieve(text=["Python"])
        assert all(len(t) == 3 for t in results)


# ── run ───────────────────────────────────────────────────────────────────────

class TestRunMethod:
    def test_returns_empty_when_job_not_found(self, retriever):
        retriever.db.fetch_one.return_value = None
        result = retriever.run(job_id=99, jd_obj=None)
        assert result == []

    def test_uses_provided_jd_obj_directly(self, retriever, sample_jd):
        retriever.retrieve = MagicMock(return_value=[("skills", None, [])])
        result = retriever.run(job_id=1, jd_obj=sample_jd)
        retriever.retrieve.assert_called_once()
        assert result == [("skills", None, [])]

    def test_fetches_jd_when_not_provided(self, retriever, sample_jd):
        import json
        retriever.db.fetch_one.return_value = {"jd_parsed": json.dumps(sample_jd.to_dict())}
        retriever.retrieve = MagicMock(return_value=[])
        retriever.run(job_id=1, jd_obj=None)
        retriever.retrieve.assert_called_once()


# ── rank_and_filter ───────────────────────────────────────────────────────────

class TestRankAndFilter:
    def _flat_run_result(self, section_name, score_val=0.9):
        meta = {"resume_id": 1, "section_type": section_name}
        return [(section_name, None, [(score_val, "doc_A", meta)])]

    def _atomic_run_result(self):
        meta_g = {"resume_id": 1, "section_type": "experience", "sql_id": "10"}
        meta_m = {"resume_id": 1, "section_type": "experience", "sql_id": "20"}
        return [
            ("experience", "Google", [(0.9, "d1", meta_g), (0.8, "d2", meta_g)]),
            ("experience", "Meta",   [(0.6, "d3", meta_m)]),
        ]

    def test_flat_section_passes_through(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 1, "content_text": "Python"}
        results = retriever.rank_and_filter(self._flat_run_result("skills"))
        assert len(results) == 1
        assert results[0]["section_name"] == "skills"
        assert results[0]["item_name"] is None

    def test_atomic_items_sorted_by_aggregated_score(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 1, "content_text": "some text"}
        results = retriever.rank_and_filter(self._atomic_run_result())
        atom = [r for r in results if r["section_name"] == "experience"]
        # Google (two chunks with high scores) should rank above Meta (one lower score)
        assert atom[0]["item_name"] == "Google"
        assert atom[1]["item_name"] == "Meta"

    def test_flat_scores_list_matches_chunk_count(self, retriever):
        retriever.db.fetch_one.return_value = {}
        results = retriever.rank_and_filter(self._flat_run_result("skills", score_val=0.85))
        assert results[0]["scores"] == [0.85]

    def test_empty_run_results_returns_empty(self, retriever):
        assert retriever.rank_and_filter([]) == []


# ── run_retriever ─────────────────────────────────────────────────────────────

class TestRunRetriever:
    def test_calls_run_and_rank_and_filter(self, retriever, sample_jd):
        retriever.run = MagicMock(return_value=[("skills", None, [(0.9, "d", {})])])
        retriever.rank_and_filter = MagicMock(return_value=[{"section_name": "skills"}])

        result = retriever.run_retriever(job_id=1, jd_obj=sample_jd)

        args, kwargs = retriever.run.call_args
        assert args[:2] == (1, sample_jd)
        retriever.rank_and_filter.assert_called_once()
        assert result == [{"section_name": "skills"}]
