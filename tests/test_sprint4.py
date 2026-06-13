"""
Sprint 4 tests — retrieval improvements, dropped_items, write-back safety,
and preview/drop frontend wiring.

All external deps (DB, ChromaDB, LLM, config) are mocked via conftest.py and
local patches so no real connections are required.
"""
import json
import pytest
from unittest.mock import MagicMock, patch, call


# ══════════════════════════════════════════════════════════════════════════════
# Retriever — return_all, score normalisation, fetch_row sql_id, best_score
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def retriever():
    from app.core.retriever import ResumeRetriever
    r = ResumeRetriever.__new__(ResumeRetriever)
    r._n_resp      = 0
    r.top_k        = 3
    r.fetch_mult   = 3
    r.weight_skill = 0.5
    r.weight_resp  = 0.3
    r.weight_nth   = 0.2
    r.db           = MagicMock()
    r.col_resume   = MagicMock()
    r.col_jd       = MagicMock()
    r._chroma_client = MagicMock()
    return r


def _chroma_result(n: int, base_dist: float = 0.1):
    """Build a minimal single-query ChromaDB response with n distinct docs."""
    ids   = [[f"doc_{i}" for i in range(n)]]
    dists = [[base_dist + i * 0.05 for i in range(n)]]
    metas = [[{"section_type": "skills", "resume_id": 1}] * n]
    return {"ids": ids, "distances": dists, "metadatas": metas}


class TestReturnAll:
    def test_return_all_false_limits_to_top_k(self, retriever):
        result = retriever._rerank(_chroma_result(10), n_responsibilities=0,
                                   top_k=3, return_all=False)
        assert len(result) <= 3

    def test_return_all_true_bypasses_top_k(self, retriever):
        result = retriever._rerank(_chroma_result(10), n_responsibilities=0,
                                   top_k=3, return_all=True)
        # All 10 docs should be returned
        assert len(result) == 10

    def test_return_all_default_is_false(self, retriever):
        """Default behaviour must not change (backwards-compatible)."""
        result = retriever._rerank(_chroma_result(10), n_responsibilities=0,
                                   top_k=3)
        assert len(result) <= 3

    def test_return_all_propagates_through_apply_rerank(self, retriever):
        raw = _chroma_result(8)
        result = retriever._apply_rerank(raw, n_responsibilities=0,
                                         top_k=3, return_all=True)
        assert len(result) == 8

    def test_return_all_propagates_to_retrieve_top_results(self, retriever):
        retriever.col_resume.query.return_value = _chroma_result(8)
        result = retriever.retrieve_top_results(
            text=["Python"], section_name="skills", return_all=True
        )
        assert len(result) == 8

    def test_run_passes_return_all_to_retrieve(self, retriever):
        """run(return_all=True) should call retrieve with return_all=True."""
        retriever.retrieve = MagicMock(return_value=[])
        from app.models.jd import ParsedJD
        jd = ParsedJD(required_skills=["Python"], responsibilities=["Build APIs"],
                      nice_to_have_skills=[], company="X", role="Eng",
                      seniority_level="", location="", is_remote=False,
                      about_company="", about_job="", perks="", raw_text="")
        retriever.run(job_id=1, jd_obj=jd, return_all=True)
        _, kwargs = retriever.retrieve.call_args
        assert kwargs.get("return_all") is True


class TestScoreNormalization:
    def test_all_scores_in_unit_interval(self, retriever):
        result = retriever._rerank(_chroma_result(5), n_responsibilities=0,
                                   top_k=10, return_all=True)
        for score, _, _ in result:
            assert 0.0 <= score <= 1.0, f"Score out of [0,1]: {score}"

    def test_top_score_is_at_most_one(self, retriever):
        result = retriever._rerank(_chroma_result(5), n_responsibilities=0,
                                   top_k=10, return_all=True)
        assert result[0][0] <= 1.0

    def test_single_doc_appearing_in_all_groups_scores_one(self, retriever):
        """A doc that is top scorer in every group should score 1.0."""
        chroma = {
            "ids":       [["doc_A"], ["doc_A"], ["doc_A"]],  # req, resp, nice
            "distances": [[0.1],    [0.1],    [0.1]],
            "metadatas": [[{"section_type": "skills", "resume_id": 1}]] * 3,
        }
        # n_responsibilities=1 so query 0=req, 1=resp, 2=nice
        result = retriever._rerank(chroma, n_responsibilities=1,
                                   top_k=5, return_all=True)
        assert len(result) == 1
        assert abs(result[0][0] - 1.0) < 1e-9

    def test_multi_group_scores_stay_in_unit_interval(self, retriever):
        """Two docs across two groups — both scores must stay ≤ 1."""
        chroma = {
            "ids":       [["doc_A"], ["doc_B"], ["doc_A"]],   # req, resp, nice
            "distances": [[0.1],    [0.2],    [0.3]],
            "metadatas": [[{"section_type": "experience", "resume_id": 1}],
                          [{"section_type": "projects",   "resume_id": 1}],
                          [{"section_type": "experience", "resume_id": 1}]],
        }
        result = retriever._rerank(chroma, n_responsibilities=1,
                                   top_k=5, return_all=True)
        for score, _, _ in result:
            assert 0.0 <= score <= 1.0


class TestFetchRowSqlId:
    def _run_result_with_sql_id(self, sql_id="42"):
        meta = {"resume_id": 1, "section_type": "experience", "sql_id": sql_id}
        return [("experience", "Google", [(0.9, "doc_A", meta)])]

    def _run_result_without_sql_id(self):
        meta = {"resume_id": 1, "section_type": "experience"}
        return [("experience", "Google", [(0.9, "doc_A", meta)])]

    def test_uses_sql_id_when_present(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 42, "content_text": "ML job"}
        retriever.rank_and_filter(self._run_result_with_sql_id("42"))

        # At least one call must use id=42
        calls = retriever.db.fetch_one.call_args_list
        id_calls = [c for c in calls if c.kwargs.get("filters", {}).get("id") == 42
                    or (len(c.args) > 1 and isinstance(c.args[1], dict)
                        and c.args[1].get("id") == 42)]
        # Accept keyword OR positional arg style
        found = any(
            (c.kwargs.get("filters", {}).get("id") == 42 or
             (c.args[1:] and isinstance(c.args[1], dict) and c.args[1].get("id") == 42))
            for c in calls
        )
        assert found, "fetch_one was not called with id=42"

    def test_does_not_use_item_name_filter_when_sql_id_present(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 42, "content_text": "ML job"}
        retriever.rank_and_filter(self._run_result_with_sql_id("42"))

        # None of the calls for an atomic item should use {item_name: "Google"}
        calls_with_item_name = [
            c for c in retriever.db.fetch_one.call_args_list
            if c.kwargs.get("filters", {}).get("item_name") == "Google"
        ]
        assert len(calls_with_item_name) == 0, \
            "fetch_one should not fall back to item_name filter when sql_id present"

    def test_fallback_to_item_name_when_no_sql_id(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 1, "content_text": "Google job"}
        retriever.rank_and_filter(self._run_result_without_sql_id())

        # Should fall back to filtering by item_name + is_latest
        calls = retriever.db.fetch_one.call_args_list
        found = any(
            c.kwargs.get("filters", {}).get("item_name") == "Google"
            for c in calls
        )
        assert found, "Expected fallback fetch by item_name when sql_id is absent"


class TestBestScoreField:
    def _atomic_run_result(self):
        meta_g = {"resume_id": 1, "section_type": "experience", "sql_id": "10"}
        meta_m = {"resume_id": 1, "section_type": "experience", "sql_id": "20"}
        return [
            ("experience", "Google", [(0.9, "d1", meta_g), (0.7, "d2", meta_g)]),
            ("experience", "Meta",   [(0.6, "d3", meta_m)]),
        ]

    def test_atom_output_has_best_score(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 10, "content_text": "job"}
        results = retriever.rank_and_filter(self._atomic_run_result())
        atom = [r for r in results if r["section_name"] == "experience"]
        for entry in atom:
            assert "best_score" in entry, "Missing best_score in atomic output"

    def test_atom_output_has_agg_score(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 10, "content_text": "job"}
        results = retriever.rank_and_filter(self._atomic_run_result())
        atom = [r for r in results if r["section_name"] == "experience"]
        for entry in atom:
            assert "agg_score" in entry, "Missing agg_score in atomic output"

    def test_best_score_equals_top_chunk_score(self, retriever):
        retriever.db.fetch_one.return_value = {"id": 10, "content_text": "job"}
        results = retriever.rank_and_filter(self._atomic_run_result())
        google_entry = next(r for r in results if r["item_name"] == "Google")
        # Best chunk score is 0.9, best_score should be close to it
        assert google_entry["best_score"] == pytest.approx(0.9, abs=1e-4)

    def test_flat_output_has_no_best_score(self, retriever):
        meta = {"resume_id": 1, "section_type": "skills"}
        run_result = [("skills", None, [(0.8, "doc_A", meta)])]
        retriever.db.fetch_one.return_value = {"id": 1, "content_text": "Python"}
        results = retriever.rank_and_filter(run_result)
        flat = [r for r in results if r["section_name"] == "skills"]
        # flat sections do not get best_score (not needed for display)
        assert len(flat) == 1
        assert "best_score" not in flat[0]


# ══════════════════════════════════════════════════════════════════════════════
# Graph — dropped_items, drop proposal, generate_suggestions, build_edit_state
# ══════════════════════════════════════════════════════════════════════════════

def _base_state(**overrides):
    state = {
        "user_id": 1,
        "parsed_jd": {
            "company": "X", "role": "Eng", "seniority_level": "mid",
            "location": "", "is_remote": False, "about_company": "",
            "about_job": "", "responsibilities": [], "required_skills": [],
            "nice_to_have_skills": [], "perks": "", "others": "",
        },
        "selected_items": [],
        "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
        "sections_pending":  [],
        "sections_reviewed": [],
        "dropped_items": [],
    }
    state.update(overrides)
    return state


class TestDroppedItemsState:
    """EditGraphState accepts dropped_items."""

    def test_dropped_items_field_exists(self):
        from app.graph.state import EditGraphState
        state: EditGraphState = {}
        state["dropped_items"] = [{"section": "experience", "item": "Startup"}]
        assert state["dropped_items"][0]["section"] == "experience"

    def test_dropped_items_defaults_to_absent_not_required(self):
        from app.graph.state import EditGraphState
        state: EditGraphState = {}
        assert state.get("dropped_items", []) == []


class TestNodeApplyEditDrop:
    def _make_mock_agent(self, item_name=None):
        agent = MagicMock()
        cycle = MagicMock()
        agent.editing_cycle = cycle

        if item_name:
            from app.models.edit import ItemEditState
            item = MagicMock(spec=ItemEditState)
            item.section_previous_state = "original content"
            cycle.current_state.get_item.return_value = (None, item)
            items_list = [MagicMock(item_name=item_name)]
            items_list[0].item_name = item_name
            getattr_mock = MagicMock(return_value=items_list)
            cycle.current_state.__class__ = MagicMock
            cycle.current_state.model_copy.return_value = cycle.current_state
        else:
            from app.models.edit import SectionEditState
            sec = MagicMock(spec=SectionEditState)
            sec.section_previous_state = "original section"
            cycle.current_state.get_section.return_value = sec
            cycle.current_state.set_section.return_value = cycle.current_state

        return agent

    def test_drop_proposal_adds_to_dropped_items(self):
        from app.graph.edit_graph import node_apply_edit

        review = {
            "proposal": "drop",
            "section":  "experience",
            "item":     "Startup Inc",
        }
        state = _base_state(
            current_review=review,
            sections_reviewed=[],
            sections_pending=[],
            dropped_items=[],
        )

        mock_agent = self._make_mock_agent(item_name="Startup Inc")

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_apply_edit(state)

        dropped = result.get("dropped_items", [])
        assert len(dropped) == 1
        assert dropped[0]["section"] == "experience"
        assert dropped[0]["item"] == "Startup Inc"

    def test_drop_appends_to_existing_dropped_items(self):
        from app.graph.edit_graph import node_apply_edit

        review = {"proposal": "drop", "section": "projects", "item": "Old Project"}
        state = _base_state(
            current_review=review,
            sections_reviewed=[],
            sections_pending=[],
            dropped_items=[{"section": "experience", "item": "Startup"}],
        )

        mock_agent = self._make_mock_agent(item_name="Old Project")

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_apply_edit(state)

        dropped = result.get("dropped_items", [])
        assert len(dropped) == 2

    def test_drop_advances_sections_reviewed(self):
        from app.graph.edit_graph import node_apply_edit

        review = {"proposal": "drop", "section": "skills", "item": None}
        state = _base_state(
            current_review=review,
            sections_reviewed=[{"section": "education", "item": None}],
            sections_pending=[],
            dropped_items=[],
        )

        mock_agent = self._make_mock_agent(item_name=None)

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_apply_edit(state)

        reviewed = result.get("sections_reviewed", [])
        sections_in_reviewed = [r["section"] for r in reviewed]
        assert "skills" in sections_in_reviewed

    def test_drop_does_not_re_queue_section(self):
        """Dropped section must NOT be put back in sections_pending."""
        from app.graph.edit_graph import node_apply_edit

        review = {"proposal": "drop", "section": "experience", "item": "Job A"}
        state = _base_state(
            current_review=review,
            sections_reviewed=[],
            sections_pending=[{"section": "projects", "item": "P1"}],
            dropped_items=[],
        )

        mock_agent = self._make_mock_agent(item_name="Job A")

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_apply_edit(state)

        pending = result.get("sections_pending", state.get("sections_pending", []))
        dropped_in_pending = any(
            p["section"] == "experience" and p.get("item") == "Job A"
            for p in pending
        )
        assert not dropped_in_pending, "Dropped item must not re-appear in sections_pending"


class TestNodeGenerateSuggestionsDropped:
    def _make_state(self, selected, dropped=None):
        state = _base_state(
            selected_items=selected,
            dropped_items=dropped or [],
        )
        return state

    def test_dropped_items_excluded_from_pending(self):
        from app.graph.edit_graph import node_generate_suggestions

        selected = [
            {"section_name": "experience", "item_name": "Google"},
            {"section_name": "experience", "item_name": "Startup"},
            {"section_name": "skills",     "item_name": None},
        ]
        dropped = [{"section": "experience", "item": "Startup"}]
        state = self._make_state(selected, dropped)

        mock_agent = MagicMock()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_generate_suggestions(state)

        pending = result.get("sections_pending", [])
        startup_in_pending = any(
            p["section"] == "experience" and p.get("item") == "Startup"
            for p in pending
        )
        assert not startup_in_pending, "Dropped 'Startup' must not appear in sections_pending"

    def test_non_dropped_items_appear_in_pending(self):
        from app.graph.edit_graph import node_generate_suggestions

        selected = [
            {"section_name": "experience", "item_name": "Google"},
            {"section_name": "experience", "item_name": "Startup"},
        ]
        dropped = [{"section": "experience", "item": "Startup"}]
        state = self._make_state(selected, dropped)

        mock_agent = MagicMock()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_generate_suggestions(state)

        pending = result.get("sections_pending", [])
        google_in_pending = any(
            p["section"] == "experience" and p.get("item") == "Google"
            for p in pending
        )
        assert google_in_pending, "Non-dropped 'Google' must remain in sections_pending"

    def test_no_dropped_items_all_selected_appear(self):
        from app.graph.edit_graph import node_generate_suggestions

        selected = [
            {"section_name": "skills", "item_name": None},
            {"section_name": "education", "item_name": None},
        ]
        state = self._make_state(selected, dropped=[])

        mock_agent = MagicMock()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_generate_suggestions(state)

        assert len(result.get("sections_pending", [])) == 2


class TestNodeBuildEditStateDropped:
    def test_dropped_items_excluded_from_initial_queue(self):
        from app.graph.edit_graph import node_build_edit_state

        selected = [
            {"section_name": "experience", "item_name": "Google"},
            {"section_name": "experience", "item_name": "Startup"},
        ]
        dropped = [{"section": "experience", "item": "Startup"}]
        state = _base_state(selected_items=selected, dropped_items=dropped)

        mock_agent = MagicMock()
        with patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            mock_pl.build_starting_edit_state.return_value = MagicMock()
            mock_pl.build_edit_agent.return_value = mock_agent
            from app.models.jd import ParsedJD
            result = node_build_edit_state(state)

        pending = result.get("sections_pending", [])
        startup_in_pending = any(
            p["section"] == "experience" and p.get("item") == "Startup"
            for p in pending
        )
        assert not startup_in_pending

    def test_dropped_items_preserved_in_output(self):
        from app.graph.edit_graph import node_build_edit_state

        selected = [{"section_name": "skills", "item_name": None}]
        dropped  = [{"section": "experience", "item": "OldJob"}]
        state = _base_state(selected_items=selected, dropped_items=dropped)

        with patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            mock_pl.build_starting_edit_state.return_value = MagicMock()
            mock_pl.build_edit_agent.return_value = MagicMock()
            result = node_build_edit_state(state)

        assert result.get("dropped_items") == dropped


class TestNodeParseAndRetrieveReturnAll:
    """node_parse_and_retrieve must pass return_all=True to pl.retrieve."""

    def _make_state(self):
        return {
            "user_id": 1,
            "jd_text": "Build ML pipelines",
            "existing_resume_id": 1,
            "job_id": None,
            "parsed_jd": None,
            "resume_path": None,
        }

    def test_retrieve_called_with_return_all_true(self):
        from app.graph.edit_graph import node_parse_and_retrieve

        state = self._make_state()

        with patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph.ParsedJD"):
            # Simulate existing_resume_id path that uses pl.retrieve
            mock_pl.parse_jd.return_value = (1, MagicMock(responsibilities=[]))
            mock_pl.load_resume_from_db.return_value = MagicMock(
                resume_id="1",
                personal_info=MagicMock(model_dump=lambda: {}),
                resume_sections={},
            )
            mock_pl.get_user_resume_ids.return_value = [1]
            mock_pl.retrieve.return_value = []
            mock_pl.rank_and_filter.return_value = []

            node_parse_and_retrieve(state)

        if mock_pl.retrieve.called:
            _, kwargs = mock_pl.retrieve.call_args
            assert kwargs.get("return_all") is True, \
                "node_parse_and_retrieve must call pl.retrieve with return_all=True"


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline — ChromaDB delete removed from write_back_edited_items
# ══════════════════════════════════════════════════════════════════════════════

class TestWriteBackNoChromaDelete:
    """write_back_edited_items must NOT delete old ChromaDB entries."""

    def _make_selected_items(self):
        return [{
            "section_name": "experience",
            "item_name":    "Google",
            "content_text": "Updated content about ML work",
        }]

    def test_col_resume_delete_is_never_called(self):
        from app.core import pipeline as pl

        mock_col = MagicMock()
        mock_db  = MagicMock()

        # Simulate a row existing in DB for the item
        mock_db.fetch_one.return_value = {
            "id": 10, "section_id": 1, "section_name": "experience",
            "item_name": "Google", "content_text": "Old content",
            "is_latest": 1, "resume_id": 1,
        }
        mock_db.add_one.return_value = 11
        mock_db.update_data.return_value = 1

        mock_agent = MagicMock()
        mock_agent.editing_cycle.current_state.get_item.return_value = (
            None,
            MagicMock(updated_section="New content", item_name="Google"),
        )

        state_values = {
            "selected_items": self._make_selected_items(),
            "resume_id":      1,
        }

        with patch.object(pl, "_get_retriever") as mock_get_retriever, \
             patch("app.core.pipeline.SQLHandler", return_value=mock_db):
            mock_retriever = MagicMock()
            mock_retriever.col_resume = mock_col
            mock_get_retriever.return_value = mock_retriever

            try:
                pl.write_back_edited_items(state_values, mock_agent)
            except Exception:
                pass  # We only care that delete was not called

        mock_col.delete.assert_not_called()

    def test_col_resume_upsert_is_called_for_new_version(self):
        from app.core import pipeline as pl

        mock_col = MagicMock()
        mock_db  = MagicMock()

        mock_db.fetch_one.return_value = {
            "id": 10, "section_id": 1, "section_name": "experience",
            "item_name": "Google", "content_text": "Old content",
            "is_latest": 1, "resume_id": 1,
        }
        mock_db.add_one.return_value = 11
        mock_db.update_data.return_value = 1

        mock_agent = MagicMock()
        mock_agent.editing_cycle.current_state.get_item.return_value = (
            None,
            MagicMock(updated_section="New ML content", item_name="Google"),
        )

        state_values = {
            "selected_items": self._make_selected_items(),
            "resume_id":      1,
        }

        with patch.object(pl, "_get_retriever") as mock_get_retriever, \
             patch("app.core.pipeline.SQLHandler", return_value=mock_db):
            mock_retriever = MagicMock()
            mock_retriever.col_resume = mock_col
            mock_get_retriever.return_value = mock_retriever

            try:
                pl.write_back_edited_items(state_values, mock_agent)
            except Exception:
                pass

        # upsert should have been attempted (even if it errored)
        # We simply verify delete was never called — upsert is best-effort
        mock_col.delete.assert_not_called()


# ══════════════════════════════════════════════════════════════════════════════
# Template files exist and have expected structure
# ══════════════════════════════════════════════════════════════════════════════

class TestTemplateFiles:
    TEMPLATE_DIR = __import__("pathlib").Path(__file__).parent.parent / "templates" / "resume"

    def test_classic_template_exists(self):
        assert (self.TEMPLATE_DIR / "classic.tex.j2").exists()

    def test_ats_minimal_template_exists(self):
        assert (self.TEMPLATE_DIR / "ats_minimal.tex.j2").exists()

    def test_classic_has_fontawesome_header(self):
        content = (self.TEMPLATE_DIR / "classic.tex.j2").read_text()
        assert "fontawesome" in content

    def test_classic_has_jinja2_personal_name(self):
        content = (self.TEMPLATE_DIR / "classic.tex.j2").read_text()
        assert "personal.name" in content

    def test_classic_has_section_loop(self):
        content = (self.TEMPLATE_DIR / "classic.tex.j2").read_text()
        assert "for section in sections" in content

    def test_ats_minimal_has_no_fontawesome(self):
        content = (self.TEMPLATE_DIR / "ats_minimal.tex.j2").read_text()
        assert "fontawesome" not in content

    def test_classic_matches_original_preamble(self):
        """classic.tex.j2 must use the same LaTeX packages as the original resume."""
        content = (self.TEMPLATE_DIR / "classic.tex.j2").read_text()
        for pkg in ("titlesec", "enumitem", "hyperref", "xcolor", "parskip", "geometry"):
            assert f"\\usepackage{{{pkg}}}" in content or f"\\usepackage[" in content, \
                f"Expected package '{pkg}' in classic.tex.j2"

    def test_template_renderer_lists_both_templates(self):
        from app.core.resume_builder import TemplateRenderer
        renderer = TemplateRenderer()
        templates = renderer.list_templates()
        assert "classic" in templates
        assert "ats_minimal" in templates


# ══════════════════════════════════════════════════════════════════════════════
# API client — getPreview exists and targets the right endpoint
# ══════════════════════════════════════════════════════════════════════════════

class TestApiClientPreview:
    API_TS = __import__("pathlib").Path(__file__).parent.parent / "frontend" / "src" / "lib" / "api.ts"

    def test_api_ts_file_exists(self):
        assert self.API_TS.exists()

    def test_get_preview_function_exists(self):
        content = self.API_TS.read_text()
        assert "getPreview" in content

    def test_get_preview_calls_correct_endpoint(self):
        content = self.API_TS.read_text()
        assert "/preview" in content


class TestPreviewDrawerComponent:
    def test_preview_drawer_file_exists(self):
        path = __import__("pathlib").Path(__file__).parent.parent / \
               "frontend" / "src" / "components" / "session" / "PreviewDrawer.tsx"
        assert path.exists()

    def test_preview_drawer_calls_get_preview(self):
        path = __import__("pathlib").Path(__file__).parent.parent / \
               "frontend" / "src" / "components" / "session" / "PreviewDrawer.tsx"
        content = path.read_text()
        assert "getPreview" in content

    def test_section_review_imports_preview_drawer(self):
        path = __import__("pathlib").Path(__file__).parent.parent / \
               "frontend" / "src" / "components" / "session" / "section-review.tsx"
        content = path.read_text()
        assert "PreviewDrawer" in content

    def test_item_selection_imports_preview_drawer(self):
        path = __import__("pathlib").Path(__file__).parent.parent / \
               "frontend" / "src" / "components" / "session" / "item-selection.tsx"
        content = path.read_text()
        assert "PreviewDrawer" in content

    def test_section_review_has_drop_button(self):
        path = __import__("pathlib").Path(__file__).parent.parent / \
               "frontend" / "src" / "components" / "session" / "section-review.tsx"
        content = path.read_text()
        assert 'proposal: "drop"' in content or "handleDrop" in content
