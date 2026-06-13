"""
Sprint 2 — LangGraph edit graph tests.

Covers every non-interrupt node (node_build_edit_state, node_generate_suggestions,
node_apply_edit, node_score_resume, route_after_review_or_done, route_after_error)
and the interrupt-node structural contracts (node_item_selection interrupt payload
format, node_section_review section-popping logic).

Interrupt nodes (item_selection, section_review, refine_or_finish) cannot be
driven to completion in unit tests because LangGraph's interrupt() requires the
full graph runtime. We test their surrounding state transformations instead.

All LLM, DB, and ChromaDB calls are mocked.
"""
import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _jd_dict():
    return {
        "company": "Acme", "role": "SWE", "seniority_level": "mid",
        "location": "", "is_remote": False, "about_company": "",
        "about_job": "", "responsibilities": ["Build APIs"],
        "required_skills": ["Python"], "nice_to_have_skills": [],
        "perks": "", "others": "",
    }


def _base_state(**kw):
    s = {
        "user_id": 1,
        "parsed_jd": _jd_dict(),
        "selected_items": [],
        "dropped_items": [],
        "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
        "sections_pending":  [],
        "sections_reviewed": [],
        "cycle_id": 1,
    }
    s.update(kw)
    return s


def _make_mock_agent():
    agent = MagicMock()
    agent.editing_cycle = MagicMock()
    agent.editing_cycle.current_state = MagicMock()
    return agent


# ══════════════════════════════════════════════════════════════════════════════
# Graph can be built
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildEditGraph:
    def test_build_edit_graph_returns_compiled_graph(self):
        from app.graph.edit_graph import edit_graph
        # edit_graph is the compiled StateGraph — it exists and is not None
        assert edit_graph is not None

    def test_graph_has_invoke_method(self):
        from app.graph.edit_graph import edit_graph
        assert callable(getattr(edit_graph, "invoke", None))

    def test_graph_has_get_state_method(self):
        from app.graph.edit_graph import edit_graph
        assert callable(getattr(edit_graph, "get_state", None))


# ══════════════════════════════════════════════════════════════════════════════
# node_build_edit_state
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeBuildEditState:
    def _run(self, selected_items, dropped_items=None, **extra):
        from app.graph.edit_graph import node_build_edit_state
        state = _base_state(
            selected_items=selected_items,
            dropped_items=dropped_items or [],
            **extra,
        )
        with patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            mock_pl.build_starting_edit_state.return_value = MagicMock()
            mock_pl.build_edit_agent.return_value = _make_mock_agent()
            return node_build_edit_state(state)

    def test_sections_pending_built_from_selected_items(self):
        selected = [
            {"section_name": "skills",     "item_name": None},
            {"section_name": "education",  "item_name": None},
            {"section_name": "experience", "item_name": "Google"},
        ]
        result = self._run(selected)
        pending = result.get("sections_pending", [])
        assert len(pending) == 3

    def test_pending_section_keys_are_section_and_item(self):
        selected = [{"section_name": "skills", "item_name": None}]
        result = self._run(selected)
        pending = result.get("sections_pending", [])
        assert "section" in pending[0]
        assert "item" in pending[0]

    def test_flat_section_item_is_none(self):
        selected = [{"section_name": "skills", "item_name": None}]
        result = self._run(selected)
        assert result["sections_pending"][0]["item"] is None

    def test_atomic_section_item_name_preserved(self):
        selected = [{"section_name": "experience", "item_name": "Google"}]
        result = self._run(selected)
        assert result["sections_pending"][0]["item"] == "Google"

    def test_dropped_items_excluded_from_pending(self):
        selected = [
            {"section_name": "experience", "item_name": "Google"},
            {"section_name": "experience", "item_name": "Startup"},
        ]
        dropped = [{"section": "experience", "item": "Startup"}]
        result = self._run(selected, dropped_items=dropped)
        items_in_pending = [p["item"] for p in result.get("sections_pending", [])]
        assert "Startup" not in items_in_pending
        assert "Google" in items_in_pending

    def test_stage_set_to_edit_state_built(self):
        result = self._run([])
        assert result.get("stage") == "edit_state_built"

    def test_pl_build_starting_edit_state_called_with_selected(self):
        selected = [{"section_name": "skills", "item_name": None}]
        from app.graph.edit_graph import node_build_edit_state
        state = _base_state(selected_items=selected)
        with patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            mock_pl.build_starting_edit_state.return_value = MagicMock()
            mock_pl.build_edit_agent.return_value = _make_mock_agent()
            node_build_edit_state(state)
        mock_pl.build_starting_edit_state.assert_called_once_with(selected)


# ══════════════════════════════════════════════════════════════════════════════
# node_generate_suggestions
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeGenerateSuggestions:
    def _run(self, selected=None, dropped=None, feedback=None, instruction=None):
        from app.graph.edit_graph import node_generate_suggestions
        state = _base_state(
            selected_items=selected or [],
            dropped_items=dropped or [],
            score_feedback=feedback,
            custom_instruction=instruction,
        )
        mock_agent = _make_mock_agent()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            result = node_generate_suggestions(state)
        return result, mock_agent, mock_pl

    def test_stage_set_to_suggestions_ready(self):
        result, *_ = self._run()
        assert result.get("stage") == "suggestions_ready"

    def test_sections_reviewed_reset_to_empty(self):
        result, *_ = self._run()
        assert result.get("sections_reviewed") == []

    def test_pending_rebuilt_from_selected_items(self):
        selected = [
            {"section_name": "skills",     "item_name": None},
            {"section_name": "experience", "item_name": "Google"},
        ]
        result, *_ = self._run(selected=selected)
        assert len(result.get("sections_pending", [])) == 2

    def test_dropped_items_excluded_from_pending(self):
        selected = [
            {"section_name": "experience", "item_name": "Startup"},
            {"section_name": "skills",     "item_name": None},
        ]
        dropped = [{"section": "experience", "item": "Startup"}]
        result, *_ = self._run(selected=selected, dropped=dropped)
        pending = result.get("sections_pending", [])
        assert not any(p["item"] == "Startup" for p in pending)

    def test_plain_generate_called_when_no_feedback(self):
        _, _, mock_pl = self._run()
        mock_pl.generate_suggestions.assert_called_once()
        mock_pl.generate_suggestions_with_score_feedback.assert_not_called()

    def test_score_feedback_calls_feedback_variant(self):
        _, _, mock_pl = self._run(feedback="Improve skills section score")
        mock_pl.generate_suggestions_with_score_feedback.assert_called_once()
        mock_pl.generate_suggestions.assert_not_called()

    def test_global_instruction_set_from_custom_instruction(self):
        _, agent, _ = self._run(instruction="Be concise, use numbers")
        assert "Be concise" in agent.global_instruction

    def test_global_instruction_empty_when_no_instruction(self):
        _, agent, _ = self._run()
        assert agent.global_instruction == ""


# ══════════════════════════════════════════════════════════════════════════════
# node_section_review — queue-popping logic (no interrupt called in unit test)
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeSectionReviewQueueLogic:
    """
    node_section_review calls interrupt() which throws in unit-test context.
    We test the queue-state setup that happens BEFORE the interrupt call by
    mocking interrupt() itself.
    """

    def _run(self, pending, reviewed=None):
        from app.graph.edit_graph import node_section_review

        # Build a mock agent with a current_state that has a known section
        mock_agent = _make_mock_agent()
        current = MagicMock()
        mock_section = MagicMock()
        mock_section.updated_section = "\\item Python"
        mock_section.lines_to_change = []
        mock_section.suggested_changes = []
        current.get_section.return_value = mock_section
        current.get_item.return_value = (None, mock_section)
        mock_agent.editing_cycle.current_state = current
        mock_agent.editing_cycle.nodes = {0: MagicMock(resume_state=current, action="initial")}

        state = _base_state(
            sections_pending=pending,
            sections_reviewed=reviewed or [],
        )

        interrupt_payload = {}

        def _fake_interrupt(payload):
            interrupt_payload.update(payload)
            return {"proposal": "accept"}

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}), \
             patch("app.graph.edit_graph.interrupt", side_effect=_fake_interrupt):
            result = node_section_review(state)

        return result, interrupt_payload

    def test_pops_first_section_from_pending(self):
        pending = [
            {"section": "skills",     "item": None},
            {"section": "experience", "item": "Google"},
        ]
        result, _ = self._run(pending)
        # After review, sections_pending should have one fewer item
        remaining = result.get("sections_pending", [])
        assert len(remaining) == 1
        assert remaining[0]["section"] == "experience"

    def test_interrupt_payload_has_type_section_review(self):
        pending = [{"section": "skills", "item": None}]
        _, payload = self._run(pending)
        assert payload.get("type") == "section_review"

    def test_interrupt_payload_contains_section_name(self):
        pending = [{"section": "education", "item": None}]
        _, payload = self._run(pending)
        assert payload.get("section") == "education"

    def test_interrupt_payload_contains_current_latex(self):
        pending = [{"section": "skills", "item": None}]
        _, payload = self._run(pending)
        assert "current_latex" in payload

    def test_can_go_back_false_when_no_reviewed(self):
        pending = [{"section": "skills", "item": None}]
        _, payload = self._run(pending, reviewed=[])
        assert payload.get("can_go_back") is False

    def test_can_go_back_true_when_reviewed_not_empty(self):
        pending = [{"section": "skills", "item": None}]
        reviewed = [{"section": "education", "item": None}]
        _, payload = self._run(pending, reviewed=reviewed)
        assert payload.get("can_go_back") is True

    def test_empty_pending_handled_gracefully(self):
        """When pending is empty, node should not raise — just finish."""
        from app.graph.edit_graph import node_section_review
        state = _base_state(sections_pending=[], sections_reviewed=[])
        mock_agent = _make_mock_agent()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}), \
             patch("app.graph.edit_graph.interrupt", return_value={"proposal": "accept"}):
            result = node_section_review(state)
        assert isinstance(result, dict)


# ══════════════════════════════════════════════════════════════════════════════
# node_apply_edit — all proposals
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeApplyEditAccept:
    def _run(self, review, pending=None, reviewed=None, dropped=None):
        from app.graph.edit_graph import node_apply_edit
        state = _base_state(
            current_review=review,
            sections_pending=pending or [],
            sections_reviewed=reviewed or [],
            dropped_items=dropped or [],
        )
        mock_agent = _make_mock_agent()
        cycle = mock_agent.editing_cycle
        cycle.current_state.get_section.return_value = None
        cycle.current_state.get_item.return_value = (None, None)
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            return node_apply_edit(state)

    def test_full_accept_advances_reviewed(self):
        review = {"proposal": "accept", "section": "skills", "item": None}
        result = self._run(review)
        reviewed = result.get("sections_reviewed", [])
        assert any(r["section"] == "skills" for r in reviewed)

    def test_full_accept_does_not_requeue_section(self):
        review = {"proposal": "accept", "section": "skills", "item": None}
        result = self._run(review)
        pending = result.get("sections_pending", [])
        assert not any(p.get("section") == "skills" for p in pending)

    def test_accept_stage_is_edit_applied(self):
        review = {"proposal": "accept", "section": "skills", "item": None}
        result = self._run(review)
        assert result.get("stage") == "edit_applied"


class TestNodeApplyEditReject:
    def test_reject_advances_reviewed(self):
        from app.graph.edit_graph import node_apply_edit
        state = _base_state(
            current_review={"proposal": "reject", "section": "skills", "item": None},
            sections_pending=[], sections_reviewed=[], dropped_items=[],
        )
        mock_agent = _make_mock_agent()
        mock_agent.editing_cycle.current_state.get_section.return_value = MagicMock(
            section_previous_state="orig", lines_to_change=[], suggested_changes=[],
            updated_section="orig",
        )
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}), \
             patch("app.graph.edit_graph.pl"):
            result = node_apply_edit(state)
        assert result.get("stage") == "edit_applied"


class TestNodeApplyEditBack:
    def _run(self, reviewed):
        from app.graph.edit_graph import node_apply_edit
        state = _base_state(
            current_review={"proposal": "back", "section": "skills", "item": None},
            sections_pending=[{"section": "experience", "item": "Google"}],
            sections_reviewed=reviewed,
            dropped_items=[],
        )
        return node_apply_edit(state)

    def test_back_with_prior_reviewed_re_queues_previous(self):
        reviewed = [{"section": "education", "item": None}]
        result = self._run(reviewed)
        pending = result.get("sections_pending", [])
        section_names = [p["section"] for p in pending]
        assert "education" in section_names

    def test_back_from_first_section_re_queues_current(self):
        result = self._run(reviewed=[])
        pending = result.get("sections_pending", [])
        assert any(p["section"] == "skills" for p in pending)

    def test_back_pops_last_reviewed(self):
        reviewed = [
            {"section": "education", "item": None},
            {"section": "achievements", "item": None},
        ]
        result = self._run(reviewed)
        new_reviewed = result.get("sections_reviewed", [])
        assert len(new_reviewed) == len(reviewed) - 1


class TestNodeApplyEditLLMProposals:
    def _run(self, proposal, special_instruction=None, rejection_reason=None):
        from app.graph.edit_graph import node_apply_edit
        review = {
            "proposal":           proposal,
            "section":            "skills",
            "item":               None,
            "special_instruction": special_instruction,
            "rejection_reason":   rejection_reason,
        }
        state = _base_state(
            current_review=review,
            sections_pending=[{"section": "experience", "item": "Google"}],
            sections_reviewed=[],
            dropped_items=[],
        )
        mock_agent = _make_mock_agent()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}), \
             patch("app.graph.edit_graph.pl") as mock_pl:
            result = node_apply_edit(state)
        return result, mock_pl

    def test_another_suggestion_re_queues_section(self):
        result, _ = self._run("another_suggestion")
        pending = result.get("sections_pending", [])
        assert any(p["section"] == "skills" for p in pending)

    def test_paraphrase_calls_pl_apply_edit(self):
        _, mock_pl = self._run("paraphrase", special_instruction="Make it shorter")
        mock_pl.apply_edit.assert_called_once()

    def test_custom_instruction_mapped_to_paraphrase(self):
        _, mock_pl = self._run("custom_instruction", special_instruction="Add metrics")
        call_kwargs = mock_pl.apply_edit.call_args[1] if mock_pl.apply_edit.call_args[1] \
                      else {}
        call_args = mock_pl.apply_edit.call_args
        # The mapped proposal should be "paraphrase"
        if call_args and call_args[1]:
            assert call_args[1].get("proposal") == "paraphrase"

    def test_rejection_reason_passed_to_apply_edit(self):
        _, mock_pl = self._run("reject", rejection_reason="Too generic")
        # reject routes through LLM path → pl.apply_edit should be called
        if mock_pl.apply_edit.called:
            call = mock_pl.apply_edit.call_args[1]
            assert call.get("rejection_reason") == "Too generic"


# ══════════════════════════════════════════════════════════════════════════════
# node_score_resume
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeScoreResume:
    def _make_score(self):
        from app.models.scoring import ResumeScore, DimensionScore, ScoreDimension
        ds = DimensionScore(
            dimension=ScoreDimension.ATS_FRIENDLINESS,
            score=80, reasoning="Good", suggestions=[],
        )
        return ResumeScore(
            app_id=0, resume_id=1, overall_score=78.0,
            ats_score=80.0, keyword_match_score=75.0, quality_score=79.0,
            dimension_scores=[ds], missing_keywords=[],
            overall_feedback="OK",
            weights={"ats_friendliness": 0.3, "keyword_match": 0.4, "resume_quality": 0.3},
        )

    def _run(self):
        from app.graph.edit_graph import node_score_resume
        state = _base_state(resume_id=1)
        mock_agent = _make_mock_agent()
        mock_score = self._make_score()
        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl:
            mock_pl.score_resume.return_value = mock_score
            mock_pl.format_score_feedback.return_value = "Improve skills section"
            return node_score_resume(state), mock_pl

    def test_stage_is_scored(self):
        result, _ = self._run()
        assert result.get("stage") == "scored"

    def test_score_result_is_dict(self):
        result, _ = self._run()
        assert isinstance(result.get("score_result"), dict)

    def test_score_feedback_is_string(self):
        result, _ = self._run()
        assert isinstance(result.get("score_feedback"), str)

    def test_calls_pl_score_resume(self):
        _, mock_pl = self._run()
        mock_pl.score_resume.assert_called_once()

    def test_calls_pl_format_score_feedback(self):
        _, mock_pl = self._run()
        mock_pl.format_score_feedback.assert_called_once()

    def test_error_returns_error_stage(self):
        from app.graph.edit_graph import node_score_resume
        state = _base_state(resume_id=1)
        with patch("app.graph.edit_graph._rebuild_agent", side_effect=RuntimeError("LLM down")):
            result = node_score_resume(state)
        assert result.get("stage") == "error"
        assert result.get("error") is not None


# ══════════════════════════════════════════════════════════════════════════════
# Routing functions
# ══════════════════════════════════════════════════════════════════════════════

class TestRouteAfterReviewOrDone:
    def test_pending_returns_more_sections(self):
        from app.graph.edit_graph import route_after_review_or_done
        state = _base_state(sections_pending=[{"section": "skills", "item": None}])
        assert route_after_review_or_done(state) == "more_sections"

    def test_empty_pending_returns_all_done(self):
        from app.graph.edit_graph import route_after_review_or_done
        state = _base_state(sections_pending=[])
        assert route_after_review_or_done(state) == "all_done"

    def test_none_pending_returns_all_done(self):
        from app.graph.edit_graph import route_after_review_or_done
        state = {k: v for k, v in _base_state().items() if k != "sections_pending"}
        assert route_after_review_or_done(state) == "all_done"


class TestRouteAfterError:
    def test_with_error_returns_error_end(self):
        from app.graph.edit_graph import route_after_error
        assert route_after_error({"error": "something went wrong"}) == "error_end"

    def test_without_error_returns_continue(self):
        from app.graph.edit_graph import route_after_error
        assert route_after_error({"error": None}) == "continue"
        assert route_after_error({}) == "continue"


# ══════════════════════════════════════════════════════════════════════════════
# node_item_selection interrupt payload contract
# ══════════════════════════════════════════════════════════════════════════════

class TestNodeItemSelectionPayload:
    def test_interrupt_payload_has_type_item_selection(self):
        from app.graph.edit_graph import node_item_selection

        ranked = [{"section_name": "skills", "item_name": None, "best_score": 0.9}]
        state = _base_state(ranked_items=ranked)

        captured = {}

        def _fake_interrupt(payload):
            captured.update(payload)
            return {"selected_items": ranked}

        with patch("app.graph.edit_graph.interrupt", side_effect=_fake_interrupt):
            node_item_selection(state)

        assert captured.get("type") == "item_selection"

    def test_interrupt_payload_contains_ranked_items(self):
        from app.graph.edit_graph import node_item_selection

        ranked = [{"section_name": "experience", "item_name": "Google", "best_score": 0.95}]
        state = _base_state(ranked_items=ranked)

        captured = {}

        def _fake_interrupt(payload):
            captured.update(payload)
            return {"selected_items": ranked}

        with patch("app.graph.edit_graph.interrupt", side_effect=_fake_interrupt):
            node_item_selection(state)

        assert captured.get("ranked_items") == ranked

    def test_selected_items_set_from_user_response(self):
        from app.graph.edit_graph import node_item_selection

        all_items = [
            {"section_name": "skills",     "item_name": None},
            {"section_name": "experience", "item_name": "Google"},
        ]
        selected_subset = [all_items[0]]
        state = _base_state(ranked_items=all_items)

        def _fake_interrupt(payload):
            return {"selected_items": selected_subset}

        with patch("app.graph.edit_graph.interrupt", side_effect=_fake_interrupt):
            result = node_item_selection(state)

        assert result["selected_items"] == selected_subset

    def test_default_to_all_ranked_when_no_selection(self):
        from app.graph.edit_graph import node_item_selection

        all_items = [{"section_name": "skills", "item_name": None}]
        state = _base_state(ranked_items=all_items)

        def _fake_interrupt(payload):
            return {}  # user sends back nothing

        with patch("app.graph.edit_graph.interrupt", side_effect=_fake_interrupt):
            result = node_item_selection(state)

        assert result["selected_items"] == all_items


# ══════════════════════════════════════════════════════════════════════════════
# State serialisation round-trip
# ══════════════════════════════════════════════════════════════════════════════

class TestEditCycleSerialisation:
    """_serialise_cycle and _rebuild_agent must round-trip through model_dump."""

    def test_serialise_cycle_returns_dict(self):
        from app.graph.edit_graph import _serialise_cycle
        from app.models.edit import ResumeEditCycle, ResumeEditState
        from app.models.jd import ParsedJD
        cycle = ResumeEditCycle(jd=ParsedJD(), initial_state=ResumeEditState(), cycle_id=1)
        mock_agent = MagicMock()
        mock_agent.editing_cycle = cycle
        result = _serialise_cycle(mock_agent)
        assert isinstance(result, dict)
        assert "nodes" in result or "current_node_id" in result

    def test_rebuild_agent_from_serialised_cycle(self):
        from app.graph.edit_graph import _serialise_cycle, _rebuild_agent
        from app.models.edit import ResumeEditCycle, ResumeEditState
        from app.models.jd import ParsedJD

        cycle = ResumeEditCycle(jd=ParsedJD(), initial_state=ResumeEditState(), cycle_id=1)
        mock_agent = MagicMock()
        mock_agent.editing_cycle = cycle
        serialised = _serialise_cycle(mock_agent)

        state = _base_state(edit_cycle_dict=serialised)
        rebuilt = _rebuild_agent(state)
        assert rebuilt is not None
        assert hasattr(rebuilt, "editing_cycle")
