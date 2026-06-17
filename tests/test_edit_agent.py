"""Tests for app/agents/edit_agent.py"""
import pytest
from unittest.mock import MagicMock, patch

from app.agents.edit_agent import EditAgent
from app.models.edit import (
    OneLlmOutput,
    WholeLlmOutput,
    SectionEditState,
    ItemEditState,
    ResumeEditState,
    ResumeEditCycle,
)
from app.models.jd import ParsedJD


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_section():
    return SectionEditState(
        section_name="skills",
        section_previous_state="",
        updated_section=r"Python, PyTorch, Docker",
    )


@pytest.fixture
def atomic_item():
    return ItemEditState(
        section_name="experience",
        item_name="Google",
        section_previous_state="",
        updated_section=r"\item Built ML pipelines at Google.",
    )


@pytest.fixture
def resume_state(flat_section, atomic_item):
    edu = SectionEditState(
        section_name="education",
        section_previous_state="",
        updated_section="B.Tech IIT Delhi 2021",
    )
    return ResumeEditState(
        skills=flat_section,
        education=edu,
        experience=[atomic_item],
        projects=[],
    )


@pytest.fixture
def edit_cycle(sample_jd, resume_state):
    cycle = ResumeEditCycle(cycle_id=1, jd=sample_jd)
    cycle.init_root(resume_state)
    return cycle


@pytest.fixture
def agent(edit_cycle, resume_state, mock_sql):
    with patch("app.agents.edit_agent.SQLHandler", return_value=mock_sql):
        ag = EditAgent(editing_cycle=edit_cycle, resume=resume_state)
    return ag


# ── Prompt helpers (pure methods) ─────────────────────────────────────────────

class TestPromptHelpers:
    def test_guardrails_not_empty(self, agent):
        result = agent._prompt_guardrails()
        assert isinstance(result, str) and len(result) > 0

    def test_guardrails_contains_key_rule(self, agent):
        assert "LaTeX" in agent._prompt_guardrails()

    def test_special_instruction_none_returns_empty(self, agent):
        assert agent._special_instruction(None) == ""

    def test_special_instruction_blank_returns_empty(self, agent):
        assert agent._special_instruction("   ") == ""

    def test_special_instruction_formatted(self, agent):
        result = agent._special_instruction("Use strong verbs")
        assert "Use strong verbs" in result

    def test_jd_context_contains_required_skills(self, agent):
        ctx = agent._jd_context()
        assert "Python" in ctx

    def test_jd_context_contains_responsibilities(self, agent):
        ctx = agent._jd_context()
        assert "pipelines" in ctx or "ML" in ctx


# ── State machine: reject and checkout ───────────────────────────────────────

class TestRejectAndCheckout:
    def test_reject_reverts_to_parent(self, agent):
        """Push a state, then reject should revert to the previous one."""
        original_state = agent.editing_cycle.current_state
        new_state = original_state.model_copy()
        agent.editing_cycle.push(new_state, action="paraphrase", section_name="skills")

        assert agent.editing_cycle.current_node_id == 1

        agent.edit_section("skills", "reject")
        assert agent.editing_cycle.current_node_id == 0

    def test_reject_at_root_raises(self, agent):
        with pytest.raises(ValueError, match="root"):
            agent.edit_section("skills", "reject")

    def test_checkout_jumps_to_node(self, agent, resume_state):
        agent.editing_cycle.push(resume_state, action="paraphrase", section_name="skills")
        agent.checkout(0)
        assert agent.editing_cycle.current_node_id == 0

    def test_checkout_invalid_node_raises(self, agent):
        with pytest.raises(ValueError):
            agent.checkout(999)


# ── Paraphrase (LLM call) ─────────────────────────────────────────────────────

class TestParaphrase:
    def _make_one_llm_output(self):
        return OneLlmOutput(
            lines_to_change=[r"\item Built ML pipelines at Google."],
            suggested_changes=[r"\item Engineered and deployed ML pipelines at Google, improving throughput."],
        )

    def _mock_llm(self, agent, output):
        mock = MagicMock()
        mock.generate.return_value = MagicMock(schema_matched=True, parsed=output)
        agent.llmhandler = mock
        return mock

    def test_paraphrase_pushes_new_state(self, agent):
        self._mock_llm(agent, self._make_one_llm_output())
        agent.edit_section("skills", "paraphrase")
        assert agent.editing_cycle.current_node_id == 1

    def test_paraphrase_calls_generate(self, agent):
        output = self._make_one_llm_output()
        with patch("app.agents.edit_agent.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=output)
            agent.edit_section("skills", "paraphrase")
        mock_gtf.assert_called_once()

    def test_paraphrase_schema_mismatch_raises(self, agent):
        with patch("app.agents.edit_agent.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(schema_matched=False, parsed=None)
            with pytest.raises(ValueError, match="schema"):
                agent.edit_section("skills", "paraphrase")


# ── Another suggestion ────────────────────────────────────────────────────────

class TestAnotherSuggestion:
    def test_another_suggestion_pushes_new_state(self, agent):
        llm_output = OneLlmOutput(
            lines_to_change=["line1"],
            suggested_changes=["new line1"],
        )
        # First push a paraphrase state so another_suggestion has history
        paraphrase_state = agent.editing_cycle.current_state.model_copy()
        agent.editing_cycle.push(paraphrase_state, action="paraphrase", section_name="skills")

        agent.llmhandler = MagicMock()
        agent.llmhandler.generate.return_value = MagicMock(
            schema_matched=True, parsed=llm_output
        )
        agent.edit_section("skills", "another_suggestion")

        assert agent.editing_cycle.current_node_id >= 2


# ── Invalid proposal ──────────────────────────────────────────────────────────

class TestEditSectionInvalidProposal:
    def test_unknown_proposal_raises(self, agent):
        with pytest.raises(ValueError, match="Unknown proposal"):
            agent.edit_section("skills", "magic_edit")


# ── Accessors ─────────────────────────────────────────────────────────────────

class TestAccessors:
    def test_current_latex_returns_section_content(self, agent):
        latex = agent.current_latex("skills")
        assert "Python" in latex

    def test_current_latex_unknown_section_returns_none(self, agent):
        assert agent.current_latex("nonexistent_section") is None

    def test_history_starts_with_root(self, agent):
        history = agent.history()
        assert history[0].action == "root"

    def test_get_tree_returns_nodes_dict(self, agent):
        tree = agent.get_tree()
        assert isinstance(tree, dict)
        assert 0 in tree

    def test_complete_editing_returns_final_state(self, agent):
        final = agent.complete_editing()
        assert isinstance(final, ResumeEditState)


# ── _get_section_attempt_history ─────────────────────────────────────────────

class TestGetSectionAttemptHistory:
    def test_empty_at_root(self, agent):
        """Root-only history → no attempts recorded."""
        result = agent._get_section_attempt_history("skills", None)
        assert result == []

    def test_records_flat_section_attempt_after_paraphrase(self, agent):
        """After a paraphrase node, the section's edit history is captured."""
        # Build a new state that looks like it was edited
        edited_state = agent.editing_cycle.current_state.model_copy()
        edited_skills = SectionEditState(
            section_name="skills",
            section_previous_state="Python, PyTorch",
            lines_to_change=["Python, PyTorch"],
            suggested_changes=["Python, PyTorch, Docker"],
            updated_section="Python, PyTorch, Docker",
        )
        edited_state = edited_state.set_section("skills", edited_skills)
        agent.editing_cycle.push(edited_state, action="paraphrase", section_name="skills")

        result = agent._get_section_attempt_history("skills", None)
        assert len(result) == 1
        assert result[0]["lines_to_change"] == ["Python, PyTorch"]

    def test_ignores_unrelated_section_nodes(self, agent):
        """Nodes for a different section are excluded."""
        edited_state = agent.editing_cycle.current_state.model_copy()
        edu = SectionEditState(
            section_name="education",
            section_previous_state="B.Tech IIT Delhi 2021",
            lines_to_change=["B.Tech IIT Delhi 2021"],
            suggested_changes=["B.Tech in CS, IIT Delhi, 2021"],
            updated_section="B.Tech in CS, IIT Delhi, 2021",
        )
        edited_state = edited_state.set_section("education", edu)
        agent.editing_cycle.push(edited_state, action="paraphrase", section_name="education")

        # asking for skills history — should be empty
        result = agent._get_section_attempt_history("skills", None)
        assert result == []


# ── _resume_suggestions ───────────────────────────────────────────────────────

class TestResumeSuggestions:
    """Tests for _resume_suggestions, which runs the full-resume LLM edit."""

    def _make_whole_output(self):
        empty = OneLlmOutput(lines_to_change=[], suggested_changes=[])
        return WholeLlmOutput(
            education=empty,
            achievements=empty,
            skills=empty,
            relevant_coursework=empty,
            experience=[empty],
            projects=[empty],
        )

    def test_resume_suggestions_pushes_new_state(self, agent):
        agent.llmhandler = MagicMock()
        agent.llmhandler.generate.return_value = MagicMock(
            schema_matched=True,
            parsed=self._make_whole_output(),
        )
        # Only use sections that exist in ResumeEditState to avoid AttributeError
        with patch("app.agents.edit_agent.FLAT_SECTIONS", ["skills", "education"]), \
             patch("app.agents.edit_agent.ATOMIC_SECTIONS", []):
            agent._resume_suggestions()

        assert agent.editing_cycle.current_node_id == 1

    def test_resume_suggestions_schema_mismatch_raises(self, agent):
        # Schema mismatches are caught per-section and fall back to empty output;
        # _resume_suggestions still completes without raising.
        with patch("app.agents.edit_agent.generate_for_task") as mock_gtf, \
             patch("app.agents.edit_agent.suggestions_cache") as mock_cache, \
             patch("app.agents.edit_agent.FLAT_SECTIONS", ["skills"]), \
             patch("app.agents.edit_agent.ATOMIC_SECTIONS", []):
            mock_cache.get.return_value = None
            mock_cache.make_key.return_value = "test-key"
            mock_gtf.return_value = MagicMock(schema_matched=False, parsed=None)
            agent._resume_suggestions()  # must not raise
        assert agent.editing_cycle.current_node_id >= 1


# ── Paraphrase on atomic (item-level) section ────────────────────────────────

class TestParaphraseAtomicItem:
    def _make_one_llm_output(self):
        return OneLlmOutput(
            lines_to_change=[r"\item Built ML pipelines at Google."],
            suggested_changes=[r"\item Engineered scalable ML pipelines at Google."],
        )

    def test_paraphrase_atomic_item_pushes_state(self, agent):
        mock = MagicMock()
        mock.generate.return_value = MagicMock(
            schema_matched=True, parsed=self._make_one_llm_output()
        )
        agent.llmhandler = mock
        # "experience" is an atomic section; "Google" is the item_name in fixture
        agent.edit_section("experience", "paraphrase", item_name="Google")
        assert agent.editing_cycle.current_node_id == 1

    def test_paraphrase_missing_item_raises(self, agent):
        """Attempting to paraphrase a non-existent item raises ValueError."""
        agent.llmhandler = MagicMock()
        agent.llmhandler.generate.return_value = MagicMock(
            schema_matched=True,
            parsed=OneLlmOutput(lines_to_change=[], suggested_changes=[]),
        )
        with pytest.raises(ValueError):
            agent.edit_section("experience", "paraphrase", item_name="NonExistent")
