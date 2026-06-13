"""Tests for Sprint 1 EditAgent changes: global_instruction attribute and node_generate_suggestions."""
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_agent():
    """Create a minimal EditAgent without calling __init__ (avoids DB/LLM deps)."""
    from app.agents.edit_agent import EditAgent
    agent = object.__new__(EditAgent)
    agent.editing_cycle       = MagicMock()
    agent.section_name        = ""
    agent.item_name           = ""
    agent.special_instruction = ""
    agent.global_instruction  = ""
    agent.llmhandler          = MagicMock()
    agent.db                  = MagicMock()

    # _jd_context() reads these lists
    agent.editing_cycle.jd.required_skills = ["Python", "FastAPI"]
    agent.editing_cycle.jd.responsibilities = ["Build APIs"]
    # _resume_suggestions() calls json.dumps(jd.model_dump()) for the cache key
    agent.editing_cycle.jd.model_dump.return_value = {
        "required_skills": ["Python", "FastAPI"],
        "responsibilities": ["Build APIs"],
    }
    return agent


# ── global_instruction attribute ──────────────────────────────────────────────

class TestGlobalInstructionAttribute:

    def test_default_value_is_empty_string(self):
        agent = _make_agent()
        assert agent.global_instruction == ""

    def test_is_settable(self):
        agent = _make_agent()
        agent.global_instruction = "Focus on ML experience"
        assert agent.global_instruction == "Focus on ML experience"

    def test_survives_special_instruction_reset(self):
        """Changing special_instruction must not wipe global_instruction."""
        agent = _make_agent()
        agent.global_instruction = "Session rule"
        agent.special_instruction = "Per-call override"
        assert agent.global_instruction == "Session rule"
        assert agent.special_instruction == "Per-call override"


# ── _special_instruction helper ───────────────────────────────────────────────

class TestSpecialInstructionHelper:

    def test_non_empty_string_returns_section_header(self):
        agent = _make_agent()
        result = agent._special_instruction("Mirror the JD language")
        assert "Mirror the JD language" in result
        assert "SPECIAL INSTRUCTION" in result.upper() or "INSTRUCTION" in result

    def test_empty_string_returns_empty(self):
        agent = _make_agent()
        assert agent._special_instruction("") == ""

    def test_whitespace_only_returns_empty(self):
        agent = _make_agent()
        assert agent._special_instruction("   \n ") == ""

    def test_none_returns_empty(self):
        agent = _make_agent()
        assert agent._special_instruction(None) == ""


# ── global_instruction injected into _resume_suggestions prompt ───────────────

class TestGlobalInstructionInPrompt:

    def _fake_llm_result(self):
        from app.models.edit import WholeLlmOutput, OneLlmOutput
        output = WholeLlmOutput(
            education=OneLlmOutput(lines_to_change=[], suggested_changes=[]),
            achievements=OneLlmOutput(lines_to_change=[], suggested_changes=[]),
            skills=OneLlmOutput(lines_to_change=[], suggested_changes=[]),
            relevant_coursework=OneLlmOutput(lines_to_change=[], suggested_changes=[]),
            experience=[],
            projects=[],
        )
        r = MagicMock()
        r.schema_matched = True
        r.parsed = output
        return r

    def _setup_agent_current_state(self, agent):
        """Point the cycle's current_state sections to empty SectionEditState mocks."""
        from app.agents.edit_agent import FLAT_SECTIONS, ATOMIC_SECTIONS
        mock_state = MagicMock()
        for sec in FLAT_SECTIONS:
            sec_state = MagicMock()
            sec_state.updated_section = ""
            setattr(mock_state, sec, sec_state)
        for sec in ATOMIC_SECTIONS:
            setattr(mock_state, sec, [])
        agent.editing_cycle.current_state = mock_state

    # WholeLlmOutput fields: education, achievements, skills, relevant_coursework
    # The mock config FLAT_SECTIONS also includes "summary" which WholeLlmOutput lacks,
    # so we patch FLAT_SECTIONS/ATOMIC_SECTIONS to only the sections the model defines.
    _REAL_FLAT = ["education", "achievements", "skills", "relevant_coursework"]
    _REAL_ATOMIC = ["experience", "projects"]

    def test_global_instruction_appears_in_system_prompt(self):
        agent = _make_agent()
        agent.global_instruction = "Always emphasise quantified achievements"
        self._setup_agent_current_state(agent)

        result_mock = self._fake_llm_result()
        agent.llmhandler.generate.return_value = result_mock

        with patch("app.agents.edit_agent.suggestions_cache") as mock_cache, \
             patch("app.agents.edit_agent.FLAT_SECTIONS", self._REAL_FLAT), \
             patch("app.agents.edit_agent.ATOMIC_SECTIONS", self._REAL_ATOMIC):
            mock_cache.get.return_value = None
            mock_cache.make_key.return_value = "test-key"
            agent._resume_suggestions()

        from app.utils.llm import Prompt
        call_args = agent.llmhandler.generate.call_args
        prompt_obj: Prompt = call_args[0][0]
        assert "Always emphasise quantified achievements" in prompt_obj.system

    def test_empty_global_instruction_does_not_add_section(self):
        agent = _make_agent()
        agent.global_instruction = ""
        self._setup_agent_current_state(agent)

        result_mock = self._fake_llm_result()
        agent.llmhandler.generate.return_value = result_mock

        with patch("app.agents.edit_agent.suggestions_cache") as mock_cache, \
             patch("app.agents.edit_agent.FLAT_SECTIONS", self._REAL_FLAT), \
             patch("app.agents.edit_agent.ATOMIC_SECTIONS", self._REAL_ATOMIC):
            mock_cache.get.return_value = None
            mock_cache.make_key.return_value = "test-key"
            agent._resume_suggestions()

        call_args = agent.llmhandler.generate.call_args
        from app.utils.llm import Prompt
        prompt_obj: Prompt = call_args[0][0]
        # "SPECIAL INSTRUCTION" section should NOT appear when instruction is empty
        assert "SPECIAL INSTRUCTION" not in prompt_obj.system


# ── node_generate_suggestions ─────────────────────────────────────────────────

class TestNodeGenerateSuggestions:

    def _make_state(self, custom_instruction=None, score_feedback=None):
        from app.models.edit import ResumeEditCycle, ResumeEditState
        cycle = MagicMock()
        state: dict = {
            "edit_cycle_dict": {"nodes": {}, "current_node_id": 0},
            "parsed_jd": {
                "required_skills": [],
                "nice_to_have_skills": [],
                "responsibilities": [],
                "job_title": "Engineer",
                "company_name": "ACME",
                "experience_level": "mid",
                "domain": "software",
                "location": None,
                "remote": False,
                "salary_range": None,
            },
            "selected_items": [],
            "user_id": 1,
        }
        if custom_instruction:
            state["custom_instruction"] = custom_instruction
        if score_feedback:
            state["score_feedback"] = score_feedback
        return state

    def test_custom_instruction_set_on_agent_global_instruction(self):
        from app.graph.edit_graph import node_generate_suggestions

        mock_agent = MagicMock()
        mock_agent.editing_cycle = MagicMock()

        state = self._make_state(custom_instruction="Use action verbs")

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        assert mock_agent.global_instruction == "Use action verbs"

    def test_score_feedback_appended_to_global_instruction(self):
        from app.graph.edit_graph import node_generate_suggestions

        mock_agent = MagicMock()
        state = self._make_state(
            custom_instruction="Keep it concise",
            score_feedback="Improve skills section",
        )

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        assert "Keep it concise" in mock_agent.global_instruction
        assert "Improve skills section" in mock_agent.global_instruction

    def test_no_custom_instruction_global_instruction_is_empty(self):
        from app.graph.edit_graph import node_generate_suggestions

        mock_agent = MagicMock()
        state = self._make_state()

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        assert mock_agent.global_instruction == ""

    def test_score_feedback_calls_generate_with_feedback(self):
        from app.graph.edit_graph import node_generate_suggestions

        mock_agent = MagicMock()
        state = self._make_state(score_feedback="Score low on experience section")

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        mock_pl.generate_suggestions_with_score_feedback.assert_called_once_with(
            mock_agent, "Score low on experience section"
        )

    def test_no_score_feedback_calls_plain_generate(self):
        from app.graph.edit_graph import node_generate_suggestions

        mock_agent = MagicMock()
        state = self._make_state()

        with patch("app.graph.edit_graph._rebuild_agent", return_value=mock_agent), \
             patch("app.graph.edit_graph.pl") as mock_pl, \
             patch("app.graph.edit_graph._serialise_cycle", return_value={}):
            node_generate_suggestions(state)

        mock_pl.generate_suggestions.assert_called_once_with(mock_agent)
        mock_pl.generate_suggestions_with_score_feedback.assert_not_called()
