# src/agents/edit_agent.py

from typing import Optional
from utils.sqlite_handler import SQLHandler
from utils.llm_handler import LLMHandler
from data_models.resume_edit_dm import (
    SectionEditState, ResumeEditState, ResumeEditCycle, ResumeStateNode, LlmOutput
)
from data_models.jd_parse_dm import ParsedJD


class EditAgent:
    def __init__(
        self,
        cycle_id: int,
        jd: ParsedJD,
        resume: ResumeEditState,
        api_key: str | None = None,
        usellm: str = "gemini",
    ):
        self.editing_cycle = ResumeEditCycle(cycle_id=cycle_id, jd=jd)
        self.editing_cycle.init_root(resume)

        self.llmhandler = LLMHandler(llm_type=usellm)
        self.db = SQLHandler()

        self.section_name = ""
        self.special_instruction = ""

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _prompt_guardrails(self) -> str:
        return """
    ## RULES
    - Do not invent skills, projects, or experiences.
    - Incase of conflict with special instructions, always give priority to special instructions.
    - Preserve the original meaning.
    - Keep sentence length roughly similar.
    - Maintain valid LaTeX syntax and formatting.
    - Preserve bullets and commands (e.g., \\item, \\textbf{}).
    - Emphasize important technical keywords with \\textbf{} only if helpful.
    - If no improvement is needed, return empty lists.
    - Output must strictly follow the JSON schema.
    """

    def _special_instruction(self, special_instruction: str | None) -> str:
        if special_instruction and special_instruction.strip():
            return f"""
    ## SPECIAL INSTRUCTION (Highest Priority)
    {special_instruction}
    """
        return ""

    def _jd_context(self) -> str:
        jd = self.editing_cycle.jd
        return f"""
    ## JOB DESCRIPTION CONTEXT
    Use only for wording alignment. Do not add missing skills.

    Required Skills:
    {jd.required_skills}

    Responsibilities:
    {jd.responsibilities}

    Nice to Have:
    {jd.nice_to_have_skills}
    """
    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _paraphrase(self) -> None:
        current_section = self.editing_cycle.current_state.get_section(
            self.section_name
        )
        if current_section is None:
            raise ValueError(f"Section '{self.section_name}' not found in resume state.")
        
        system_prompt = f"""
You are an expert technical resume editor.

Improve wording of sentences in a LaTeX resume section while keeping meaning unchanged.

Allowed:
- Improve clarity, grammar, and phrasing
- Slightly strengthen impact
- Highlight important technical terms

Not allowed:
- Invent skills, tools, or achievements
- Change factual meaning

Only suggest edits for lines that can be improved.

{self._jd_context()}

{self._prompt_guardrails()}
"""
        prompt = f"""
{self.special_instruction}

## TASK
Review the LaTeX section and suggest improved wording.

Modify only lines that can be improved.
Return the exact original lines and their replacements.

If no changes are needed, return empty lists.

## INPUT SECTION
{current_section.updated_section}

## OUTPUT SCHEMA
{LlmOutput.model_json_schema()}
"""
        response = self.llmhandler.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            response_model=LlmOutput,
        )

        new_section = current_section.updated_section
        for original, replacement in zip(response['lines_to_change'], response['suggested_changes']):
            new_section = new_section.replace(original, replacement)

        updated_section_state = SectionEditState(
            section_name=self.section_name,
            section_previous_state=current_section.updated_section,
            lines_to_change=response['lines_to_change'],
            suggested_changes=response['suggested_changes'],
            updated_section=new_section,
        )

        new_resume_state = self.editing_cycle.current_state.set_section(
            self.section_name, updated_section_state
        )
        self.editing_cycle.push(new_resume_state, action="paraphrase", section_name=self.section_name)

    def _reject(self) -> None:
        """Revert to parent node — discard last suggestion."""
        self.editing_cycle.revert()

    def _another_suggestion(self) -> None:
        """Revert to parent then re-paraphrase — branches off the same parent."""
        self.editing_cycle.revert()
        self._paraphrase()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def edit_section(
        self,
        section_name: str,
        proposal: str,
        special_instruction: str | None = None,
    ) -> None:
        self.section_name = section_name
        self.special_instruction = self._special_instruction(special_instruction)

        action_map = {
            "paraphrase":         self._paraphrase,
            "reject":             self._reject,
            "another_suggestion": self._another_suggestion,
        }

        action = action_map.get(proposal.lower())
        if action is None:
            raise ValueError(
                f"Unknown proposal: '{proposal}'. Valid options: {list(action_map.keys())}"
            )
        action()

    def checkout(self, node_id: int) -> None:
        """Jump to any node — next edit branches from there."""
        self.editing_cycle.checkout(node_id)

    def current_latex(self, section_name: str) -> Optional[str]:
        """Current latex for a section — for preview."""
        section = self.editing_cycle.current_state.get_section(section_name)
        return section.updated_section if section else None

    def get_tree(self) -> dict[int, ResumeStateNode]:
        """Full edit tree — expose to frontend for visualising branches."""
        return self.editing_cycle.nodes

    def history(self):
        """Active branch path from root to current node."""
        return self.editing_cycle.history()

    def complete_editing(self, add_to_db: bool = False) -> ResumeEditState:
        final_state = self.editing_cycle.current_state
        if add_to_db:
            # TODO: serialize and write final_state to resume_sections table
            pass
        return final_state