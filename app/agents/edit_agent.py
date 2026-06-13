# src/agents/edit_agent.py

import hashlib
import json
from typing import Optional
from app.core.retriever import ATOMIC_SECTIONS, FLAT_SECTIONS
from app.utils import SQLHandler, get_logger
from app.utils.llm import llm as _llm, Prompt
from app.models.edit import (
    ItemEditState, SectionEditState, ResumeEditState, ResumeEditCycle, ResumeStateNode, OneLlmOutput, WholeLlmOutput
)
from app.models.jd import ParsedJD
from app.core.gen_cache import suggestions_cache
from config.config import get_config_dict

resume_config = get_config_dict()['resume_defaults']
FLAT_SECTIONS = resume_config['flat_sections']
ATOMIC_SECTIONS = resume_config['atomic_sections']



class EditAgent:
    def __init__(
        self,
        editing_cycle: ResumeEditCycle,
        resume: ResumeEditState,
        usellm: str = "gemini",
    ):
        self.editing_cycle = editing_cycle
        self.editing_cycle.init_root(resume)

        self.llmhandler = _llm
        self.db = SQLHandler()

        self.section_name       = ""
        self.item_name          = ""
        self.special_instruction  = ""   # per-call (score feedback, paraphrase instruction)
        self.global_instruction   = ""   # session-wide custom instruction

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
    - Use \\textbf{} only inside experience or project bullet points to highlight a key achievement or metric. Never add \\textbf{} to the skills section — skills are already listed as keywords and bolding them adds no value.
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
        req   = jd.required_skills[:10]
        resp  = jd.responsibilities[:5]
        return f"""
    ## JOB DESCRIPTION CONTEXT
    Align wording with these — do not invent skills or experience that are absent.

    Required Skills (top 10):
    {req}

    Key Responsibilities (top 5):
    {resp}
    """
    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _resume_suggestions(self):
        input_str = ""
        for sec in FLAT_SECTIONS:
            sec_state: SectionEditState = getattr(self.editing_cycle.current_state, sec)
            content = sec_state.updated_section if sec_state else ""
            input_str += sec + ": " + content + "\n\n"
        for sec in ATOMIC_SECTIONS:
            sec_item_state: list[ItemEditState] = getattr(self.editing_cycle.current_state, sec) or []
            input_str += sec + "-\n"
            for item in sec_item_state:
                if item is None:
                    continue
                input_str += item.item_name + ": " + item.updated_section + "\n"
        system_prompt = f"""
You are an expert technical resume writer specialising in software engineering and data/ML roles.

Your task is to rewrite resume bullets so they pass ATS filters AND impress human reviewers.

## REWRITING RULES (apply in this order)

1. **Lead with impact** — put the outcome or metric first, not the action.
   Weak:  "Developed a pipeline that reduced training time"
   Strong: "Cut model training time 40 % by redesigning the data pipeline"

2. **Use the X → Y → Z formula where data permits**
   "Achieved [result] by [action] using [method/tool]"
   Only use real numbers from the original line. Never invent metrics.

3. **Mirror JD keywords naturally** — if the JD says "MLOps" and the resume says
   "model deployment workflows", rewrite to include "MLOps". Do not force-fit every keyword.

4. **Action verb upgrade** — replace weak openers (Worked on, Helped, Assisted, Was responsible for)
   with strong verbs (Engineered, Designed, Reduced, Automated, Deployed, Led).

5. **Trim filler** — remove phrases like "successfully", "various", "multiple", "as part of a team"
   unless they add meaning.

6. **LaTeX hygiene** — preserve all commands (\\item, \\textbf{{}}, \\href{{}}{{}}). Use \\textbf{{}}
   only inside experience/project bullets to highlight one key metric or tool per bullet.
   Never bold items in the skills section.

## HARD CONSTRAINTS
- Do NOT invent tools, companies, metrics, or responsibilities.
- Do NOT change factual meaning.
- Only return lines that genuinely improve — empty lists are fine.

{self._jd_context()}

{self._special_instruction(self.global_instruction)}

{self._prompt_guardrails()}

## OUTPUT SCHEMA
{WholeLlmOutput.model_json_schema()}
"""
        prompt = f"""
## TASK
Review the LaTeX sections below and suggest improved wording.

For flat sections (education, skills, etc.) return an OneLlmOutput with name=null.
For atomic sections (experience, projects) return one OneLlmOutput per item with
`name` set to EXACTLY the item name shown in the input (e.g. "GIST Impact", "Job Assistant").
Return one entry per item even when no changes are needed (use empty lists in that case).

Identify ALL lines worth improving per section — return every line that can be meaningfully improved, not just the most obvious one.
Return the exact original lines and their replacements.

If no changes are needed for an item, return empty lists for that item.

## INPUT
{input_str}
"""
        jd = self.editing_cycle.jd
        jd_hash   = hashlib.sha256(
            json.dumps(jd.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:16]
        cache_key = suggestions_cache.make_key(input_str, jd_hash)
        cached    = suggestions_cache.get(cache_key)

        if cached is not None:
            response = WholeLlmOutput.model_validate_json(cached)
        else:
            result = self.llmhandler.generate(
                Prompt(system=system_prompt, user=prompt),
                provider="gemini",
                response_model=WholeLlmOutput,
            )
            if not result.schema_matched or result.parsed is None:
                raise ValueError("LLM response did not match WholeLlmOutput schema")
            response: WholeLlmOutput = result.parsed
            suggestions_cache.set(cache_key, response.model_dump_json())

        next_resume_state = ResumeEditState()
        for sec in FLAT_SECTIONS:
            current_sec_state = getattr(self.editing_cycle.current_state, sec)
            if current_sec_state is None:
                continue
            sec_suggestions: OneLlmOutput = getattr(response, sec)
            prev_state = current_sec_state.updated_section
            updated_state = prev_state
            for prev, new in zip(sec_suggestions.lines_to_change, sec_suggestions.suggested_changes):
                updated_state = updated_state.replace(prev, new)

            setattr(next_resume_state, sec, SectionEditState(**{"section_name": sec,
                                               "section_previous_state": prev_state,
                                               "lines_to_change": sec_suggestions.lines_to_change,
                                               "suggested_changes": sec_suggestions.suggested_changes,
                                               "updated_section": updated_state}))
        for sec in ATOMIC_SECTIONS:
            sec_prev_state = getattr(self.editing_cycle.current_state, sec) or []
            sec_item_sugg: list[OneLlmOutput] = getattr(response, sec) or []
            # Build a name→suggestion map for reliable matching; fall back to
            # positional index only when the LLM omits the name field.
            _empty_sugg = OneLlmOutput(lines_to_change=[], suggested_changes=[])
            sugg_by_name = {s.name: s for s in sec_item_sugg if s and s.name}
            new_sec = []
            for i, item_prev_state in enumerate(sec_prev_state):
                if item_prev_state is None:
                    continue
                item_sugg = (
                    sugg_by_name.get(item_prev_state.item_name)
                    or (sec_item_sugg[i] if i < len(sec_item_sugg) else _empty_sugg)
                )
                item_updated_state = item_prev_state.updated_section
                for prev, new in zip(item_sugg.lines_to_change, item_sugg.suggested_changes):
                    item_updated_state = item_updated_state.replace(prev, new)
                new_item = ItemEditState(section_name=sec,
                                         item_name=item_prev_state.item_name,
                                         section_previous_state=item_prev_state.updated_section,
                                         lines_to_change=item_sugg.lines_to_change,
                                         suggested_changes=item_sugg.suggested_changes,
                                         updated_section=item_updated_state)
                new_sec.append(new_item)
            setattr(next_resume_state, sec, new_sec)
        
        self.editing_cycle.push(next_resume_state, "resume_suggestion", "resume")

    def _paraphrase(self) -> None:
        idx = -1
        if self.section_name in FLAT_SECTIONS:
            item_to_update = self.editing_cycle.current_state.get_section(
                self.section_name
            )
        else:
            idx, item_to_update = self.editing_cycle.current_state.get_item(
                section_name=self.section_name, item_name=self.item_name
            )
        if item_to_update is None:
            raise ValueError(f"Section '{self.section_name}' not found in resume state.")
        
        system_prompt = f"""
You are an expert technical resume writer.

Rewrite the given LaTeX resume section to maximise impact and ATS relevance.

Apply these techniques:
- Lead bullets with a metric or outcome rather than the action ("Cut X by Y" not "Worked on reducing X")
- Replace weak openers (Worked on, Helped, Was responsible for) with strong action verbs
- Naturally incorporate relevant JD keywords where they fit without forcing them
- Remove filler words (successfully, various, multiple)
- Use \\textbf{{}} for one key metric or tool per bullet in experience/project sections only

Hard constraints:
- Do NOT invent tools, metrics, companies, or responsibilities
- Do NOT change factual meaning
- Preserve all LaTeX commands

{self._jd_context()}

{self._prompt_guardrails()}

## OUTPUT SCHEMA
{OneLlmOutput.model_json_schema()}
"""
        prompt = f"""
{self.special_instruction}

## TASK
Review the LaTeX section and suggest improved wording.

Modify only lines that can be improved.
Return the exact original lines and their replacements.

If no changes are needed, return empty lists.

## INPUT
{item_to_update.updated_section}
"""
        result = self.llmhandler.generate(
            Prompt(system=system_prompt, user=prompt),
            provider="gemini",
            response_model=OneLlmOutput,
        )
        if not result.schema_matched or result.parsed is None:
            raise ValueError("LLM response did not match OneLlmOutput schema")
        response: OneLlmOutput = result.parsed

        new_section = item_to_update.updated_section
        for original, replacement in zip(response.lines_to_change, response.suggested_changes):
            new_section = new_section.replace(original, replacement)

        updated_dict = {
            "section_name": self.section_name,
            "section_previous_state": item_to_update.updated_section,
            "lines_to_change": response.lines_to_change,
            "suggested_changes": response.suggested_changes,
            "updated_section": new_section,
        }
        if self.section_name in FLAT_SECTIONS:
            updated_section_state = SectionEditState(**updated_dict)
            new_resume_state = self.editing_cycle.current_state.set_section(
                self.section_name, updated_section_state
            )
        else:
            updated_dict['item_name'] = self.item_name
            updated_item = ItemEditState(**updated_dict)
            cur_items = list(getattr(self.editing_cycle.current_state, self.section_name) or [])
            new_items = [updated_item if (it and it.item_name == self.item_name) else it for it in cur_items]
            new_resume_state = self.editing_cycle.current_state.model_copy(update={self.section_name: new_items})

        self.editing_cycle.push(new_resume_state, action="paraphrase", section_name=self.section_name)

    def _reject(self) -> None:
        """Revert to parent node — discard last suggestion."""
        self.editing_cycle.revert()

    
    def _another_suggestion(self, rejection_reason: str | None = None) -> None:
        """
        Called when user wants another suggestion for the current section/item.
        - Collects all previous attempts for this section from the tree
        - LLM receives previous attempts and either:
            a) Understands what was wrong and suggests better edits
            b) Drops those lines entirely and finds different lines to improve
        - New attempt is pushed as a child of current node (branch, not overwrite)
        """
        if self.section_name in FLAT_SECTIONS:
            item_to_update = self.editing_cycle.current_state.get_section(self.section_name)
        else:
            _, item_to_update = self.editing_cycle.current_state.get_item(
                section_name=self.section_name, item_name=self.item_name
            )
        if item_to_update is None:
            raise ValueError(f"Section '{self.section_name}' not found in resume state.")

        previous_attempts = self._get_section_attempt_history(self.section_name, self.item_name or None)

        # format previous attempts for prompt
        attempts_str = ""
        for i, attempt in enumerate(previous_attempts, 1):
            attempts_str += f"\n### Attempt {i}\n"
            for orig, sugg in zip(attempt["lines_to_change"], attempt["suggested_changes"]):
                attempts_str += f"  Original:    {orig}\n"
                attempts_str += f"  Suggested:   {sugg}\n"

        rejection_str = f"\n## USER FEEDBACK ON LAST ATTEMPT\n{rejection_reason}" if rejection_reason and rejection_reason.strip() else ""

        system_prompt = f"""
    You are an expert technical resume editor.

    You have already suggested edits for a LaTeX resume section, but the user was not satisfied.
    Your job is to reflect on what went wrong and produce a meaningfully different suggestion.

    You have two options — choose whichever produces better results:
    A) Keep the same lines but rewrite the suggestions more effectively
    B) Drop those lines entirely and find different lines in the section worth improving

    Allowed:
    - Improve clarity, grammar, and phrasing
    - Slightly strengthen impact
    - Highlight important technical terms with \\textbf{{}} only in experience/project bullet points, never in the skills section

    Not allowed:
    - Invent skills, tools, or achievements
    - Change factual meaning
    - Repeat any previously attempted suggestion

    {self._jd_context()}

    {self._prompt_guardrails()}

    ## OUTPUT SCHEMA
    {OneLlmOutput.model_json_schema()}
    """

        prompt = f"""
    ## TASK
    The user rejected previous suggestion(s) for this section.
    Reflect on what was likely wrong, then produce a meaningfully different improvement.

    You may either improve the same lines differently, or find completely different lines to edit.
    Do NOT repeat any previously attempted suggestion.

    ## CURRENT SECTION CONTENT
    {item_to_update.updated_section}

    ## PREVIOUS ATTEMPTS (in order)
    {attempts_str if attempts_str else "No previous attempts recorded."}
    {rejection_str}

    {self._special_instruction(self.special_instruction)}
    """

        result = self.llmhandler.generate(
            Prompt(system=system_prompt, user=prompt),
            provider="gemini",
            response_model=OneLlmOutput,
        )
        if not result.schema_matched or result.parsed is None:
            raise ValueError("LLM response did not match OneLlmOutput schema")
        response: OneLlmOutput = result.parsed

        new_section = item_to_update.updated_section
        for original, replacement in zip(response.lines_to_change, response.suggested_changes):
            new_section = new_section.replace(original, replacement)

        updated_dict = {
            "section_name":          self.section_name,
            "section_previous_state": item_to_update.updated_section,
            "lines_to_change":       response.lines_to_change,
            "suggested_changes":     response.suggested_changes,
            "updated_section":       new_section,
        }
        if self.section_name in FLAT_SECTIONS:
            updated_section_state = SectionEditState(**updated_dict)
            new_resume_state = self.editing_cycle.current_state.set_section(
                self.section_name, updated_section_state
            )
        else:
            updated_dict['item_name'] = self.item_name
            updated_item = ItemEditState(**updated_dict)
            cur_items = list(getattr(self.editing_cycle.current_state, self.section_name) or [])
            new_items = [updated_item if (it and it.item_name == self.item_name) else it for it in cur_items]
            new_resume_state = self.editing_cycle.current_state.model_copy(update={self.section_name: new_items})

        self.editing_cycle.push(
            new_resume_state,
            action="another_suggestion",
            section_name=self.section_name,
        )
    #     """Revert to parent then re-paraphrase — branches off the same parent."""
    #     self.editing_cycle.revert()
    #     self._paraphrase()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def _get_section_attempt_history(self, section_name: str, item_name: str | None) -> list[dict]:
        """
        Walk the current branch from root to current node.
        Only collect nodes where THIS section/item was explicitly edited
        (node.section_name == section_name) to avoid duplicates from
        unrelated edits that carry the same section content unchanged.
        """
        attempts = []
        for node in self.editing_cycle.history():
            # skip root and nodes where a different section was edited
            if node.action in ("root", None):
                continue
            if node.section_name != section_name:
                continue

            if item_name:
                _, item_state = node.resume_state.get_item(section_name, item_name)
                if item_state and item_state.lines_to_change:
                    attempts.append({
                        "lines_to_change":  item_state.lines_to_change,
                        "suggested_changes": item_state.suggested_changes,
                    })
            else:
                sec_state = node.resume_state.get_section(section_name)
                if sec_state and sec_state.lines_to_change:
                    attempts.append({
                        "lines_to_change":  sec_state.lines_to_change,
                        "suggested_changes": sec_state.suggested_changes,
                    })

        return attempts

    def edit_section(
        self,
        section_name: str,
        proposal: str,
        item_name: str | None = None,
        special_instruction: str | None = None,
        rejection_reason: str | None = None,   # only used for another_suggestion
    ) -> None:
        self.section_name = section_name
        self.item_name = item_name if item_name else ""
        self.special_instruction = special_instruction or ""

        action_map = {
            "paraphrase":         self._paraphrase,
            "reject":             self._reject,
            "another_suggestion": lambda: self._another_suggestion(rejection_reason),
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