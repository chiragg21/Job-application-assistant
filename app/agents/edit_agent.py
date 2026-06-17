# src/agents/edit_agent.py

import hashlib
import json
from typing import Optional
from app.core.retriever import ATOMIC_SECTIONS, FLAT_SECTIONS
from app.utils import SQLHandler, get_logger
from app.utils.llm import generate_for_task, Prompt
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
    ):
        self.editing_cycle = editing_cycle
        self.editing_cycle.init_root(resume)

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

    # System prompt shared by all per-section parallel calls.
    def _per_section_system_prompt(self) -> str:
        return f"""
You are an expert technical resume writer specialising in software engineering and data/ML roles.

Your task is to rewrite resume bullets so they pass ATS filters AND impress human reviewers.

## REWRITING RULES (apply in this order)

1. **Lead with impact** — put the outcome or metric first, not the action.
   Weak:  "Developed a pipeline that reduced training time"
   Strong: "Cut model training time 40 % by redesigning the data pipeline"

2. **Use the X → Y → Z formula where data permits**
   "Achieved [result] by [action] using [method/tool]"
   Only use real numbers from the original line. Never invent metrics.

3. **Mirror JD keywords naturally** — do not force-fit every keyword.

4. **Action verb upgrade** — replace weak openers (Worked on, Helped, Assisted, Was responsible for)
   with strong verbs (Engineered, Designed, Reduced, Automated, Deployed, Led).

5. **Trim filler** — remove phrases like "successfully", "various", "multiple", "as part of a team"
   unless they add meaning.

6. **LaTeX hygiene** — preserve all commands (\\item, \\textbf{{}}, \\href{{}}{{}}). Use \\textbf{{}}
   only inside experience/project bullets to highlight one key metric or tool per bullet.
   Never bold items in the skills section.

## BULLET RELEVANCE
Identify any `\\item` lines that are clearly irrelevant to this JD and list them in
`bullets_to_drop`.  Use a high bar: only drop if the bullet adds no value for THIS role.
When in doubt, leave it in — the user makes the final decision.

## HARD CONSTRAINTS
- Do NOT invent tools, companies, metrics, or responsibilities.
- Do NOT change factual meaning.
- Only return lines that genuinely improve — empty lists are fine.

{self._jd_context()}

{self._special_instruction(self.global_instruction)}

{self._prompt_guardrails()}

## OUTPUT SCHEMA
{OneLlmOutput.model_json_schema()}
"""

    def _call_one_section(
        self,
        sec: str,
        item_name: str | None,
        content: str,
        system_prompt: str,
    ) -> OneLlmOutput:
        """Make one LLM call for a single section/item. Returns OneLlmOutput."""
        label = f"{sec}/{item_name}" if item_name else sec
        user_prompt = f"""
## TASK
Review this LaTeX resume section and suggest improved wording.

Return the exact original lines and their replacements.
Also populate `bullets_to_drop` with any \\item lines clearly irrelevant to this JD.
If no changes are needed, return empty lists.

## INPUT
Section: {label}

{content}
"""
        try:
            result = generate_for_task(
                "editing",
                Prompt(system=system_prompt, user=user_prompt),
                OneLlmOutput,
            )
            if result.schema_matched and result.parsed is not None:
                return result.parsed
        except Exception:
            pass
        return OneLlmOutput(lines_to_change=[], suggested_changes=[])

    def _build_input_str(self) -> str:
        """Build the canonical full-resume input string used as the cache key."""
        parts = []
        for sec in FLAT_SECTIONS:
            sec_state = getattr(self.editing_cycle.current_state, sec)
            content = sec_state.updated_section if sec_state else ""
            parts.append(f"{sec}: {content}\n\n")
        for sec in ATOMIC_SECTIONS:
            items = getattr(self.editing_cycle.current_state, sec) or []
            parts.append(f"{sec}-\n")
            for item in items:
                if item is None:
                    continue
                parts.append(f"{item.item_name}: {item.updated_section}\n")
        return "".join(parts)

    def _apply_whole_output_to_state(
        self, response: WholeLlmOutput
    ) -> "ResumeEditState":
        """
        Assemble a new ResumeEditState from a WholeLlmOutput (cached or freshly
        built from parallel calls). Returns the new state and pushes it onto the
        cycle.
        """
        next_state = ResumeEditState()
        for sec in FLAT_SECTIONS:
            cur = getattr(self.editing_cycle.current_state, sec)
            if cur is None:
                continue
            sugg: OneLlmOutput = getattr(response, sec)
            prev = cur.updated_section
            updated = prev
            for orig, repl in zip(sugg.lines_to_change, sugg.suggested_changes):
                updated = updated.replace(orig, repl)
            setattr(next_state, sec, SectionEditState(
                section_name=sec,
                section_previous_state=prev,
                lines_to_change=sugg.lines_to_change,
                suggested_changes=sugg.suggested_changes,
                updated_section=updated,
                bullets_dropped=sugg.bullets_to_drop,
            ))
        for sec in ATOMIC_SECTIONS:
            prev_items = getattr(self.editing_cycle.current_state, sec) or []
            item_suggs: list[OneLlmOutput] = getattr(response, sec) or []
            _empty = OneLlmOutput(lines_to_change=[], suggested_changes=[])
            sugg_by_name = {s.name: s for s in item_suggs if s and s.name}
            new_sec = []
            for i, prev_item in enumerate(prev_items):
                if prev_item is None:
                    continue
                sugg = (
                    sugg_by_name.get(prev_item.item_name)
                    or (item_suggs[i] if i < len(item_suggs) else _empty)
                )
                updated = prev_item.updated_section
                for orig, repl in zip(sugg.lines_to_change, sugg.suggested_changes):
                    updated = updated.replace(orig, repl)
                new_sec.append(ItemEditState(
                    section_name=sec,
                    item_name=prev_item.item_name,
                    section_previous_state=prev_item.updated_section,
                    lines_to_change=sugg.lines_to_change,
                    suggested_changes=sugg.suggested_changes,
                    updated_section=updated,
                    bullets_dropped=sugg.bullets_to_drop,
                ))
            setattr(next_state, sec, new_sec)
        self.editing_cycle.push(next_state, "resume_suggestion", "resume")
        return next_state

    def _resume_suggestions(self, thread_id: str | None = None) -> None:
        """
        Generate improvement suggestions for all resume sections.

        When *thread_id* is provided, suggestions are generated in parallel
        (one LLM call per section/item) and each result is pushed to the SSE
        queue via app.core.stream so the frontend receives sections as they
        complete.  When thread_id is None the same parallel approach is used
        but without SSE push (streaming not requested).

        The full result is cached as WholeLlmOutput JSON so that repeat runs
        with the same resume + JD content return immediately.
        """
        import concurrent.futures
        from app.core import stream as stream_module

        jd = self.editing_cycle.jd
        jd_hash = hashlib.sha256(
            json.dumps(jd.model_dump(), sort_keys=True).encode()
        ).hexdigest()[:16]
        input_str = self._build_input_str()
        cache_key = suggestions_cache.make_key(input_str, jd_hash)

        # ── Reset SSE queue before this pass ──────────────────────────────
        if thread_id:
            stream_module.reset(thread_id)

        # ── Cache hit: push all sections immediately, then done ───────────
        cached = suggestions_cache.get(cache_key)
        if cached is not None:
            response = WholeLlmOutput.model_validate_json(cached)
            if thread_id:
                for sec in FLAT_SECTIONS:
                    sugg: OneLlmOutput = getattr(response, sec)
                    if sugg:
                        stream_module.put(thread_id, {
                            "type":             "section_done",
                            "section":          sec,
                            "item":             None,
                            "lines_to_change":  sugg.lines_to_change,
                            "suggested_changes": sugg.suggested_changes,
                            "bullets_to_drop":  sugg.bullets_to_drop,
                        })
                for sec in ATOMIC_SECTIONS:
                    for sugg in (getattr(response, sec) or []):
                        if sugg:
                            stream_module.put(thread_id, {
                                "type":             "section_done",
                                "section":          sec,
                                "item":             sugg.name,
                                "lines_to_change":  sugg.lines_to_change,
                                "suggested_changes": sugg.suggested_changes,
                                "bullets_to_drop":  sugg.bullets_to_drop,
                            })
                stream_module.put(thread_id, {"type": "done"})
            self._apply_whole_output_to_state(response)
            return

        # ── Cache miss: build work units, fan out, stream results ─────────
        # Collect (sec, item_name, content) for every present section/item.
        units: list[tuple[str, str | None, str]] = []
        for sec in FLAT_SECTIONS:
            state = getattr(self.editing_cycle.current_state, sec)
            if state:
                units.append((sec, None, state.updated_section))
        for sec in ATOMIC_SECTIONS:
            for item in (getattr(self.editing_cycle.current_state, sec) or []):
                if item:
                    units.append((sec, item.item_name, item.updated_section))

        if not units:
            if thread_id:
                stream_module.put(thread_id, {"type": "done"})
            return

        system_prompt = self._per_section_system_prompt()
        per_section_results: dict[tuple[str, str | None], OneLlmOutput] = {}

        def _call(unit: tuple[str, str | None, str]):
            sec, item_name, content = unit
            out = self._call_one_section(sec, item_name, content, system_prompt)
            stream_module.put(thread_id, {
                "type":             "section_done",
                "section":          sec,
                "item":             item_name,
                "lines_to_change":  out.lines_to_change,
                "suggested_changes": out.suggested_changes,
                "bullets_to_drop":  out.bullets_to_drop,
            })
            return (sec, item_name), out

        max_workers = min(len(units), 6)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_call, u): u for u in units}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    key, out = fut.result()
                    per_section_results[key] = out
                except Exception:
                    sec, item_name, _ = futures[fut]
                    per_section_results[(sec, item_name)] = OneLlmOutput(
                        lines_to_change=[], suggested_changes=[]
                    )

        if thread_id:
            stream_module.put(thread_id, {"type": "done"})

        # ── Assemble WholeLlmOutput, cache it, apply to cycle ────────────
        _empty = OneLlmOutput(lines_to_change=[], suggested_changes=[])
        experience_items = [
            it for it in (getattr(self.editing_cycle.current_state, "experience") or []) if it
        ]
        projects_items = [
            it for it in (getattr(self.editing_cycle.current_state, "projects") or []) if it
        ]

        def _with_name(sec: str, item_name: str | None) -> OneLlmOutput:
            out = per_section_results.get((sec, item_name), _empty)
            if item_name and not out.name:
                out = OneLlmOutput(
                    name=item_name,
                    lines_to_change=out.lines_to_change,
                    suggested_changes=out.suggested_changes,
                    bullets_to_drop=out.bullets_to_drop,
                )
            return out

        response = WholeLlmOutput(
            education=per_section_results.get(("education", None), _empty),
            achievements=per_section_results.get(("achievements", None), _empty),
            skills=per_section_results.get(("skills", None), _empty),
            relevant_coursework=per_section_results.get(("relevant_coursework", None), _empty),
            experience=[_with_name("experience", it.item_name) for it in experience_items],
            projects=[_with_name("projects", it.item_name) for it in projects_items],
        )
        suggestions_cache.set(cache_key, response.model_dump_json())
        self._apply_whole_output_to_state(response)

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
        result = generate_for_task(
            "editing",
            Prompt(system=system_prompt, user=prompt),
            OneLlmOutput,
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

        result = generate_for_task(
            "editing",
            Prompt(system=system_prompt, user=prompt),
            OneLlmOutput,
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