# Resume Tailoring System — Refined Workflow

> Every table reference below maps directly to the final schema.
> The unified `resume_section_items` table (child of `resume_sections`) replaces
> the older `resume_experience_items` and `resume_project_items` tables.

---

## Table Reference Cheat Sheet

| Table | Role in this workflow |
|---|---|
| `users` | Identity anchor — `user_id` threads through every table |
| `resumes` | Master resume record, holds `resume_path` to the `.tex` file |
| `resume_sections` | One row per section (experience, projects, skills, …) — holds full section blob |
| `resume_section_items` | One row per atomic block (one company, one project) — holds `content_latex`, `content_text`, `is_master`, `section_id` FK |
| `jobs` | One row per unique JD — deduplicated by `raw_hash` |
| `skills` | Normalised skill vocabulary |
| `job_skills` | JD ↔ skill join (`is_required` flag) |
| `resume_skills` | Resume ↔ skill join (user's demonstrated skills) |
| `applications` | Ties a `user_id` to a `job_id`; tracks `status`, `similarity_score`, `current_latex_snapshot` |
| `resume_edits` | Sentence-level edit log with full undo/redo tree via `parent_edit_id` |
| `generated_outputs` | All LLM-generated artefacts (drafts, cover letter, email) keyed to `application_id` |

---

## Stage 0 — Entry & Duplicate Guard

**Inputs:** `user_id`, `resume_id`, `jd_raw` text.

```
READ  users          WHERE id = user_id
READ  resumes        WHERE id = resume_id AND user_id = user_id
```

**Duplicate check:**

```
hash(jd_raw) → raw_hash

READ  jobs           WHERE raw_hash = raw_hash
```

| `jobs` lookup result | Next action |
|---|---|
| **Miss** — no matching `raw_hash` | Proceed to Stage 1 (parse JD) |
| **Hit** — job exists | Check `applications WHERE user_id = ? AND job_id = ?` |
| Hit + application exists + `generated_outputs.is_draft = 0` | Return the final output. **Stop.** |
| Hit + application exists + only `is_draft = 1` rows | Skip Stage 1 & 2. Resume from Stage 3. |
| Hit + no application yet | Create application row, skip Stage 1. Proceed to Stage 2. |

**On new application:**
```
INSERT applications (
    user_id, job_id,
    status          = 'draft',
    updated_at      = NOW()
)
→ app_id
```

---

## Stage 1 — JD Parsing

**Goal:** Decompose the raw JD into structured fields and a normalised skill list.

**Run:** `JDParser(jd_raw)` — extracts company, role, responsibilities, required/nice-to-have skills, seniority, location, remote flag, and a structured JSON summary.

**Writes:**

```
INSERT jobs (
    user_id, jd_raw,
    company, role,
    jd_parsed        = <structured JSON from parser>,
    seniority_level, location, is_remote,
    source_url,
    raw_hash         = hash(jd_raw),
    parsed_hash      = hash(jd_parsed)
)
→ job_id

UPDATE applications SET job_id = job_id WHERE id = app_id
```

**For each extracted skill:**
```
UPSERT skills (name, category)          → skill_id
INSERT job_skills (job_id, skill_id, is_required)
```

> `is_required = 1` for explicitly required skills, `0` for nice-to-have.

**Compute skill overlap against the user's resume:**
```
READ  resume_skills  WHERE user_id = user_id   → user_skill_ids
READ  job_skills     WHERE job_id = job_id     → jd_skill_ids

overlap_count = len(intersection(user_skill_ids, jd_skill_ids))

UPDATE applications SET skill_overlap_count = overlap_count
```

---

## Stage 2 — Semantic Retrieval (RAG)

**Goal:** Pull only the resume blocks most relevant to this JD — not the whole resume.

### Step A — Query ChromaDB

Use the responsibilities extracted in Stage 1 as the query text.

```python
results = chroma_col.query(
    query_texts = jd_responsibilities,   # list of strings from jd_parsed
    n_results   = TOP_K,
    where       = {"resume_id": resume_id}
)
# Each result carries metadata: sql_id, section_type, item_name, role_title
```

ChromaDB IDs follow the convention `sec_item_{sql_id}`, where `sql_id` maps directly to `resume_section_items.id`.

### Step B — Fetch Atomic Items from SQL

```
READ  resume_section_items
WHERE id IN (<sql_ids from ChromaDB results>)
  AND is_master = 1
→ returns: id, section_id, section_name, item_name, role_title,
           content_latex, content_text, item_index
```

Fetch the parent section blob for structural context (used in final assembly):

```
READ  resume_sections
WHERE id IN (<section_ids from above items>)
→ returns: id, section_name, content_latex
```

### Step C — Compute Similarity Score

```python
similarity_score = mean_cosine_similarity(jd_embedding, retrieved_item_embeddings)

UPDATE applications SET similarity_score = similarity_score
```

---

## Stage 3 — Phase I: Draft Generation

**Goal:** Produce two fast, non-interactive draft resumes as LaTeX.

**Inputs:**
- `resume_section_items.content_latex` for retrieved atomic items
- `resume_sections.content_latex` for flat sections (skills, education, etc.) from:
  ```
  READ  resume_sections
  WHERE resume_id = resume_id
    AND section_name IN ('skills', 'education', 'achievements', 'relevant_coursework')
  ```
- `jobs.jd_parsed` — for JD skills and keywords
- `resumes.resume_path` — for the master LaTeX template/preamble

**LLM Prompt (×2 variants):**

```
Draft 1 — ATS mode:
  "Assemble a LaTeX resume from these blocks [content_latex list].
   Maximise keyword density for: [job_skills WHERE is_required=1].
   Keep total line count ≤ MAX_LINES."

Draft 2 — Impact mode:
  "Assemble a LaTeX resume from these blocks [content_latex list].
   Lead with quantified achievements. Keywords: [job_skills].
   Keep total line count ≤ MAX_LINES."
```

**Writes:**
```
INSERT generated_outputs (
    application_id  = app_id,
    output_type     = 'resume_ats_draft' | 'resume_impact_draft',
    content         = <full LaTeX string>,
    model_used      = <model name>,
    prompt_tokens, completion_tokens,
    is_draft        = 1
)

UPDATE applications SET
    current_latex_snapshot = <draft LaTeX>,
    updated_at             = NOW()
```

> `current_latex_snapshot` in `applications` always holds the **latest working state** of the LaTeX, updated after every significant change.

---

## Stage 4 — Phase II: Agentic Edit Loop

**Goal:** Sentence-level personalisation with full user control and undo capability.

### The Edit State Machine

Each bullet point in a retrieved `resume_section_items` block is a discrete unit. The agent proposes one edit at a time and waits for user action before proceeding.

```
resume_edits.status values:
  'pending'     — agent has proposed, user has not acted yet
  'accepted'    — user accepted suggested_text as final_text
  'rejected'    — user discarded, original_text stands
  'paraphrased' — user modified; a child edit was created via parent_edit_id
```

### Loop Logic

```python
for item in retrieved_section_items:           # resume_section_items rows
    sentences = split_latex_into_sentences(item.content_latex)

    for idx, sentence in enumerate(sentences):
        proposed = agent.propose(sentence, jd_context)

        # --- WRITE: log the pending edit ---
        INSERT resume_edits (
            application_id = app_id,
            section        = item.section_name,   # 'experience', 'projects', etc.
            original_text  = sentence,
            suggested_text = proposed,
            status         = 'pending',
            sentence_index = idx,                 # position within this item's bullets
            line_count_delta = line_delta(sentence, proposed),
            parent_edit_id = NULL                 # root edit
        ) → edit_id

        action = wait_for_user(edit_id)           # API/frontend pauses here

        if action == 'accept':
            UPDATE resume_edits SET
                final_text = suggested_text,
                status     = 'accepted'
            WHERE id = edit_id

        elif action == 'reject':
            UPDATE resume_edits SET
                status = 'rejected'
            WHERE id = edit_id
            # original_text is implicitly preserved — no final_text needed

        elif action == 'paraphrase':
            # User provides feedback → agent re-proposes → new child record
            new_proposal = agent.re_propose(suggested_text, user_feedback)

            INSERT resume_edits (
                application_id = app_id,
                section        = item.section_name,
                original_text  = sentence,        # always the master original
                suggested_text = new_proposal,
                status         = 'pending',
                sentence_index = idx,
                line_count_delta = line_delta(sentence, new_proposal),
                parent_edit_id = edit_id          # links back to the root edit
            ) → child_edit_id
            # loop again from wait_for_user with child_edit_id

    # --- After all sentences in this item: snapshot running state ---
    UPDATE applications SET
        current_latex_snapshot = assemble_latex(app_id),
        updated_at             = NOW()
```

### Revert Logic

Revert walks up the `parent_edit_id` chain:

```python
def revert(edit_id):
    READ resume_edits WHERE id = edit_id → edit

    if edit.parent_edit_id is NULL:
        # Root — restore original_text
        return edit.original_text
    else:
        READ resume_edits WHERE id = edit.parent_edit_id → parent
        UPDATE resume_edits SET status = 'rejected' WHERE id = edit_id
        return parent.suggested_text   # restore to previous proposal
```

---

## Stage 5 — Final Assembly & Space Optimisation

**Goal:** Reconstruct the final LaTeX, enforce the one-page constraint, then store all deliverables.

### Step A — Reconstruct LaTeX

For every retrieved `resume_section_items` row, resolve its final text:

```python
def resolve_item_text(item_id, app_id):
    """
    Priority: last accepted edit > original content_latex
    """
    READ resume_edits
    WHERE application_id = app_id
      AND sentence_index IS NOT NULL
      AND status = 'accepted'
    ORDER BY id DESC
    # Group by sentence_index to get the latest accepted edit per sentence

    if accepted_edits exist for this item:
        # Splice accepted final_text back into content_latex at sentence_index positions
        return patched_latex
    else:
        READ resume_section_items WHERE id = item_id
        return content_latex    # untouched original
```

Flat sections (skills, education, etc.) come directly from `resume_sections.content_latex` — they have no `resume_section_items` children and are never edited in Phase II.

### Step B — Line Pressure Check

```python
final_latex = assemble_full_latex(resolved_items, flat_sections, template)

total_delta = SUM(
    SELECT line_count_delta FROM resume_edits
    WHERE application_id = app_id AND status = 'accepted'
)

line_count = count_latex_lines(final_latex)

if line_count > MAX_LINES:
    overage = line_count - MAX_LINES
    final_latex = agent.condense(
        latex   = final_latex,
        message = f"Resume is {overage} lines over limit. "
                   "Shorten the oldest experience item and summary "
                   f"while keeping keywords: {required_keywords}."
    )
```

### Step C — Write Final Outputs

```
# Save the compiled resume
INSERT generated_outputs (
    application_id = app_id,
    output_type    = 'resume_final',
    content        = final_latex,
    model_used     = <model>,
    prompt_tokens, completion_tokens,
    is_draft       = 0              ← marks this as the authoritative output
)

# Save cover letter
INSERT generated_outputs (
    application_id = app_id,
    output_type    = 'cover_letter',
    content        = <generated cover letter>,
    is_draft       = 0
)

# Save outreach email
INSERT generated_outputs (
    application_id = app_id,
    output_type    = 'hr_email',
    content        = <generated outreach email>,
    is_draft       = 0
)

# Persist the final .tex file path
UPDATE applications SET
    resume_version_path    = <path to saved .tex file>,
    current_latex_snapshot = final_latex,
    status                 = 'applied',
    applied_at             = NOW(),
    updated_at             = NOW()
```

---

## Data Flow Summary

```
users
 └─ resumes
     ├─ resume_sections          (one row per section — full blob)
     │   └─ resume_section_items (one row per company/project — atomic LaTeX)
     │         [ChromaDB: sec_item_{id}]
     └─ resume_skills ──► skills ◄── job_skills
                                          │
jobs ◄─────────────────────────────────── ┘
 └─ applications (status: draft → applied)
     ├─ resume_edits             (sentence edits, undo tree via parent_edit_id)
     └─ generated_outputs        (drafts is_draft=1, finals is_draft=0)
```

---

## RAG vs. Agentic Roles — Clarified

**RAG** (`resume_section_items` + ChromaDB) provides scoped context. Instead of feeding the LLM your entire resume history, it retrieves only the 3–5 atomic blocks semantically closest to the JD responsibilities. This keeps prompts small and relevance high.

**Agentic AI** (`resume_edits`) provides reasoning and persistence. The agent maintains the sentence-level edit state, manages the undo/redo tree through `parent_edit_id`, accumulates `line_count_delta` across all edits to detect page overflow early, and self-corrects in Stage 5 if needed.

The two roles are cleanly separated: RAG handles what to include, the agent handles how to say it.