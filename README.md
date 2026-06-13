# Job Assistant

An AI-powered resume tailoring and job application assistant. Upload your resume, paste a job description, and the system retrieves the most relevant sections from your resume history, lets you review AI-suggested edits section by section, scores the result, and generates a cover letter, HR email, and outreach message.

---

## Features

### Resume Parsing & Storage
- Parse a `.tex` resume with an LLM — extracts structured sections (experience, projects, skills, education, achievements, coursework)
- Each section and section-item is stored in SQLite and embedded into ChromaDB for semantic retrieval
- Multiple resume versions for the same user are supported; the system automatically picks the most JD-relevant version of each section

### JD Parsing
- Extracts role, company, required skills, nice-to-have skills, and responsibilities from raw JD text
- De-duplicates JDs by content hash so parsing only runs once per unique JD
- Embeds the JD into ChromaDB for retrieval queries

### Semantic Retrieval (RAG)
- Queries ChromaDB with the JD skills + responsibilities as query texts
- Two-stage fusion reranking: CombMNZ within groups (required skills, responsibilities, nice-to-have) then weighted CombSUM across groups
- Retrieval is scoped per user — picks the best version of each section across all of that user's resume history

### Interactive Editing Session (LangGraph)
- Section-by-section AI edit suggestions with side-by-side diff view
- LLM returns **all improvable lines** per section (not just the most obvious one), giving more comprehensive suggestions per call
- **Per-change selection** — each suggested line change has its own checkbox; accept the ones you want and skip the rest rather than accepting or rejecting the whole section at once. The side-by-side preview updates live to reflect your selection
- Actions per section: **Accept** (all or selected changes), **Reject**, **Paraphrase**, **Custom instruction**, **Another suggestion**
- Full edit history per section with one-click restore to any previous version (including partial-accept snapshots)
- Back-navigation to re-review previous sections
- Multi-cycle support: after reviewing all sections the system scores the result, then offers a **Refine** pass (re-runs LLM with score feedback) or **Finish**
- LLM suggestions use `\textbf{}` only inside experience and project bullet points — never in the skills section, where bolding adds no value

### Resume Scoring
- Three dimensions: **Keyword Match** (coverage + embedding similarity), **ATS Friendliness** (LLM), **Resume Quality** (LLM)
- Per-dimension scores, reasoning, and improvement suggestions
- Missing required keywords highlighted

### Document Generation
- Generate **Cover Letter**, **HR Email**, and **Outreach Message** at any point:
  - **Quick Generate (welcome screen)** — skips editing entirely, uses JD-similarity to auto-select best sections
  - **Quick Generate (post-parse)** — after JD is parsed and sections are retrieved, generate without entering the edit loop
  - **Session Generate** — generate using the tailored resume from an active editing session
  - **Post-session Generate** — generate after finishing the editing session
- All document types are generated **in parallel** (ThreadPoolExecutor) — requesting Cover Letter + Email + Outreach together takes no longer than the slowest single generation
- **Generation cache** — results are cached in SQLite keyed by a hash of `(JD, resume, type, instruction, no_jd)`. Generating Email first and then requesting all three skips the Email LLM call entirely and only generates the remaining two

### Live Resume Layout Customisation
- Available at the end of an editing session (Refine / Finish screen)
- Adjust **section order** (↑ / ↓ per section), **font size** (10 / 11 / 12 pt), **section spacing**, and **item spacing**
- Click "Build Preview" to see the rendered PDF preview and download `.tex` / `.pdf`

### Full LaTeX Preview
- Compile the resume to PDF at any point during or after editing
- Rendered as a PNG image in the browser; falls back to syntax-highlighted LaTeX source if `pdflatex` is not installed
- **Compile cache** — identical LaTeX is never sent to `pdflatex` twice in the same session; compilation results are cached by content hash via `@st.cache_data`

### Performance & Caching

All expensive operations are cached to minimise repeated LLM and compute calls:

| Layer | Cache type | Key |
|---|---|---|
| Document generation | SQLite (persistent, cross-session) | SHA-256 of `(JD, resume, type, instruction, no_jd)` |
| Resume scoring | In-memory LRU (64 entries) | SHA-256 of `(resume_latex, jd)` |
| Sentence-transformer embeddings | `functools.lru_cache` (256 entries) | Embedding input text |
| ChromaDB retrieval | In-memory dict (process lifetime) | `(job_id, sorted resume_ids)` |
| LaTeX → PDF compilation | Streamlit `@st.cache_data` | Full LaTeX string |
| JD parsing | SQLite (via raw + parsed content hash) | Built into `JDParser.run()` |

---

## Requirements

| Tool | Purpose |
|---|---|
| Python ≥ 3.13 | Runtime |
| [uv](https://docs.astral.sh/uv/) | Package manager (replaces pip/venv) |
| MiKTeX or TeX Live | LaTeX compilation for PDF preview (optional) |

API keys required in `.env`:
- `GEMINI_API_KEY` — used for all LLM calls (parsing, editing, scoring, generation)
- `HUGGING_FACE_TOKEN` — used for the `all-mpnet-base-v2` sentence-transformer embedding model

---

## Setup

```bash
# 1. Clone
git clone <repo-url>
cd job-assistant

# 2. Copy and fill in the environment file
cp .env.example .env
# Edit .env — set GEMINI_API_KEY, HUGGING_FACE_TOKEN, BASE_DIR, etc.

# 3. Install dependencies
uv sync

# 4. Initialise the database
uv run python db/init_db.py

# 5. Run any pending migrations
uv run python db/migrate.py
```

---

## Starting the App

Open two terminals:

```bash
# Terminal 1 — FastAPI backend
uv run uvicorn app.api.main:app --reload

# Terminal 2 — Streamlit frontend
uv run streamlit run streamlit_app.py
```

The Streamlit UI is available at `http://localhost:8501`.
The API docs (Swagger) are at `http://localhost:8000/docs`.

---

## How to Use

### First time — upload your resume
1. In the sidebar, select **"Upload new .tex file"** and upload your LaTeX resume.
2. Paste the job description into the **"Job Description"** text area.
3. Click **"🔍 Parse JD & Resume"**.

### Subsequent jobs — use your saved profile
1. In the sidebar, select **"Use profile from DB"** and choose your candidate profile.
2. Paste the job description.
3. Click **"🔍 Parse JD & Resume"**.

### After parsing — choose your path
The system retrieves the most relevant resume sections and shows them with relevance scores. You now have two options:

**✨ Quick Generate tab**
- Select document types (Cover Letter, HR Email, Outreach Message)
- Click **"Generate Now"** — done in seconds, no editing required

**📝 Start Editing Session tab**
- Click **"Start Editing Session"** to enter the interactive editing loop
- Review each section one at a time; accept, reject, paraphrase, or give a custom instruction
- After all sections: view your score, optionally **Refine** (re-run with score feedback), then **Finish**
- In the **Refine / Finish** screen: preview the resume, customise the layout, and generate cover letter / email / outreach

### Welcome screen Quick Generate
If you just want to generate documents without any parsing step at all:
1. Select a saved profile in the sidebar
2. Paste the JD
3. On the welcome screen, open the **✨ Quick Generate** tab and click **"Generate Now"**

---

## Running Tests

```bash
uv run pytest tests/
```

To reset the test database:
```bash
uv run python tests/resetdb.py
```

---

## File & Folder Reference

```
job-assistant/
│
├── streamlit_app.py          # Entire Streamlit UI — sidebar, all interrupt screens,
│                             #   welcome / quick-generate, layout customiser
│
├── main.py                   # Thin entry point (imports FastAPI app)
│
├── app/
│   ├── api/
│   │   ├── main.py           # FastAPI app creation, router registration, /health
│   │   ├── deps.py           # Shared FastAPI dependencies
│   │   └── routes/
│   │       ├── edit.py       # All editing-session endpoints:
│   │       │                 #   POST /edit/start, /{tid}/resume, /{tid}/score,
│   │       │                 #   /{tid}/preview, /{tid}/generate,
│   │       │                 #   /{tid}/generate-from-items, /quick-generate
│   │       ├── resume.py     # Resume upload, list, user profiles  (GET /resume/users)
│   │       ├── jd.py         # Standalone JD parse endpoint
│   │       ├── generate.py   # Standalone generation endpoint
│   │       └── score.py      # Standalone score endpoint
│   │
│   ├── graph/
│   │   ├── edit_graph.py     # LangGraph StateGraph for the editing session:
│   │   │                     #   node_parse_and_retrieve → item_selection interrupt
│   │   │                     #   → node_build_edit_state → node_suggest
│   │   │                     #   → section_review interrupt (per section, per change)
│   │   │                     #   → refine_or_finish interrupt → done
│   │   ├── generation_graph.py  # LangGraph for batch document generation;
│   │   │                     #   runs all requested types in parallel via
│   │   │                     #   ThreadPoolExecutor, no human-in-the-loop
│   │   └── state.py          # EditGraphState + GenerationGraphState TypedDicts
│   │
│   ├── agents/
│   │   ├── edit_agent.py     # EditAgent — holds ResumeEditCycle, calls LLM for
│   │   │                     #   suggestions (all improvable lines per section) and
│   │   │                     #   section edits, manages history tree
│   │   ├── score_agent.py    # score() — keyword coverage + semantic + LLM scoring;
│   │   │                     #   lru_cache on embeddings, in-memory score result cache
│   │   ├── generator_agent.py# generate() — cover letter / email / outreach via LLM;
│   │   │                     #   checks generation cache before every LLM call
│   │   └── orchestrator.py   # High-level orchestration helpers
│   │
│   ├── core/
│   │   ├── pipeline.py       # Thin facade over all core modules — the single
│   │   │                     #   import used by graph nodes and API routes;
│   │   │                     #   includes in-memory retrieval cache
│   │   ├── gen_cache.py      # SQLite-backed generation result cache;
│   │   │                     #   make_key() + get() + set() used by generator_agent
│   │   ├── jd_parser.py      # JDParser — LLM extracts structured JD, saves to
│   │   │                     #   SQLite + embeds to ChromaDB; hash-based dedup
│   │   ├── resume_parser.py  # ResumeParser — LLM extracts structured resume,
│   │   │                     #   saves sections/items to SQLite + ChromaDB
│   │   ├── retriever.py      # ResumeRetriever — ChromaDB queries, two-stage
│   │   │                     #   CombMNZ/CombSUM reranking, rank_and_filter
│   │   └── resume_builder.py # ResumeBuilder — assembles final LaTeX from sections;
│   │                         #   accepts section_order, font_size, spacing overrides
│   │
│   ├── models/
│   │   ├── jd.py             # ParsedJD pydantic model
│   │   ├── resume.py         # ParsedResume, PersonalInfo, ResumeSection, AtomicItem
│   │   ├── edit.py           # ResumeEditState, ResumeEditCycle, SectionEditState,
│   │   │                     #   ItemEditState
│   │   ├── scoring.py        # ResumeScore, DimensionScore, LLMScoreOutput
│   │   ├── generation.py     # Generation request/response models
│   │   ├── resume_builder.py # ResumeBuilderInput
│   │   └── retriever.py      # Retriever result models
│   │
│   └── utils/
│       ├── llm.py            # LLMClient wrapper (Gemini / OpenAI), Prompt dataclass
│       ├── logger.py         # get_logger() — structured logging setup
│       ├── sqlite_handler.py # SQLHandler — fetch_one, fetch_table_where,
│       │                     #   execute_raw wrappers over SQLAlchemy
│       └── __init__.py       # Re-exports SQLHandler, get_logger
│
├── config/
│   ├── config.ini            # Config template — values injected from .env
│   └── config.py             # get_config_dict() — loads config.ini + .env,
│                             #   returns nested dict (cached after first call)
│
├── db/
│   ├── schema.sql            # Full database schema (reference)
│   ├── init_db.py            # Creates all tables from schema.sql
│   ├── migrate.py            # Runs numbered migration files in order
│   └── migrations/           # SQL migration files (001 … 007)
│
└── tests/
    ├── conftest.py           # Pytest fixtures (DB setup, mock clients)
    ├── resetdb.py            # Wipes and re-initialises the test database
    ├── test_jd_parser.py     # JDParser unit tests
    ├── test_resume_parser.py # ResumeParser unit tests
    ├── test_retriever.py     # ResumeRetriever unit tests
    ├── test_edit_agent.py    # EditAgent unit tests
    ├── test_score_agent.py   # ScoreAgent unit tests
    └── test_generator_agent.py  # GeneratorAgent unit tests
```
