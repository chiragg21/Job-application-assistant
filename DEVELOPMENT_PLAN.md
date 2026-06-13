# Job Assistant — Development Plan

## Current Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Backend | FastAPI + LangGraph |
| LLM | Gemini 2.0 Flash (via infrakit LLMClient) |
| Embeddings | `all-mpnet-base-v2` (HuggingFace, local) |
| Vector store | ChromaDB (persistent, file-based) |
| Database | SQLite (via SQLHandler / SQLAlchemy) |
| Resume format | LaTeX only (one specific template) |

---

## Known Issues (pre-development audit)

### Performance
- Streamlit rerenders the entire page on every interaction — no partial updates
- `_resume_suggestions()` sends all sections + full JD + JSON schema in one large prompt (~3–5K tokens), blocking for 4–10s with no progress feedback
- `_rebuild_agent(state)` deserializes the full `ResumeEditCycle` on every graph node — adds latency per section interaction
- `all-mpnet-base-v2` (~420MB) loads fresh on every server restart — first request is slow
- `pdflatex` runs twice per compile (cross-reference pass); already cached but cold calls take 2–6s
- `MemorySaver` (LangGraph checkpointer) is in-memory only — grows unbounded, no TTL, leaks over long uptime

### UX Freedom
- Sessions stored in `MemorySaver` — server restart or browser refresh loses everything; no session recovery
- Graph goes to `END` after finish — no way to reopen a completed session and tweak a section without restarting the full pipeline
- `selected_items` locked after item-selection interrupt — can't add/remove sections mid-review
- No cross-session version history — tailored resume exists only in live graph state, gone when session ends
- No mid-session export — download only available at the refine/finish screen
- No full-resume diff — only per-section diffs, no original-vs-final whole-resume view
- `user_id = 1` hardcoded in defaults — no real multi-user isolation
- No global custom instruction across all sections

### Resume Parser
- `extract_personal_info` hardcoded to one LaTeX template (FontAwesome header: `\Huge\textbf`, `\faPhone`, `\faLinkedin`, etc.)
- `_parse_atomic_items` splits blocks by `\textbf{}\hfill` — breaks on any other LaTeX template (Awesome-CV, moderncv, `\subsection`, `\cventry`, etc.)
- Only `.tex` files accepted — no PDF, no DOCX, no plain text

### LLM Quality
- Edit prompts instruct "keep sentence length roughly similar" and "wording alignment only" — LLM acts as copy editor, not resume writer
- No impact-first restructuring, no XYZ formula, no few-shot examples
- JD context sent as "use only for wording alignment" — underutilises the JD

### API / Key Management
- `_resume_suggestions()` output not cached — same resume + JD reruns the LLM on every new session
- No fallback when Gemini rate-limits — hard fail instead of queue/retry
- Only Gemini keys supported — no second provider as overflow

---

## UI Decision — Replace Streamlit with React + Next.js

**Keep the FastAPI backend 100% unchanged.** Only the frontend layer is replaced.

| Problem | Streamlit | React / Next.js |
|---|---|---|
| Page rerenders on every click | Always, full page | No, only changed components |
| Streaming LLM tokens | Not possible without hacks | Native (SSE / WebSockets) |
| Browser extension sidebar | Impossible (server-side Python rendering) | Same React components reused directly |
| PWA / installable on mobile | No | Yes (one config + manifest) |
| Perceived speed | Full server round-trip per interaction | Local state + async fetches |
| Offline support | None | Cache last result in localStorage |

---

## API Key Strategy

### Free tier limits (per key)

| Provider | Requests/day | Req/min | Notes |
|---|---|---|---|
| Gemini 2.0 Flash | 1,500 | 15 | Primary |
| Gemini 2.0 Flash (key 2) | 1,500 | 15 | Rotation / overflow |
| **Groq** (Llama 3.3 70B) | **14,400** | **30** | Fallback — free, very fast |
| Mistral (Mistral Small) | Free tier | — | Second fallback |

With 2 Gemini keys + Groq: effectively **~17,400 req/day** across providers.

### Typical session cost (LLM calls)

| Operation | Calls | Cached after? |
|---|---|---|
| JD parsing | 1 | Yes — raw hash dedup |
| `_resume_suggestions()` | 1 | Not yet — needs fix |
| Per-section paraphrase / another | 1 each | No (user-triggered) |
| Resume scoring | 2 | Yes — LRU by resume+JD hash |
| Cover letter + email + outreach | 3 | Yes — SQLite by content hash |
| **Typical first session** | **~5–8** | — |
| **Repeat same JD** | **1–2** | — |

375+ full uncached sessions/day is feasible on 2 keys alone.

### Key rotation and fallback logic

- `LLMClient` already reads `gemini_keys` as a list — key rotation is partly built
- Add Groq as a named provider alongside Gemini
- Fallback order: Gemini key 1 → Gemini key 2 → Groq → Mistral
- Retry with exponential backoff on rate-limit (429) instead of hard fail
- Queue requests when all providers are at RPM limit rather than dropping them

### Token reduction (40–60% savings)

1. **Cache `_resume_suggestions()` output** — key: `sha256(resume_latex + jd_hash)`. Skip LLM entirely on repeat.
2. **Compress JD context in prompts** — currently sends full skill lists + all responsibilities. Cap at top 10 required skills + top 5 responsibilities.
3. **Move JSON schema to system prompt** — Gemini caches system prompts across calls; repeating the schema in every user prompt wastes tokens.
4. **Model tiering** — use `gemini-2.0-flash` for all edit/score calls (fast, cheap). Reserve a higher-quality model only for final document generation if quality is noticeably better.

### Model assignment

| Task | Model | Reason |
|---|---|---|
| JD parsing | `gemini-2.0-flash` | Structured extraction, fast enough |
| `_resume_suggestions()` | `gemini-2.0-flash` | Most frequent call, speed matters |
| Resume scoring (ATS + quality) | `gemini-2.0-flash` | LLM scoring, cached aggressively |
| Cover letter / email / outreach | `gemini-2.0-flash` | Generation cache handles repeats |
| JD URL text cleanup | `gemini-2.0-flash` or Groq | Lightweight extraction |

---

## All Improvements — Master List

### Category 1 — Input & Convenience

| # | Improvement | Effort |
|---|---|---|
| 1 | **JD from URL** — paste a LinkedIn/Indeed/Glassdoor/Naukri URL; app extracts JD via Jina Reader (`r.jina.ai/{url}`, free, no key, handles JS-rendered pages) | Low |
| 2 | **Browser extension** — sidebar panel injected into LinkedIn/job sites. Auto-detects job post, extracts JD, calls quick-generate, shows resume + cover letter in panel without leaving the page. Works on Android via Firefox for Android extension. Built from same React components as main frontend. | High |
| 3 | **Email-to-JD** — forward recruiter emails to a dedicated address (Mailgun free tier); app auto-extracts JD and adds to history | Medium |
| 4 | **Resume format flexibility** — add LLM-based fallback parser so any LaTeX template, PDF, or plain text is accepted. Current regex-based parser becomes fast path for the known template only. | Medium |

### Category 2 — LLM Quality

| # | Improvement | Effort |
|---|---|---|
| 5 | **Stronger edit prompts** — replace "keep sentence length similar / wording alignment only" with "restructure bullets to lead with impact metric, use X achieved Y by Z formula". Add few-shot weak→strong rewrite examples in system prompt. | Low |
| 6 | **Streaming LLM responses** — stream tokens back via FastAPI `StreamingResponse` + SSE. User sees suggestions appearing in real time instead of blocking spinner. Requires React frontend (Phase 1a). | Medium |
| 7 | **Per-section parallel suggestions** — instead of one giant prompt for all sections, fan out to one LLM call per section concurrently. Sections with no changes return fast; user sees first result almost immediately. | Medium |

### Category 3 — Performance

| # | Improvement | Effort |
|---|---|---|
| 8 | **React + Next.js frontend** — eliminates full-page rerenders, enables streaming, enables PWA, enables extension sidebar reuse. FastAPI backend unchanged. | High |
| 9 | **Multi-provider key rotation with fallback** — Gemini key 1 → key 2 → Groq → Mistral. Retry on 429, queue at RPM limit. | Medium |
| 10 | **Cache `_resume_suggestions()` output** — currently uncached. Key: `sha256(resume_latex + jd_hash)`. Most repeat sessions skip the biggest LLM call entirely. | Low |
| 11 | **Compress prompts** — cap JD context, move schema to system prompt, strip redundant boilerplate. Cuts per-call token count 40–60%. | Low |
| 12 | **Persistent LangGraph checkpointer** — replace `MemorySaver` with SQLite-backed or Redis-backed checkpointer. Sessions survive server restarts. Can resume from any device. `MemorySaver` grows unbounded; this also fixes the memory leak. | Medium |
| 13 | **Pre-warm embedding model** on FastAPI startup — load `all-mpnet-base-v2` at boot instead of lazy-loading on first request. Eliminates cold-start latency. | Low |
| 14 | **Single pdflatex pass for preview** — resumes have no TOC or cross-references; the second pass adds 1–3s for no benefit. | Low |

### Category 4 — UX Freedom & Production Grade

| # | Improvement | Effort |
|---|---|---|
| 15 | **Persist completed sessions** — save final `ResumeEditState` + generated docs to DB linked to `(user_id, job_id)`. Any past application is revisitable. | Medium |
| 16 | **Re-open completed sessions** — "Edit again" button that reloads a finished session back into the editing loop for tweaking individual sections. | Medium |
| 17 | **Mid-session export** — download button (TeX + PDF) available at every section review step, not only at the finish screen. | Low |
| 18 | **Full-resume diff view** — original vs. final side-by-side for the whole resume, not just per section. | Low |
| 19 | **Global custom instruction** — one instruction applied to all sections ("write more aggressively", "target senior roles") set before the edit loop begins. | Low |
| 20 | **User authentication** — Google OAuth or magic-link login. Removes hardcoded `user_id = 1`. Required for any public hosting. | Medium |
| 21 | **MemorySaver TTL / eviction** — add TTL-based cleanup even before replacing with a persistent checkpointer. Stops the memory leak on long-running servers. | Low |

### Category 5 — Job Discovery & Intelligence

| # | Improvement | Effort |
|---|---|---|
| 22 | **Application tracker / Kanban** — DB table per application: status (Applied → Screen → Technical → Offer/Rejected), which resume version + cover letter was sent, notes, follow-up date. | Medium |
| 23 | **Job feed / recommendations** — given stored skills + past JDs, surface matching jobs from LinkedIn/Indeed/Naukri ranked by resume fit score. | High |
| 24 | **Skill gap analysis** — across all processed JDs, show which required skills you consistently lack. "Learn these 3 things = qualify for 80% more roles in your history." | Medium |
| 25 | **LinkedIn profile generator** — same JD-tailoring logic applied to LinkedIn About / Headline / Experience bullets. | Low |
| 26 | **Interview prep** — post-session: generate 5–10 likely interview questions + talking points grounded in your actual resume experience vs the JD. | Low |
| 27 | **Salary range lookup** — display typical compensation for the role/location from public sources (Levels.fyi, Glassdoor scrape) alongside the JD. | Medium |

### Category 6 — API Key Management & Personalisation

| # | Improvement | Effort |
|---|---|---|
| 28 | **API key usage dashboard** — expose `llm.status()` from infrakit via a new endpoint. Show per-key: current RPM, daily tokens used/remaining, total requests/errors, last N request latency. Already fully implemented in the library — just needs an endpoint + UI panel. | Low |
| 29 | **User-managed keys (single-user)** — settings panel to add new API keys, set per-key quota limits (`rpm_limit`, `daily_token_limit`, model scope) via `llm.set_quota()`. Persists to `.env` and re-inits the `LLMClient` singleton. Supports Gemini + Groq. | Medium |
| 29b | **Per-user encrypted keys (multi-user, post-auth)** — after Sprint 5 auth: keys stored encrypted per `user_id` in DB, decrypted at session start. Requires Sprint 5 (auth) as prerequisite. | Medium |
| 30 | **Adaptive writing style** (data logging) — add `edit_feedback` DB table, log every accept/reject/custom-instruction decision from `apply_edit` node with the original and accepted text. This is the data collection layer. | Low |
| 31 | **Adaptive writing style** (inference + injection) — `StyleProfileAgent` runs after every 3 sessions: sends feedback history to LLM, extracts structured style profile (`{tone, bullet_length, complexity, preferred_verbs, avoid_phrases, ...}`), stores in `user_style_profiles` table. Profile is injected as a high-priority block in all edit system prompts. | Medium |
| 32 | **Style profile UI** — settings panel showing the inferred style profile, editable field by field, on/off toggle per session. User can manually override any preference or reset to default. | Low |

---

## Implementation Order

### Ordering Principles

1. **Backend before frontend** — stabilise the backend before migrating the UI. Streamlit keeps working throughout Sprints 0–2.
2. **Performance and caching before new features** — don't add load to a slow, fragile system.
3. **Plumbing before features that need it** — persistent checkpointer must exist before session history; auth must exist before application tracker.
4. **React before anything that depends on React** — streaming, extension, PWA all need the frontend migration done first.
5. **Auth before public hosting** — never expose the app without login.
6. **Each sprint leaves the app in a fully working state** — no half-broken intermediate states.

---

### Sprint 0 — Backend Quick Wins
*Time estimate: 1–2 days. Zero migration risk. Streamlit still runs unchanged.*

These are isolated, low-risk edits entirely within existing files. Do these first because they improve the app immediately and are wasted if done later alongside bigger changes.

| Order | Task | Item | File(s) touched | Why first |
|---|---|---|---|---|
| 0.1 | Remove second `pdflatex` pass | #14 | `streamlit_app.py` | 5-min change, saves 1–3s on every cold compile |
| 0.2 | Pre-warm embedding model at startup | #13 | `app/api/main.py` | 15-min change, eliminates cold-start delay on first request |
| 0.3 | MemorySaver TTL cleanup | #21 | `app/graph/edit_graph.py` | Stops memory leak immediately; interim fix before persistent checkpointer |
| 0.4 | Move JSON schema to system prompt | #11 | `app/agents/edit_agent.py`, `generator_agent.py`, `score_agent.py` | Token savings on every LLM call, zero quality impact |
| 0.5 | Compress JD context in prompts | #11 | `app/agents/edit_agent.py` | Cap required skills to 10, responsibilities to 5; cuts prompt tokens 40% |
| 0.6 | Stronger edit prompts | #5 | `app/agents/edit_agent.py` | Highest ROI change — makes LLM rewrites actually useful without any infrastructure work |
| 0.7 | Cache `_resume_suggestions()` output | #10 | `app/core/gen_cache.py`, `app/agents/edit_agent.py` | Eliminates the biggest LLM call on repeat sessions; key: `sha256(resume_latex + jd_hash)` |

**Dependency note:** 0.4 and 0.5 are prerequisite to 0.6 (clean up prompts before rewriting them). Everything else in Sprint 0 is independent.

---

### Sprint 1 — API Reliability + Key Management + New Endpoints
*Time estimate: 3–4 days. Still on Streamlit.*

Harden the API layer, expose key usage, and add URL extraction before touching the frontend.

| Order | Task | Item | File(s) touched | Why here |
|---|---|---|---|---|
| 1.1 | API key usage dashboard endpoint | #28 | `app/api/routes/keys.py` (new) | Zero-effort win — library already has it. Do first so you can monitor key health throughout the rest of Sprint 1. |
| 1.2 | Register Groq API key + wire as fallback provider | #9 | `app/utils/llm.py`, `config/config.ini` | Key rotation before adding more LLM-calling features; protects Sprint 0 gains from rate-limit failures |
| 1.3 | Retry with exponential backoff on 429 | #9 | `app/utils/llm.py` | Companion to 1.2 — rotation without retry still drops requests |
| 1.4 | User-managed keys settings endpoint | #29 | `app/api/routes/keys.py`, `app/utils/llm.py` | Add key + set quota via API; persist to `.env` + re-init LLMClient. Do after 1.1–1.3 so the dashboard already shows the new key. |
| 1.5 | JD from URL endpoint | #1 | `app/core/jd_extractor.py` (new), `app/api/routes/jd.py` | Self-contained new feature, no dependencies; immediate convenience before the UI migration |
| 1.6 | Global custom instruction field | #19 | `app/api/routes/edit.py`, `app/graph/edit_graph.py`, `app/graph/state.py` | Small API change; available in both Streamlit and React UI |
| 1.7 | Mid-session export endpoint | #17 | `app/api/routes/edit.py` | Expose TeX/PDF download at every stage; UI surfaces it in Streamlit now and React later |

---

### Sprint 2 — Persistence Layer
*Time estimate: 3–4 days. Still on Streamlit. This sprint is the plumbing everything else depends on.*

Nothing in Sprint 4 or 5 is possible without this. Do it before the React migration so the new frontend is built on top of a persistent backend from day one.

| Order | Task | Item | File(s) touched | Why here |
|---|---|---|---|---|
| 2.1 | Replace `MemorySaver` with SQLite-backed LangGraph checkpointer | #12 | `app/graph/edit_graph.py`, `db/migrations/` | Foundational — sessions survive restarts. Must be done before 2.2 can work. |
| 2.2 | Persist completed sessions to DB | #15 | `app/graph/edit_graph.py`, `db/schema.sql`, `db/migrations/` | New tables: `sessions`, `session_documents`. Requires 2.1. |
| 2.3 | Full-resume diff endpoint | #18 | `app/api/routes/edit.py` | Pure backend — computes original vs final LaTeX diff; frontend can surface it in both Streamlit and React |
| 2.4 | Resume format flexibility — LLM fallback parser | #4 | `app/core/resume_parser.py` | Do before React migration so the new UI can accept any file type from day one; regex parser stays as fast path |

**DB migrations required in this sprint:**
- `008_sessions.sql` — `sessions(id, user_id, job_id, thread_id, status, created_at, finished_at)`
- `009_session_documents.sql` — `session_documents(id, session_id, type, content, created_at)`

---

### Sprint 3 — React + Next.js Frontend Migration
*Time estimate: 1–2 weeks. This is the big lift. Streamlit is removed at the end of this sprint.*

Do this after Sprints 0–2 so the backend is fast, reliable, and persistent before the new UI is built on top of it.

| Order | Task | Notes |
|---|---|---|
| 3.1 | Scaffold Next.js app in `frontend/` | TypeScript, Tailwind CSS, shadcn/ui components. Configure API base URL via env var. |
| 3.2 | API client layer | Single `lib/api.ts` module — mirrors all FastAPI endpoints. Type-safe with Zod schemas matching Pydantic models. |
| 3.3 | Setup / welcome screen | Profile selector, JD text area + URL input (1.3), resume upload, quick-generate tab |
| 3.4 | Section review screen | Most complex screen — diff view, per-change checkboxes, history timeline, back navigation. Build with local React state, no full-page reloads. |
| 3.5 | Score + refine screen | Score bars, dimension breakdown, refine/finish buttons |
| 3.6 | Finish screen | Layout customiser, final download, document generation panel |
| 3.7 | Session history panel | Lists past sessions from DB (Sprint 2.2), links to re-open or download |
| 3.8 | Application tracker UI | Kanban board for application status — builds on session history |
| 3.9 | Remove `streamlit_app.py` | Only after all screens are verified in React |

**Dependency note:** 3.7 requires Sprint 2.2. All other screens can be built in parallel once 3.1–3.2 are done.

---

### Sprint 4 — Data Integrity + Template Engine + Quick Frontend Wins
*Time estimate: ~10 days. Backend-first. Leaves the app with clean content storage and a working template system before any new UI is built on top.*

**Phase A — Resume Data Integrity (backend)**

The edit session finish currently leaves `resume_section_items` unchanged. ChromaDB is never re-synced. There is no version history. This phase fixes the data layer before anything else touches it.

| Order | Task | Files | Notes |
|---|---|---|---|
| 4.1 | DB migration 009 — item versioning | `db/migrations/009_resume_versioning.sql` | Add `is_latest INTEGER DEFAULT 1` and `parent_item_id INTEGER` (self-FK) to `resume_section_items`. Add `input_resume_id`, `output_resume_id` to `edit_sessions`. |
| 4.2 | Finish endpoint writes content back to DB | `app/api/routes/edit.py`, `app/utils/sqlite_handler.py` | `POST /edit/{thread_id}/finish` iterates `state.modified_items`, inserts new rows with `is_latest=1`, marks replaced rows `is_latest=0` with `parent_item_id` pointing back. |
| 4.3 | ChromaDB sync after finish | `app/core/resume_parser.py` | Delete old ChromaDB entries for modified sections, re-embed only changed items. |
| 4.4 | Version history endpoint | `app/api/routes/resume.py` | `GET /resume/{id}/history` — returns all item rows grouped by section, ordered by edit date, for the "what changed" view. |

**Phase B — Template Engine & Section Flexibility (backend)**

Decouples resume content from presentation. Any resume (parsed from any format) can be rendered with any template.

| Order | Task | Files | Notes |
|---|---|---|---|
| 4.5 | Jinja2 template engine | `app/core/resume_builder.py` | Replace hardcoded LaTeX string-building with `jinja2.Environment`. `build_resume(sections, layout_config) → str`. |
| 4.6 | Classic template | `templates/resume/classic.tex.j2` | Port the existing hardcoded template to Jinja2 — same visual output, validates the engine. |
| 4.7 | ATS Minimal template | `templates/resume/ats_minimal.tex.j2` | No icons, plain `\section{}` headers, standard bullets. Passes ATS scanners that reject FontAwesome/complex LaTeX. |
| 4.8 | DB migration 010 — layouts + section type | `db/migrations/010_resume_layouts.sql` | New `resume_layouts` table: `(resume_id, template_id, section_order JSON, font_size, margin, font_family)`. Add `section_type TEXT DEFAULT 'atomic'` to `resume_sections`. |
| 4.9 | Default layout row on parse | `app/core/resume_parser.py` | After `add_to_sql_and_chroma`, insert a default `resume_layouts` row (`template='classic'`, section_order from env, font_size=11). |
| 4.10 | Section flexibility — remove hardcoded constraints | `app/core/resume_parser.py`, `app/utils/sqlite_handler.py` | Any section name is valid. Remove ATOMIC_SECTIONS/FLAT_SECTIONS env-only constraint; use `section_type` DB column instead. |
| 4.11 | Section CRUD + layout endpoints | `app/api/routes/resume.py` | `POST /resume/{id}/sections` (add), `DELETE /resume/{id}/sections/{name}`, `PATCH /resume/{id}/sections/{name}` (rename). `PATCH /resume/{id}/layout` (save template + order + font). |
| 4.12 | Preview endpoint | `app/api/routes/resume.py` | `GET /resume/{id}/preview?template=X&font_size=Y` → PDF bytes (`Content-Disposition: inline`). Compiles without saving layout changes. |
| 4.13 | Parse preview + commit endpoints | `app/api/routes/resume.py` | `POST /resume/parse-preview` — dry-run parse, returns structured JSON, nothing saved. `POST /resume/commit` — saves structured JSON to DB + ChromaDB + default layout row. |

**Phase C — Quick Frontend Wins (no new backend needed)**

| Order | Task | Files | Notes |
|---|---|---|---|
| 4.14 | Re-open completed sessions | `frontend/src/app/sessions/page.tsx` | "Edit again" button per session row — calls `POST /edit/start` with `thread_id` to rehydrate graph state from `SqliteSaver`. |
| 4.15 | Full-resume diff view | `frontend/src/components/session/ResumeDiff.tsx` | Side-by-side original vs final view. Wires to existing `GET /edit/{thread_id}/diff` endpoint. Accessible from finish screen and session history. |

---

### Sprint 5 — Streaming + Resume Management UI
*Time estimate: ~8 days. Requires Sprint 4 complete. React-dependent for streaming; template backend (Sprint 4B) required for the resume UI.*

| Order | Task | Files | Notes |
|---|---|---|---|
| 5.1 | Streaming LLM responses — backend | `app/api/routes/edit.py`, `app/agents/edit_agent.py` | FastAPI `StreamingResponse` + SSE for `_resume_suggestions()`. Yields token chunks as `data: {...}\n\n`. |
| 5.2 | Streaming — frontend consumer | `frontend/src/app/session/[threadId]/page.tsx`, `frontend/src/components/session/SectionReview.tsx` | `EventSource` client. Tokens stream into the suggestion text area in real time instead of blocking spinner. |
| 5.3 | Per-section parallel suggestions | `app/agents/edit_agent.py`, `app/api/routes/edit.py` | `ThreadPoolExecutor` fan-out — one LLM call per section concurrently. Frontend receives results as they arrive via SSE. Requires 5.1–5.2. |
| 5.4 | Upload → verify flow redesign | `frontend/src/app/resume/new/page.tsx` | Two-tab entry: "Upload file" (calls parse-preview, shows result in form) and "Enter manually" (empty form). Replaces direct parse-and-commit. |
| 5.5 | Parse verification + manual entry form | `frontend/src/components/resume/ResumeEditForm.tsx` | Editable personal info row + per-section accordion. Atomic: add/remove/reorder items, edit bullet list. Flat: textarea. "Confirm & Save" → commit endpoint. |
| 5.6 | Template picker | `frontend/src/components/resume/TemplatePicker.tsx` | Thumbnail grid of templates. Click → calls preview endpoint → live PDF in adjacent iframe. |
| 5.7 | Layout controls | `frontend/src/components/resume/LayoutControls.tsx` | Section drag-to-reorder (`@dnd-kit/core`), font size dropdown, margin selector. "Save Layout" → `PATCH /resume/{id}/layout`. |
| 5.8 | Resume detail page | `frontend/src/app/resume/[resumeId]/page.tsx` | Version history timeline, template + layout panel, PDF preview, download button, "Start edit session" CTA. |

---

### Sprint 6 — Auth + Intelligence Features
*Time estimate: ~8 days. Auth must come first — application tracker and skill gap need real user IDs.*

| Order | Task | Files | Notes |
|---|---|---|---|
| 6.1 | Google OAuth backend | `app/api/routes/auth.py` (new) | `authlib` + FastAPI. `GET /auth/login` → Google redirect. `GET /auth/callback` → JWT issued as httpOnly cookie. `POST /auth/logout`. |
| 6.2 | DB migration 011 — auth | `db/migrations/011_auth.sql` | Add `oauth_sub`, `email`, `email_verified`, `avatar_url` to `users` table. |
| 6.3 | JWT middleware | `app/api/middleware.py` (new) | `get_current_user` FastAPI dependency. Protected routes require valid JWT. Dev mode allows `user_id=1` fallback. |
| 6.4 | Frontend auth flow | `frontend/src/app/login/page.tsx`, `frontend/src/lib/auth.ts` | Login page with Google OAuth button. Token stored in httpOnly cookie. Protected route wrapper HOC. |
| 6.5 | Application tracker backend | `app/api/routes/applications.py` (new), `db/migrations/012_applications.sql` | `applications(id, user_id, job_id, resume_id, status, notes, applied_at, follow_up_at)`. Statuses: Applied → Phone Screen → Technical → Offer / Rejected. Full CRUD. |
| 6.6 | Kanban UI | `frontend/src/app/tracker/page.tsx` | Drag cards between status columns. Each card shows: role, company, resume version used, generated docs, follow-up date. |
| 6.7 | Skill gap analysis | `app/api/routes/skills.py` (new) | `GET /skills/gaps` — cross-references required skills across all user JDs against the user's skills section. Returns ranked gaps with frequency count. |
| 6.8 | Interview prep generation | `app/api/routes/generate.py` | `POST /generate/interview-prep` — takes `resume_id` + `job_id`, generates 5–10 questions + talking points. Cached by `sha256(resume_id + job_id)`. |
| 6.9 | LinkedIn profile generator | `app/api/routes/generate.py` | `POST /generate/linkedin` — takes `resume_id` + target role, generates About / Headline / Experience bullets. |

---

### Sprint 7 — Browser Extension
*Time estimate: 1 week. Requires Sprint 3 (React components) + Sprint 1.5 (JD URL endpoint) + Sprint 6 (auth for token storage).*

| Order | Task | Notes |
|---|---|---|
| 7.1 | Extension scaffold | `manifest.json` (Manifest V3 for Chrome, compatible with Firefox). Permissions: `activeTab`, `storage`, host permissions for job sites. |
| 7.2 | Content script — JD extraction | `content_script.js`: `MutationObserver` for SPA navigation, per-site DOM selectors for LinkedIn / Indeed / Naukri / Glassdoor. Falls back to Jina Reader if DOM extraction fails. |
| 7.3 | Sidebar panel | Shadow DOM injection (bypasses LinkedIn CSP). Bundles the quick-generate React component from Sprint 3 directly into the extension. |
| 7.4 | Background service worker | Manages API calls to hosted backend. Handles auth token storage from Sprint 6. |
| 7.5 | Firefox compatibility pass | Manifest V2 fallback for Firefox for Android. Test on Kiwi Browser (Android Chrome extension support). |

---

### Sprint 8 — Hosting + Mobile
*Time estimate: 3–5 days. Requires Sprint 6 (auth).*

Never host without auth. Auth is the gate.

| Order | Task | Item | Notes |
|---|---|---|---|
| 8.1 | Deploy FastAPI + Next.js to Railway or Render | — | Free tier. Env vars for all API keys — never in code. |
| 8.2 | Next.js PWA config | — | `next-pwa` package + `manifest.json`. "Add to home screen" on Android and iOS Safari. |
| 8.3 | Android TWA (Trusted Web Activity) | — | Wraps the PWA as a Play Store APK for free using Bubblewrap CLI. No new code required. |
| 8.4 | Salary range lookup | #27 | Scrape Levels.fyi / Glassdoor for role+location; display alongside parsed JD. |
| 8.5 | Email-to-JD via Mailgun | #3 | Mailgun free tier (1,000 emails/month). Inbound webhook → JD extraction → stored in DB. |
| 8.6 | Job feed / recommendations | #23 | Highest effort feature. Scrape LinkedIn/Indeed by skill keywords; rank results by ChromaDB similarity against user's resume. |

---

### Dependency Graph (visual summary)

```
Sprint 0 (backend quick wins)
    └── Sprint 1 (API reliability + JD URL)
            └── Sprint 2 (persistence layer)
                    └── Sprint 3 (React migration)
                            └── Sprint 4 (data integrity + template engine + quick frontend wins)
                                    └── Sprint 5 (streaming + resume management UI)
                                            └── Sprint 6 (auth + intelligence)
                                                    ├── Sprint 7 (browser extension)
                                                    └── Sprint 8 (hosting + mobile)
```

Each sprint is a vertical slice — working software at the end of every sprint.

---

## Phased Roadmap

*Phases map to sprint groups. Each phase ends with a stable, usable version of the app.*

### Phase 1 — Foundation (Sprints 0–2)
*Goal: fast, reliable, session-persistent app with a hardened backend. Streamlit still running throughout.*

| Sprint | Task | Item |
|---|---|---|
| 0 | Single pdflatex pass | #14 |
| 0 | Pre-warm embedding model at startup | #13 |
| 0 | MemorySaver TTL fix | #21 |
| 0 | Compress prompts + schema to system prompt | #11 |
| 0 | Stronger edit prompts (XYZ formula, few-shot examples) | #5 |
| 0 | Cache `_resume_suggestions()` output | #10 |
| 1 | Multi-provider key rotation + Groq fallback + retry | #9 |
| 1 | JD from URL (Jina Reader + new API endpoint) | #1 |
| 1 | Global custom instruction field | #19 |
| 1 | Mid-session export endpoint | #17 |
| 2 | Persistent LangGraph checkpointer (SQLite-backed) | #12 |
| 2 | Persist completed sessions to DB | #15 |
| 2 | Full-resume diff endpoint | #18 |
| 2 | Resume format flexibility (LLM fallback parser, PDF/DOCX) | #4 |

**Phase 1 exit criteria:** sessions survive restarts, suggestions are fast and useful, any resume format accepted, Groq absorbs rate-limit overflow.

---

### Phase 2 — Data Integrity + Templates + Streaming + Auth (Sprints 3–6)
*Goal: React app live, data layer clean, multiple templates, streaming UX, auth-gated, intelligence features enabled.*

| Sprint | Task | Item |
|---|---|---|
| 3 | React + Next.js frontend — all screens migrated | #8 |
| 3 | Settings page — API key management | #28/#29 |
| 4 | Resume data integrity — edit writes back to DB, ChromaDB sync, versioning | #15/#12 |
| 4 | Jinja2 template engine + Classic + ATS Minimal templates | — |
| 4 | Section flexibility — custom sections, layout config | — |
| 4 | Parse preview + commit endpoints | #4 |
| 4 | Re-open completed sessions (quick win) | #16 |
| 4 | Full-resume diff view in UI (quick win) | #18 |
| 5 | Streaming LLM responses via SSE | #6 |
| 5 | Per-section parallel LLM suggestions | #7 |
| 5 | Resume management UI — upload/verify/manual entry, template picker, layout controls | — |
| 6 | User authentication (Google OAuth) | #20 |
| 6 | Application tracker / Kanban | #22 |
| 6 | Skill gap analysis | #24 |
| 6 | Interview prep generation | #26 |
| 6 | LinkedIn profile section generator | #25 |

**Phase 2 exit criteria:** React app fully replaces Streamlit, data is version-tracked, any resume format + any template, streaming responses, auth-gated, application pipeline tracked end-to-end.

---

### Phase 3 — Extension + Mobile + Hosting (Sprints 7–8)
*Goal: accessible on all devices, installable, browser extension working.*

| Sprint | Task | Item |
|---|---|---|
| 7 | Browser extension (Chrome + Firefox) — content script + sidebar | #2 |
| 7 | Firefox for Android compatibility pass | #2 |
| 8 | Deploy to Railway / Render (env vars for keys) | — |
| 8 | Next.js PWA config (Android + iOS Safari installable) | — |
| 8 | Android TWA — Play Store APK wrapping the PWA | — |
| 8 | Salary range lookup | #27 |
| 8 | Email-to-JD via Mailgun | #3 |
| 8 | Job feed / recommendations | #23 |

**Phase 3 exit criteria:** app hosted publicly, installable on phone, browser extension published (or sideloaded), job discovery functional.

---

## File Changes Summary (Phase 1)

| File / Location | Change |
|---|---|
| `frontend/` (new) | Next.js app — all Streamlit screens as React components |
| `app/core/jd_extractor.py` (new) | `extract_jd_from_url(url)` via Jina Reader + fallback |
| `app/api/routes/jd.py` | Add `POST /jd/extract-from-url` endpoint |
| `app/agents/edit_agent.py` | Stronger prompts in `_resume_suggestions()` and `_paraphrase()` |
| `app/core/gen_cache.py` | Add suggestions cache (key: resume+jd hash) |
| `app/graph/edit_graph.py` | Replace `MemorySaver` with SQLite checkpointer; add session persistence nodes |
| `app/utils/llm.py` | Add Groq provider; key rotation + fallback + retry logic |
| `config/config.ini` | Add `groq_api`, `mistral_api` sections |
| `streamlit_app.py` | Kept temporarily during React migration; removed in Phase 1 final |

---

## Quick-Win Checklist (Sprints 0–1, current Streamlit app)

All of these work on top of the existing codebase with no migration risk.
Complete before starting the React migration (Sprint 3).

**Sprint 0 — ✓ COMPLETE:**
- [x] `(0.1)` Remove second `pdflatex` pass in `try_compile_latex()` — `streamlit_app.py`
- [x] `(0.2)` Pre-warm embedding model in FastAPI `lifespan` startup handler — `app/api/main.py`
- [x] `(0.3)` Add MemorySaver TTL cleanup (prune threads older than 24h on startup) — `app/graph/edit_graph.py`
- [x] `(0.4)` Move JSON schema to system prompt in all LLM calls — `edit_agent.py`, `generator_agent.py`, `score_agent.py`
- [x] `(0.5)` Compress JD context in all prompts (cap to top 10 skills + top 5 responsibilities) — `edit_agent.py`
- [x] `(0.6)` Rewrite `_resume_suggestions()` prompt: impact-first bullets, XYZ formula, action verb upgrade — `edit_agent.py`
- [x] `(0.7)` Cache `_resume_suggestions()` output by `sha256(resume_latex + jd_hash)` — `gen_cache.py` + `edit_agent.py`

**Sprint 1 — ✓ COMPLETE:**
- [x] `(1.1)` API key usage dashboard endpoint `GET /keys/status` + `PATCH /keys/quota` — `app/api/routes/keys.py`
- [x] `(1.2)` Register Groq API key + wire as fallback provider — `app/utils/llm.py`, `config/config.ini`
- [x] `(1.3)` Exponential backoff retry (built into infrakit LLMClient v0.1.4)
- [x] `(1.4)` User-managed keys endpoint `POST /keys`, `DELETE /keys/{provider}/{key_id}` with `data/user_keys.json` persistence
- [x] `(1.5)` `app/core/jd_extractor.py` (Jina Reader) + `POST /jd/extract-from-url`
- [x] `(1.6)` Global custom instruction field (`custom_instruction`) on edit session start
- [x] `(1.7)` Mid-session TeX/PDF export `GET /edit/{thread_id}/export`

**Sprint 2 — Persistence Layer (✓ 2.1–2.3 done; 2.4 deferred to backend backlog):**
- [x] `(2.1)` Replace `MemorySaver` with SQLite-backed `SqliteSaver` (`langgraph-checkpoint-sqlite`) — `app/graph/edit_graph.py`
- [x] `(2.2)` Persist completed sessions to DB — `db/migrations/008_edit_sessions.sql`, `app/utils/sqlite_handler.py`, `POST /edit/{thread_id}/finish`, `GET /edit/sessions/{user_id}`
- [x] `(2.3)` Full-resume diff endpoint `GET /edit/{thread_id}/diff` — `app/api/routes/edit.py`
- [x] `(2.4)` Resume format flexibility — LLM fallback parser for PDF/DOCX/any LaTeX template

**Sprint 3 — React + Next.js Frontend (✓ COMPLETE):**
- [x] `(3.1)` Scaffold Next.js 16 + React 19 + Tailwind v4 app in `frontend/` (webpack mode for Windows compat)
- [x] `(3.2)` API client layer `src/lib/api.ts` + Pydantic-mirrored types in `src/lib/types.ts`
- [x] `(3.3)` Dashboard `/` + New session form `/session/new` (JD paste or URL fetch)
- [x] `(3.4)` Dynamic session page `/session/[threadId]` — handles all 3 LangGraph interrupts (ItemSelection, SectionReview, RefineOrFinish)
- [x] `(3.5)` Score view `/session/[threadId]/score` — ATS + quality score display
- [x] `(3.6)` Generate page `/session/[threadId]/generate` — cover letter / email / LinkedIn generation
- [x] `(3.7)` Session history list `/sessions`
- [x] `(3.8)` Settings page `/settings` — API key management (add/remove keys, per-key usage dashboard, quota editing, model status cards)
- [ ] `(3.9)` Remove `streamlit_app.py` *(deferred — keep until React app is fully verified in production)*

**Sprint 4 — Data Integrity + Template Engine + Quick Frontend Wins ← NEXT UP:**

*Phase A — Resume Data Integrity (backend):*
- [ ] `(4.1)` DB migration 009 — add `is_latest`, `parent_item_id` to `resume_section_items`; add `input_resume_id`, `output_resume_id` to `edit_sessions` — `db/migrations/009_resume_versioning.sql`
- [ ] `(4.2)` Finish endpoint writes modified items back to `resume_section_items`, marks superseded rows `is_latest=0` — `app/api/routes/edit.py`, `app/utils/sqlite_handler.py`
- [ ] `(4.3)` ChromaDB sync after finish — delete + re-embed only modified sections — `app/core/resume_parser.py`
- [ ] `(4.4)` Version history endpoint `GET /resume/{id}/history` — `app/api/routes/resume.py`

*Phase B — Template Engine & Section Flexibility (backend):*
- [ ] `(4.5)` Jinja2 template engine — rewrite `resume_builder.py` to render from structured data + layout config — `app/core/resume_builder.py`
- [ ] `(4.6)` Classic template ported to Jinja2 — `templates/resume/classic.tex.j2`
- [ ] `(4.7)` ATS Minimal template (no icons, plain sections) — `templates/resume/ats_minimal.tex.j2`
- [ ] `(4.8)` DB migration 010 — `resume_layouts` table + `section_type` column on `resume_sections` — `db/migrations/010_resume_layouts.sql`
- [ ] `(4.9)` Default layout row created on resume parse — `app/core/resume_parser.py`
- [ ] `(4.10)` Section flexibility — remove hardcoded ATOMIC/FLAT env constraint, use `section_type` DB column — `app/core/resume_parser.py`
- [ ] `(4.11)` Section CRUD endpoints + layout PATCH endpoint — `app/api/routes/resume.py`
- [ ] `(4.12)` Preview endpoint `GET /resume/{id}/preview` — compiles without saving — `app/api/routes/resume.py`
- [ ] `(4.13)` Parse preview `POST /resume/parse-preview` (dry run) + commit `POST /resume/commit` (save structured JSON) — `app/api/routes/resume.py`

*Phase C — Quick Frontend Wins:*
- [ ] `(4.14)` Re-open sessions — "Edit again" button in session history — `frontend/src/app/sessions/page.tsx`
- [ ] `(4.15)` Full-resume diff view — side-by-side component wired to existing `/diff` endpoint — `frontend/src/components/session/ResumeDiff.tsx`

**Sprint 5 — Streaming + Resume Management UI:**
- [ ] `(5.1)` Streaming LLM responses — FastAPI `StreamingResponse` SSE for `_resume_suggestions()` — `app/api/routes/edit.py`, `app/agents/edit_agent.py`
- [ ] `(5.2)` Streaming frontend consumer — `EventSource` in `SectionReview` component — `frontend/src/components/session/SectionReview.tsx`
- [ ] `(5.3)` Per-section parallel suggestions — `ThreadPoolExecutor` fan-out + stream-merge frontend — `app/agents/edit_agent.py`
- [ ] `(5.4)` Upload → verify flow redesign — two-tab (upload / manual entry), calls parse-preview — `frontend/src/app/resume/new/page.tsx`
- [ ] `(5.5)` Parse verification + manual entry form — editable personal info + sections + bullets, "Confirm & Save" → commit — `frontend/src/components/resume/ResumeEditForm.tsx`
- [ ] `(5.6)` Template picker — thumbnail grid, click → live PDF preview in iframe — `frontend/src/components/resume/TemplatePicker.tsx`
- [ ] `(5.7)` Layout controls — section drag-to-reorder, font size, margin, "Save Layout" — `frontend/src/components/resume/LayoutControls.tsx`
- [ ] `(5.8)` Resume detail page `/resume/[resumeId]` — version history, template/layout panel, preview, download — `frontend/src/app/resume/[resumeId]/page.tsx`

**Sprint 6 — Auth + Intelligence Features:**
- [ ] `(6.1)` Google OAuth backend — `authlib` + FastAPI, JWT issued as httpOnly cookie — `app/api/routes/auth.py`
- [ ] `(6.2)` DB migration 011 — add `oauth_sub`, `email`, `email_verified`, `avatar_url` to `users` — `db/migrations/011_auth.sql`
- [ ] `(6.3)` JWT middleware — `get_current_user` FastAPI dependency, dev fallback to `user_id=1` — `app/api/middleware.py`
- [ ] `(6.4)` Frontend auth flow — login page, Google OAuth button, protected route wrapper — `frontend/src/app/login/page.tsx`
- [ ] `(6.5)` Application tracker backend + DB migration 012 — CRUD for applications with Kanban statuses — `app/api/routes/applications.py`, `db/migrations/012_applications.sql`
- [ ] `(6.6)` Kanban UI — drag cards between status columns, links to resume version + docs — `frontend/src/app/tracker/page.tsx`
- [ ] `(6.7)` Skill gap analysis `GET /skills/gaps` — cross-reference JD required skills vs resume skills section — `app/api/routes/skills.py`
- [ ] `(6.8)` Interview prep generation `POST /generate/interview-prep` — 5–10 questions + talking points, cached by resume+job hash — `app/api/routes/generate.py`
- [ ] `(6.9)` LinkedIn profile generator `POST /generate/linkedin` — About / Headline / Experience bullets — `app/api/routes/generate.py`
