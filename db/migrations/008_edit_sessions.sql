-- Sprint 2: persist LangGraph edit sessions and generated documents.
-- One row per edit session (= one LangGraph thread / resume tailoring run).
CREATE TABLE IF NOT EXISTS edit_sessions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id             INTEGER REFERENCES users(id),
    job_id              INTEGER REFERENCES jobs(id),
    thread_id           TEXT    NOT NULL UNIQUE,
    status              TEXT    NOT NULL DEFAULT 'active',   -- active | completed | abandoned
    custom_instruction  TEXT,
    final_latex         TEXT,           -- compiled LaTeX snapshot saved at finish
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    finished_at         TIMESTAMP
);

-- Documents generated during a session (cover letter, email, outreach message, etc.)
CREATE TABLE IF NOT EXISTS session_documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES edit_sessions(id),
    doc_type    TEXT    NOT NULL,   -- cover_letter | email | outreach_message | linkedin
    content     TEXT    NOT NULL,
    created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
