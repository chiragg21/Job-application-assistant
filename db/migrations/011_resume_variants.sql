-- Sprint 5: Resume variants library + adaptive style feedback

-- Stores saved resume variants (tailored, one-page, master copies, etc.)
-- Files live on disk at folder_path: data/resumes/user_{id}/{id}__{name}/
CREATE TABLE IF NOT EXISTS resume_variants (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    name            TEXT NOT NULL,
    variant_type    TEXT NOT NULL DEFAULT 'custom',  -- master | tailored | one_page | custom
    base_resume_id  INTEGER,
    folder_path     TEXT NOT NULL UNIQUE,
    page_count      INTEGER DEFAULT 1,
    tags            TEXT DEFAULT '[]',        -- JSON array of strings
    job_ids         TEXT DEFAULT '[]',        -- JSON array of job ids this was tailored for
    session_ids     TEXT DEFAULT '[]',        -- JSON array of edit session ids
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Captures per-section user decisions during the review loop for style learning
CREATE TABLE IF NOT EXISTS edit_feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER,
    user_id         INTEGER NOT NULL DEFAULT 1,
    section_name    TEXT NOT NULL,
    item_name       TEXT,
    action          TEXT NOT NULL,   -- accept | reject | paraphrase | custom | drop
    original_text   TEXT,
    final_text      TEXT,
    created_at      TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
