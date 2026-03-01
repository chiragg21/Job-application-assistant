-- ======================================================
-- 1. NEW TABLES FOR ATOMIC TRACKING
-- ======================================================

-- Stores individual work history blocks (Atomic Items)
-- This prevents token bloat by allowing us to select specific roles
CREATE TABLE IF NOT EXISTS resume_experience_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    company_name TEXT,
    role_title TEXT,
    content_latex TEXT, -- Raw LaTeX for just this role
    content_text TEXT,  -- Cleaned text for ChromaDB embedding
    is_master INTEGER DEFAULT 1, -- 1 = original, 0 = AI variation
    FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE
);

-- Stores atomic project blocks
CREATE TABLE IF NOT EXISTS resume_project_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id INTEGER NOT NULL,
    project_name TEXT,
    content_latex TEXT,
    content_text TEXT,
    is_master INTEGER DEFAULT 1,
    FOREIGN KEY (resume_id) REFERENCES resumes(id) ON DELETE CASCADE
);

-- ======================================================
-- 2. UPDATING EXISTING TABLES FOR VERSIONING & STATE
-- ======================================================

-- Add constraints to 'jobs' to ensure duplicate check works at DB level
-- Note: SQLite doesn't support 'ADD UNIQUE' via ALTER, so we create a unique index
CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_raw_hash ON jobs(raw_hash);

-- Update 'resume_edits' to support Sentence-level Diffs and Reverts
-- We use parent_edit_id to create a history tree (Original -> Suggestion 1 -> Paraphrase)
ALTER TABLE resume_edits ADD COLUMN parent_edit_id INTEGER REFERENCES resume_edits(id);
ALTER TABLE resume_edits ADD COLUMN sentence_index INTEGER; -- Which bullet point is this?
ALTER TABLE resume_edits ADD COLUMN line_count_delta INTEGER DEFAULT 0; -- Line change for space optimization

-- Update 'applications' to track current working state
ALTER TABLE applications ADD COLUMN current_latex_snapshot TEXT; -- The "Live" resume state

-- Update 'generated_outputs' to track draft vs final
ALTER TABLE generated_outputs ADD COLUMN is_draft INTEGER DEFAULT 1; -- 1 = draft, 0 = final/sent

-- ======================================================
-- 3. PERFORMANCE INDEXING
-- ======================================================

-- Speed up duplicate checks and retrieval
CREATE INDEX IF NOT EXISTS idx_resume_items_res_id ON resume_experience_items(resume_id);
CREATE INDEX IF NOT EXISTS idx_edits_app_id ON resume_edits(application_id);
CREATE INDEX IF NOT EXISTS idx_skills_name_nocase ON skills(name COLLATE NOCASE);