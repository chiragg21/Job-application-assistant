-- =============================================================================
-- Migration: Unify resume_experience_items + resume_project_items
--            into resume_section_items, with FK to resume_sections.
-- =============================================================================

-- 1. Add section_id FK column to resume_sections (already exists, but
--    confirm it has content_latex / content_text for all section types).
--    No schema change needed for resume_sections itself.

-- 2. Create the unified child table
CREATE TABLE resume_section_items (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Traceability: item -> section -> resume
    resume_id      INTEGER NOT NULL,
    section_id     INTEGER NOT NULL,               -- FK -> resume_sections.id

    -- Denormalized for fast single-table filtering
    section_name   TEXT NOT NULL,                  -- 'experience' | 'projects'

    -- Item identity
    item_name      TEXT,                           -- company name OR project name
    role_title     TEXT,                           -- job title; NULL for projects

    -- Content
    content_latex  TEXT,
    content_text   TEXT,

    -- Ordering and versioning
    item_index     INTEGER NOT NULL DEFAULT 0,     -- position within the section
    is_master      INTEGER NOT NULL DEFAULT 1,     -- 1 = original, 0 = AI variation

    FOREIGN KEY (resume_id)  REFERENCES resumes(id)         ON DELETE CASCADE,
    FOREIGN KEY (section_id) REFERENCES resume_sections(id) ON DELETE CASCADE
);

-- 3. Migrate existing experience data
--    resume_sections rows for 'experience' must exist before this runs.
--    If you are migrating live data, insert resume_sections rows first,
--    then run the INSERT below.
INSERT INTO resume_section_items
    (resume_id, section_id, section_name, item_name, role_title,
     content_latex, content_text, item_index, is_master)
SELECT
    rei.resume_id,
    rs.id          AS section_id,
    'experience'   AS section_name,
    rei.company_name,
    rei.role_title,
    rei.content_latex,
    rei.content_text,
    ROW_NUMBER() OVER (PARTITION BY rei.resume_id ORDER BY rei.id) - 1 AS item_index,
    rei.is_master
FROM resume_experience_items rei
JOIN resume_sections rs
  ON rs.resume_id    = rei.resume_id
 AND rs.section_name = 'experience';

-- 4. Migrate existing project data
INSERT INTO resume_section_items
    (resume_id, section_id, section_name, item_name, role_title,
     content_latex, content_text, item_index, is_master)
SELECT
    rpi.resume_id,
    rs.id        AS section_id,
    'projects'   AS section_name,
    rpi.project_name,
    NULL         AS role_title,   -- projects have no role_title
    rpi.content_latex,
    rpi.content_text,
    ROW_NUMBER() OVER (PARTITION BY rpi.resume_id ORDER BY rpi.id) - 1 AS item_index,
    rpi.is_master
FROM resume_project_items rpi
JOIN resume_sections rs
  ON rs.resume_id    = rpi.resume_id
 AND rs.section_name = 'projects';

-- 5. Drop the old tables (run only after verifying migration is correct)
DROP TABLE resume_experience_items;
DROP TABLE resume_project_items;

-- Useful indexes
CREATE INDEX idx_rsi_resume_id    ON resume_section_items (resume_id);
CREATE INDEX idx_rsi_section_id   ON resume_section_items (section_id);
CREATE INDEX idx_rsi_section_name ON resume_section_items (section_name);