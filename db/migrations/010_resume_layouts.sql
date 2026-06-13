-- Sprint 4: Resume layout config per resume + section type tracking

-- Layout configuration for each resume (template, section order, typography)
CREATE TABLE IF NOT EXISTS resume_layouts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    resume_id    INTEGER NOT NULL UNIQUE REFERENCES resumes(id),
    template_id  TEXT    NOT NULL DEFAULT 'classic',
    section_order TEXT,           -- JSON array of section keys, e.g. ["experience","education","skills"]
    font_size    INTEGER,         -- override global default (10 | 11 | 12)
    margin       REAL,            -- page margin in inches, e.g. 0.65
    created_at   TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

-- Track whether each resume_sections row is flat or atomic
-- (avoids scanning ATOMIC_SECTIONS / FLAT_SECTIONS lists at runtime)
ALTER TABLE resume_sections ADD COLUMN section_type TEXT NOT NULL DEFAULT 'flat';

-- Backfill: experience and projects are atomic; everything else is flat
UPDATE resume_sections SET section_type = 'atomic'
WHERE section_name IN ('experience', 'projects');
