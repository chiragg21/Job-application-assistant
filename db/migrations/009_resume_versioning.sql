-- Sprint 4: Resume item versioning — track edit history per item
ALTER TABLE resume_section_items ADD COLUMN is_latest INTEGER NOT NULL DEFAULT 1;
ALTER TABLE resume_section_items ADD COLUMN parent_item_id INTEGER REFERENCES resume_section_items(id);

-- Backfill: all existing rows are the latest version
UPDATE resume_section_items SET is_latest = 1;

-- Sprint 4: Track which resume version was used as input/output per session
ALTER TABLE edit_sessions ADD COLUMN input_resume_id INTEGER REFERENCES resumes(id);
ALTER TABLE edit_sessions ADD COLUMN output_resume_id INTEGER REFERENCES resumes(id);
