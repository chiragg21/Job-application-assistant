-- 1. clear existing data
DELETE FROM jobs;

-- 2. add the three new columns
ALTER TABLE jobs ADD COLUMN responsibilities    TEXT;
ALTER TABLE jobs ADD COLUMN required_skills     TEXT;
ALTER TABLE jobs ADD COLUMN nice_to_have_skills TEXT;