-- db/migrations/002_add_hash_columns_to_jobs.sql
-- --------------------------------------------------------------------
-- Adds raw_hash and parsed_hash columns to the jobs table for
-- fast exact-duplicate detection in jd_parser.py
--
-- raw_hash    : SHA-256 of stripped raw JD text
-- parsed_hash : SHA-256 of sorted parsed JSON output
--
-- Run with: python db/migrate.py
-- --------------------------------------------------------------------

ALTER TABLE jobs ADD COLUMN raw_hash    TEXT;
ALTER TABLE jobs ADD COLUMN parsed_hash TEXT;

CREATE INDEX IF NOT EXISTS idx_jobs_raw_hash    ON jobs(raw_hash);
CREATE INDEX IF NOT EXISTS idx_jobs_parsed_hash ON jobs(parsed_hash);