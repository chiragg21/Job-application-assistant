-- 1. clear existing data
DELETE FROM users;

-- 2. add the three new columns
ALTER TABLE users ADD COLUMN github    TEXT;
ALTER TABLE users ADD COLUMN phone_number     TEXT;
ALTER TABLE users ADD COLUMN linkedin TEXT;