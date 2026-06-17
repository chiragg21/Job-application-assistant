-- Sprint 6: Google OAuth fields on users table
-- Note: id, name, email, github, phone_number, linkedin already exist.
-- oauth_sub may already exist from partial run — handled via migration runner fallback.

ALTER TABLE users ADD COLUMN email_verified  INTEGER DEFAULT 0;
ALTER TABLE users ADD COLUMN avatar_url      TEXT;
ALTER TABLE users ADD COLUMN last_login_at   TEXT;

-- Unique index so we can look up by oauth_sub quickly
CREATE UNIQUE INDEX IF NOT EXISTS idx_users_oauth_sub ON users(oauth_sub)
    WHERE oauth_sub IS NOT NULL;
