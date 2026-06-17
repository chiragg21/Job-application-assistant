-- Sprint 6: Application tracker

CREATE TABLE IF NOT EXISTS applications_tracker (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        INTEGER NOT NULL DEFAULT 1,
    job_id         INTEGER,               -- FK to jobs table (nullable — manual entry allowed)
    resume_id      INTEGER,               -- which resume was used
    session_id     INTEGER,               -- which edit session produced the resume
    variant_id     INTEGER,               -- which library variant was sent
    company        TEXT NOT NULL,
    role           TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'applied',
    -- status values: applied | phone_screen | technical | final_round | offer | rejected | withdrawn
    notes          TEXT,
    applied_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    follow_up_at   TEXT,
    updated_at     TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
