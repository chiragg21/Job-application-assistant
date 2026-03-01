CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT,
    current_role TEXT,
    Company TEXT,
    resume_path TEXT,          -- path to master .tex file
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    company TEXT,
    role TEXT,
    jd_raw TEXT,               -- full original JD text
    jd_parsed JSON,            -- structured output from Qwen parser
    source_url TEXT,
    seniority_level TEXT,      -- junior/mid/senior extracted from JD
    location TEXT,
    is_remote INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    category TEXT              -- e.g. 'language', 'framework', 'tool', 'soft'
);

CREATE TABLE IF NOT EXISTS job_skills (
    job_id INTEGER REFERENCES jobs(id),
    skill_id INTEGER REFERENCES skills(id),
    is_required INTEGER DEFAULT 1,   -- 1=required, 0=nice-to-have
    PRIMARY KEY (job_id, skill_id)
);

CREATE TABLE IF NOT EXISTS resume_skills (
    user_id INTEGER REFERENCES users(id),
    skill_id INTEGER REFERENCES skills(id),
    PRIMARY KEY (user_id, skill_id)
);

CREATE TABLE IF NOT EXISTS applications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER REFERENCES users(id),
    job_id INTEGER REFERENCES jobs(id),
    resume_version_path TEXT,        -- snapshot of .tex used for this application
    status TEXT DEFAULT 'draft',     -- draft/applied/interview/offer/rejected
    similarity_score REAL,           -- JD vs resume match score at time of apply
    skill_overlap_count INTEGER,
    applied_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generated_outputs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER REFERENCES applications(id),
    output_type TEXT,          -- 'cover_letter', 'hr_email', 'outreach', 'linkedin'
    content TEXT,
    model_used TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS resume_edits (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    application_id INTEGER REFERENCES applications(id),
    section TEXT,              -- 'experience', 'skills', 'summary' etc
    original_text TEXT,
    suggested_text TEXT,
    final_text TEXT,           -- what user actually accepted/paraphrased to
    status TEXT,               -- 'accepted', 'rejected', 'paraphrased'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- CREATE TABLE IF NOT EXISTS interview_questions (
--     id INTEGER PRIMARY KEY AUTOINCREMENT,
--     application_id INTEGER REFERENCES applications(id),
--     question TEXT,
--     question_type TEXT,        -- 'behavioral', 'technical', 'situational'
--     user_answer TEXT,
--     llm_feedback TEXT,
--     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
-- );
```

The 'skills' + 'job_skills' + 'resume_skills' normalized structure is important — it's what makes your skill visibility dashboard and gap analysis cheap queries rather than LLM calls.

---

**ChromaDB Collections and Schema**

ChromaDB doesn't have tables, it has **collections**. Each document stored has three parts: the vector (auto-generated from your embedding model), the raw text (called `document`), and `metadata` (a flat dict of key-value pairs you define).
```
/data
    /vectorstore
        /jd_chunks          ← ChromaDB collection
        /resume_sections    ← ChromaDB collection
        /cached_jd_outputs  ← ChromaDB collection (optional, for similar JD reuse)