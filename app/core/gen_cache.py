"""
app/core/gen_cache.py
---------------------
Persistent generation cache for LLM-produced documents.

Cache key: SHA-256 of (jd_dict, resume_dict, gen_type, custom_instruction, no_jd).
Backend:   SQLite — survives server restarts and cross-session reuse.
Thread-safe via a module-level lock.

Usage
-----
    from app.core.gen_cache import generation_cache

    key = generation_cache.make_key(jd_dict, resume_dict, gen_type, instruction, no_jd)
    result = generation_cache.get(key)          # None on miss
    if result is None:
        result = _call_llm(...)
        generation_cache.set(key, result)
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

from app.utils.logger import get_logger

log = get_logger(__name__)

_DB_PATH = Path("data/gen_cache.db")
_lock    = threading.Lock()


class _GenerationCache:
    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None → autocommit; check_same_thread=False → ThreadPoolExecutor safe
        self._conn = sqlite3.connect(
            str(_DB_PATH), check_same_thread=False, isolation_level=None
        )
        with _lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gen_cache (
                    cache_key  TEXT PRIMARY KEY,
                    gen_type   TEXT NOT NULL,
                    result     TEXT NOT NULL,
                    hit_count  INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def make_key(
        self,
        jd_dict:            dict | None,   # None when jd is absent or ignored (no_jd=True)
        resume_dict:        dict,
        gen_type:           str,
        custom_instruction: str | None,    # None / "" / whitespace-only all treated as absent
        no_jd:              bool,
    ) -> str:
        """
        Deterministic SHA-256 key over all generation inputs.

        Normalisation rules so that semantically identical calls share a key:
          jd_dict      — pass None (not {}) when no JD is used; serialises to JSON null
          instruction  — None / "" / whitespace-only → stored as null in the key,
                         so "no instruction" is always the same cache entry regardless
                         of how the caller expresses the absence
          no_jd        — kept in the key because jd=None with no_jd=False (no JD provided
                         at all) produces a different prompt than jd=None with no_jd=True
                         (cold-outreach mode which injects _NO_JD_NOTE)
        """
        normalised_instruction = (custom_instruction or "").strip() or None  # "" → None
        payload = {
            "jd":          jd_dict,                            # None or full JD dict
            "resume":      resume_dict,
            "type":        gen_type.lower().replace("_", ""),
            "instruction": normalised_instruction,             # None or non-empty string
            "no_jd":       no_jd,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def get(self, key: str) -> str | None:
        """Return cached result or None on miss."""
        with _lock:
            row = self._conn.execute(
                "SELECT result FROM gen_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE gen_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (key,),
                )
                log.info("[gen_cache] HIT  key=%.12s…", key)
                return row[0]
        log.debug("[gen_cache] MISS key=%.12s…", key)
        return None

    def set(self, key: str, gen_type: str, result: str) -> None:
        """Store a generation result."""
        with _lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO gen_cache (cache_key, gen_type, result) VALUES (?, ?, ?)",
                (key, gen_type, result),
            )
        log.info("[gen_cache] STORED key=%.12s… type=%s", key, gen_type)

    def invalidate(self, key: str) -> None:
        """Remove a single entry (e.g. when forcing a regeneration)."""
        with _lock:
            self._conn.execute("DELETE FROM gen_cache WHERE cache_key = ?", (key,))

    def stats(self) -> dict:
        """Return hit counts per gen_type — useful for debugging."""
        with _lock:
            rows = self._conn.execute(
                "SELECT gen_type, COUNT(*) as entries, SUM(hit_count) as hits FROM gen_cache GROUP BY gen_type"
            ).fetchall()
        return {r[0]: {"entries": r[1], "hits": r[2]} for r in rows}


# Module-level singleton
generation_cache = _GenerationCache()


# ---------------------------------------------------------------------------
# Suggestions cache — persists _resume_suggestions() LLM output so the same
# resume+JD pair never triggers a second LLM call within or across sessions.
# Key: SHA-256 of (resume_latex, jd_hash).  Value: serialised WholeLlmOutput.
# ---------------------------------------------------------------------------

class _SuggestionsCache:
    def __init__(self) -> None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            str(_DB_PATH), check_same_thread=False, isolation_level=None
        )
        with _lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS suggestions_cache (
                    cache_key  TEXT PRIMARY KEY,
                    result     TEXT NOT NULL,
                    hit_count  INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def make_key(self, resume_latex: str, jd_hash: str) -> str:
        payload = resume_latex + "|" + jd_hash
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        with _lock:
            row = self._conn.execute(
                "SELECT result FROM suggestions_cache WHERE cache_key = ?", (key,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE suggestions_cache SET hit_count = hit_count + 1 WHERE cache_key = ?",
                    (key,),
                )
                log.info("[suggestions_cache] HIT  key=%.12s…", key)
                return row[0]
        log.debug("[suggestions_cache] MISS key=%.12s…", key)
        return None

    def set(self, key: str, result: str) -> None:
        with _lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO suggestions_cache (cache_key, result) VALUES (?, ?)",
                (key, result),
            )
        log.info("[suggestions_cache] STORED key=%.12s…", key)


suggestions_cache = _SuggestionsCache()
