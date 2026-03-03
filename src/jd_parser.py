"""
pipelines/jd_parser.py
----------------------
Parses a raw job description (plain text) using Qwen-2.5-72B via
LLM handler uses gemini or openai based on input, stores structured output
in SQLite via SQLHandler, and chunks + embeds into ChromaDB.

Duplicate handling:
    1. Hash raw JD text → if match found in SQLite, skip everything
    2. Parse with LLM
    3. Hash parsed JSON → if match found, skip SQLite + ChromaDB write
    4. Otherwise → full store + embed

Similarity (ChromaDB cached_jd_outputs):
    - NOT used to skip parsing — JD is always stored fresh
    - Used ONLY by downstream pipelines (cover letter, email etc.)
      to reuse previously generated outputs for similar roles

Usage:
    from pipelines.jd_parser import JDParser

    parser = JDParser()
    result = parser.run(jd_text="...", user_id=1)
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pydantic import ValidationError

from data_models.jd_parse_dm import ParsedJD
from config.config import get_config_dict
from utils.app_logger import get_logger, log_llm_call, log_exception
from utils.sqlite_handler import SQLHandler
from utils.llm_handler import LLMHandler

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg      = get_config_dict()
path_cfg = cfg["path_dir"]
rag_cfg  = cfg["rag_config"]
llm_cfg  = cfg["openai_api"]

VECTORSTORE_PATH = Path(path_cfg["data_dir"]) / "vectorstore"
EMBEDDING_MODEL  = rag_cfg["embedding_model"]

COL_JD_CHUNKS      = "jd_chunks"
COL_CACHED_OUTPUTS = "cached_jd_outputs"




# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hash_text(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()

def _hash_dict(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JDParser
# ---------------------------------------------------------------------------

class JDParser:
    """
    Full JD ingestion pipeline with duplicate detection.

    Methods:
        check_duplicate()   — hash-based duplicate check via SQLHandler
        parse_with_llm()    — LLM helper call to parse raw JD text into structured ParsedJD
        save_to_sqlite()    — save job + skills via SQLHandler
        chunk_and_embed()   — chunk and upsert into ChromaDB
        run()               — orchestrate full pipeline
    """

    def __init__(self):
        self.db = SQLHandler()

        self.llm_helper = LLMHandler(llm_type="gemini")

        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL
        )
        chroma = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))
        self._col_jd_chunks = chroma.get_collection(COL_JD_CHUNKS)
        self._col_cached    = chroma.get_collection(COL_CACHED_OUTPUTS)

        log.info("JDParser ready", extra={"model": self.llm_helper.handler.model})

    # ------------------------------------------------------------------
    # 1. Duplicate detection
    # ------------------------------------------------------------------

    def check_duplicate(
        self,
        jd_text: str,
        parsed: Optional[ParsedJD] = None,
    ) -> tuple[Optional[int], Optional[str]]:
        """
        Pass only jd_text      → checks raw_hash (before LLM call)
        Pass jd_text + parsed  → checks parsed_hash (after LLM call)

        Returns:
            (job_id, duplicate_type)  if duplicate found
            (None,   None)            otherwise
        """
        if parsed is None:
            raw_hash = _hash_text(jd_text)
            row = self.db.fetch_by_hash("jobs", "raw_hash", raw_hash)
            if row:
                log.info("Raw hash duplicate found", extra={"job_id": row["id"]})
                return row["id"], "raw_hash"
        else:
            parsed_hash = _hash_dict(parsed.to_dict())
            row = self.db.fetch_by_hash("jobs", "parsed_hash", parsed_hash)
            if row:
                log.info("Parsed hash duplicate found", extra={"job_id": row["id"]})
                return row["id"], "parsed_hash"

        return None, None

    # ------------------------------------------------------------------
    # 2. Parse with LLM
    # ------------------------------------------------------------------

    def parse_with_llm(self, jd_text: str) -> ParsedJD:
        """
        Parse raw JD text using LLM handler.
        """
        log.info("Sending JD to LLM handler using model", extra={"model": self.llm_helper.handler.model})

        # Build schema string to include in prompt so model knows exact output format
        schema = ParsedJD.model_json_schema()

        system_prompt = (
            "You are an expert recruiter and job description analyst. "
            "Extract structured information from the job description provided by the user. "
            "Reply ONLY with a valid JSON object — no markdown, no backticks, no explanation.\n\n"
            f"Output must match this exact JSON schema:\n{json.dumps(schema, indent=2)}"
        )

        try:
            response = self.llm_helper.generate(
                f"Parse this job description:\n\n{jd_text}",
                system_prompt,
                ParsedJD,
                1200
            )
            
            parsed = response.get('content', {})

            # Set raw_text manually — excluded from schema so LLM doesn't try to fill it
            parsed.raw_text = jd_text

            # Correct token attribute names for openai SDK v1.x
            usage = response.get("usage", {})
            log_llm_call(
                pipeline="jd_parser",
                model=self.llm_helper.handler.model,
                input_tokens=usage['input_tokens'] if usage else 0,        # NOT input_tokens
                output_tokens=usage['output_tokens'] if usage else 0, # NOT output_tokens
                output_type="jd_parse",
            )

            log.info(
                "JD parsed successfully",
                extra={
                    "company":          parsed.company,
                    "role":             parsed.role,
                    "seniority":        parsed.seniority_level,
                    "required_skills":  len(parsed.required_skills),
                    "nice_to_have":     len(parsed.nice_to_have_skills),
                    "responsibilities": len(parsed.responsibilities),
                },
            )
            return parsed

        # except json.JSONDecodeError as e:
        #     log_exception(log, "LLM returned invalid JSON", raw=raw_json)
        #     raise ValueError(f"LLM returned invalid JSON: {e}") from e
        except Exception:
            log_exception(log, "LLM call failed")
            raise

    # ------------------------------------------------------------------
    # 3. Save to SQLite via SQLHandler
    # ------------------------------------------------------------------

    def save_to_sqlite(self, parsed: ParsedJD, user_id: int) -> int:
        """
        Insert parsed JD into jobs, skills, job_skills via SQLHandler.
        Returns new job_id.
        """
        log.info("Saving JD to SQLite", extra={"user_id": user_id})

        job_id = self.db.add_one("jobs", {
            "user_id":              user_id,
            "company":              parsed.company,
            "role":                 parsed.role,
            "seniority_level":      parsed.seniority_level,
            "location":             parsed.location,
            "is_remote":            int(parsed.is_remote),
            "responsibilities":     json.dumps(parsed.responsibilities),
            "required_skills":      json.dumps(parsed.required_skills),
            "nice_to_have_skills":  json.dumps(parsed.nice_to_have_skills),
            "jd_raw":               parsed.raw_text,
            "jd_parsed":            json.dumps(parsed.to_dict()),
            "raw_hash":             _hash_text(parsed.raw_text),
            "parsed_hash":          _hash_dict(parsed.to_dict()),
            "created_at":           datetime.now(timezone.utc),
        })

        self._save_skills(job_id, parsed.required_skills,     is_required=1)
        self._save_skills(job_id, parsed.nice_to_have_skills, is_required=0)

        log.info("JD saved to SQLite", extra={"job_id": job_id})
        return job_id

    def _save_skills(self, job_id: int, skills: list[str], is_required: int) -> None:
        """
        Upsert each skill into skills table, then bulk insert job_skills links.
        insert_or_ignore handles the UNIQUE constraint on skills.name.
        """
        job_skill_rows = []

        for skill in skills:
            skill = skill.strip().lower()
            if not skill:
                continue

            # Insert skill if new, skip if already exists
            self.db.insert_or_ignore("skills", {"name": skill})

            # Fetch its id
            row = self.db.fetch_one("skills", filters={"name": skill}, columns=["id"])
            if not row:
                continue

            job_skill_rows.append({
                "job_id":      job_id,
                "skill_id":    row["id"],
                "is_required": is_required,
            })

        if job_skill_rows:
            self.db.bulk_insert("job_skills", job_skill_rows)

    # ------------------------------------------------------------------
    # 4. Chunk and embed into ChromaDB
    # ------------------------------------------------------------------

    def chunk_and_embed(self, parsed: ParsedJD, job_id: int) -> None:
        """
        Split ParsedJD into typed chunks → upsert into jd_chunks.
        Upsert full JD into cached_jd_outputs for downstream output reuse.
        """
        log.info("Chunking and embedding JD", extra={"job_id": job_id})

        base_meta = {
            "job_id":    job_id,
            "company":   parsed.company,
            "role":      parsed.role,
            "seniority": parsed.seniority_level,
            "location":  parsed.location,
            "is_remote": str(parsed.is_remote),
        }

        documents, metadatas, ids = [], [], []

        def _add(chunk_type: str, text: str, extra: dict = {}):
            if not text.strip():
                return
            doc_id = f"job{job_id}_{chunk_type}_{uuid.uuid5(uuid.NAMESPACE_DNS, text)}"
            documents.append(text)
            metadatas.append({**base_meta, "chunk_type": chunk_type, **extra})
            ids.append(doc_id)

        _add(
            "company_info",
            f"Company: {parsed.company}\nRole: {parsed.role}\n"
            f"Seniority: {parsed.seniority_level}\n"
            f"Location: {parsed.location} (Remote: {parsed.is_remote})",
        )

        # One chunk per responsibility for granular retrieval
        for i, resp in enumerate(parsed.responsibilities):
            _add("responsibilities", resp, {"index": str(i)})

        if parsed.required_skills:
            _add("required_skills", ", ".join(parsed.required_skills))

        if parsed.nice_to_have_skills:
            _add("nice_to_have_skills", ", ".join(parsed.nice_to_have_skills))

        if documents:
            self._col_jd_chunks.upsert(documents=documents, metadatas=metadatas, ids=ids)
            log.info("JD chunks upserted", extra={"job_id": job_id, "chunks": len(documents)})

        # Full JD → cached_jd_outputs (cover letter / email pipelines query this)
        self._col_cached.upsert(
            documents=[parsed.raw_text],
            metadatas=[{
                **base_meta,
                "has_cover_letter": "false",
                "has_hr_email":     "false",
                "has_outreach":     "false",
            }],
            ids=[f"job{job_id}_full"],
        )
        log.info("Full JD upserted to cache", extra={"job_id": job_id})

    # ------------------------------------------------------------------
    # 5. Orchestrator
    # ------------------------------------------------------------------

    def run(self, jd_text: str, user_id: int) -> dict:
        """
        Step 1 — Raw hash check (no LLM call)
                  └─ duplicate → return stored data immediately
        Step 2 — Parse with LLM
        Step 3 — Parsed hash check
                  └─ duplicate → return parsed, skip DB write
        Step 4 — Save to SQLite
        Step 5 — Chunk and embed

        Returns:
            {
                "job_id":         int,
                "parsed":         dict,
                "duplicate":      bool,
                "duplicate_type": "raw_hash" | "parsed_hash" | None
            }
        """
        jd_text = jd_text.strip()
        if not jd_text:
            raise ValueError("JD text cannot be empty")

        log.info("JD pipeline started", extra={"user_id": user_id})

        # Step 1 — raw hash check before any LLM call
        job_id, dup_type = self.check_duplicate(jd_text)
        if job_id:
            log.info("Exact duplicate — skipping pipeline", extra={"job_id": job_id})
            row = self.db.fetch_one("jobs", filters={"id": job_id}, columns=["jd_parsed"])
            return {
                "job_id":         job_id,
                "parsed":         json.loads(row["jd_parsed"]) if row else {},
                "duplicate":      True,
                "duplicate_type": dup_type,
            }

        # Step 2 — parse with LLM
        parsed = self.parse_with_llm(jd_text)

        # Step 3 — parsed hash check
        job_id, dup_type = self.check_duplicate(jd_text, parsed)
        if job_id:
            log.info("Parsed duplicate — skipping DB write", extra={"job_id": job_id})
            return {
                "job_id":         job_id,
                "parsed":         parsed.to_dict(),
                "duplicate":      True,
                "duplicate_type": dup_type,
            }

        # Step 4 — save to SQLite
        job_id = self.save_to_sqlite(parsed, user_id)

        # Step 5 — chunk and embed
        self.chunk_and_embed(parsed, job_id)

        log.info("JD pipeline complete", extra={"job_id": job_id})

        return {
            "job_id":         job_id,
            "parsed":         parsed.to_dict(),
            "duplicate":      False,
            "duplicate_type": None,
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_jd = """
    About the Role
    We are looking for a Junior Machine Learning Engineer at DeepMind Labs, Bangalore.
    This is a remote-friendly role.

    Responsibilities:
    - Build and maintain ML pipelines for production models
    - Collaborate with data scientists to deploy models
    - Write clean, well-tested Python code

    Required Skills:
    - Python, PyTorch or TensorFlow
    - FastAPI, Docker, Git

    Nice to Have:
    - LangChain or LlamaIndex
    - MLflow, AWS or GCP
    """

    parser = JDParser()

    print("=== First run — should parse and store ===")
    result = parser.run(jd_text=sample_jd, user_id=1)
    print(json.dumps(result, indent=2))

    print("\n=== Second run — same JD, should detect duplicate ===")
    result2 = parser.run(jd_text=sample_jd, user_id=1)
    print(json.dumps(result2, indent=2))