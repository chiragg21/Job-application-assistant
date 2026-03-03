"""
utils/sqlite_handler.py
-----------------------
Centralized SQLite handler using SQLAlchemy.
All database operations across the project go through this class.

Core methods:
    fetch_one()           - fetch a single row as dict
    exists()              - check if a row exists by criteria
    upsert()              - insert or update on conflict
    fetch_by_hash()       - fast hash-based lookup (for jd_parser duplicate check)
    bulk_insert()         - insert many rows efficiently in one transaction
    count()               - count rows with optional filters
    get_all_table_names() - list all tables in the DB

Workflow-specific methods (Resume Tailoring System):
    # Stage 0
    get_application_state()       - resolve the full duplicate-guard decision in one call
    create_application()          - insert a fresh draft application, return app_id

    # Stage 1
    save_parsed_job()             - insert jobs row + link to application
    upsert_skill()                - upsert skills row, return skill_id
    save_job_skills()             - bulk-insert job_skills rows
    get_skill_overlap_count()     - intersect resume_skills vs job_skills
    update_skill_overlap()        - persist overlap_count to applications

    # Stage 2
    fetch_section_items_by_ids()  - fetch resume_section_items WHERE id IN (...)
    fetch_sections_by_ids()       - fetch resume_sections WHERE id IN (...)
    update_similarity_score()     - persist similarity_score to applications

    # Stage 3
    fetch_flat_sections()         - fetch skills/education/etc. blobs for draft assembly
    save_generated_output()       - insert a generated_outputs row
    snapshot_latex()              - update applications.current_latex_snapshot

    # Stage 4
    create_edit()                 - insert a pending resume_edits row, return edit_id
    resolve_edit()                - accept / reject an edit
    get_edit()                    - fetch a single edit row by id
    get_accepted_edits()          - fetch all accepted edits for an application
    get_line_count_delta()        - SUM(line_count_delta) for accepted edits
    revert_edit()                 - walk parent_edit_id chain and revert

    # Stage 5
    get_final_output()            - fetch the is_draft=0 output for an application
    finalise_application()        - mark application as 'applied', persist final path
"""

import sys
import pathlib
import os

import pandas as pd
from sqlalchemy import (
    create_engine, text, Table, MetaData,
    insert, update, delete, select, inspect, func
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from config.config import get_config_dict
from utils.app_logger import get_logger, log_exception

log = get_logger(__name__)
sqldb = get_config_dict()["sqlite"]


class SQLHandler:
    def __init__(self):
        full_db_path = sqldb['sql_path']
        print(f"Initializing SQLHandler with DB path: {full_db_path}")
        db_dir = os.path.dirname(full_db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.engine   = create_engine(f"sqlite:///{full_db_path}", echo=False)
        self.metadata = MetaData()

        log.info("SQLHandler initialised", extra={"db": full_db_path})

    # ──────────────────────────────────────────────────────────────────────
    # Internal helper
    # ──────────────────────────────────────────────────────────────────────

    def _get_table(self, table_name: str) -> Table:
        """Load table definition from DB (cached by SQLAlchemy after first load)."""
        return Table(table_name, self.metadata, autoload_with=self.engine, extend_existing=True)

    # ──────────────────────────────────────────────────────────────────────
    # READ operations
    # ──────────────────────────────────────────────────────────────────────

    def fetch_multiple_tables(self, table_specs: dict) -> dict[str, pd.DataFrame]:
        """
        Fetch from multiple tables in one connection.

        Input:
            {
                'users': {'columns': ['name', 'email'], 'limit': 10},
                'jobs':  {'columns': ['title'], 'where': "status = 'open'"}
            }
        """
        results = {}
        with self.engine.connect() as conn:
            for table_name, params in table_specs.items():
                cols      = ", ".join(params.get('columns', ['*']))
                query_str = f"SELECT {cols} FROM {table_name}"
                if 'where' in params:
                    query_str += f" WHERE {params['where']}"
                if 'limit' in params:
                    query_str += f" LIMIT {params['limit']}"
                results[table_name] = pd.read_sql(text(query_str), conn)
        return results

    def fetch_table_where(
        self,
        table_name: str,
        columns: list | None = None,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch rows from a single table with optional column selection and filters.
        Returns a DataFrame.
        """
        table = self._get_table(table_name)
        stmt  = select(*[getattr(table.c, col) for col in columns]) if columns else select(table)

        if filters:
            for key, value in filters.items():
                stmt = stmt.where(getattr(table.c, key) == value)
        if limit:
            stmt = stmt.limit(limit)

        with self.engine.connect() as conn:
            return pd.read_sql(stmt, conn)

    def fetch_one(
        self,
        table_name: str,
        filters: dict|None = None,
        columns: list|None = None,
    ) -> dict | None:
        """
        Fetch a single row as a dict. Returns None if not found.
        """
        table = self._get_table(table_name)
        stmt  = select(*[getattr(table.c, col) for col in columns]) if columns else select(table)

        if filters:
            for key, value in filters.items():
                stmt = stmt.where(getattr(table.c, key) == value)
        stmt = stmt.limit(1)

        with self.engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        return dict(row._mapping) if row else None

    def fetch_by_hash(self, table_name: str, hash_column: str, hash_value: str) -> dict | None:
        """
        Fast single-column hash lookup. Used by jd_parser for duplicate detection.
        """
        return self.fetch_one(table_name, filters={hash_column: hash_value}, columns=["id", hash_column])

    def exists(self, table_name: str, filters: dict) -> bool:
        """Check if any row matches the given filters. Returns True/False."""
        first_col = list(self.get_table_schema(table_name).keys())[0]
        pk_col    = "id" if "id" in self.get_table_schema(table_name) else first_col
        return self.fetch_one(table_name, filters=filters, columns=[pk_col]) is not None

    def count(self, table_name: str, filters: dict | None = None) -> int:
        """Count rows in a table with optional filters."""
        table = self._get_table(table_name)
        stmt  = select(func.count()).select_from(table)

        if filters:
            for key, value in filters.items():
                stmt = stmt.where(getattr(table.c, key) == value)

        with self.engine.connect() as conn:
            return conn.execute(stmt).scalar() or 0

    # ──────────────────────────────────────────────────────────────────────
    # WRITE operations
    # ──────────────────────────────────────────────────────────────────────

    def add_data(self, table_name: str, data: list[dict]) -> None:
        """Insert one or multiple rows."""
        table = self._get_table(table_name)
        with self.engine.begin() as conn:
            conn.execute(insert(table), data)
        log.info("Rows inserted", extra={"table": table_name, "count": len(data)})

    def add_one(self, table_name: str, data: dict) -> int:
        """Insert a single row and return the new row's id."""
        table = self._get_table(table_name)
        with self.engine.begin() as conn:
            result = conn.execute(insert(table), data)
            if result.inserted_primary_key is None or not result.inserted_primary_key:
                raise ValueError(f"Failed to insert row into {table_name}: no primary key returned")
            new_id = result.inserted_primary_key[0]
        log.info("Row inserted", extra={"table": table_name, "id": new_id})
        return new_id

    def bulk_insert(self, table_name: str, data: list[dict]) -> int:
        """
        Insert many rows in a single transaction. Returns number of rows inserted.
        """
        if not data:
            return 0
        table = self._get_table(table_name)
        with self.engine.begin() as conn:
            conn.execute(insert(table), data)
        log.info("Bulk insert complete", extra={"table": table_name, "rows": len(data)})
        return len(data)

    def insert_or_ignore(self, table_name: str, data: dict) -> int | None:
        """
        Insert a row. If a UNIQUE constraint triggers, silently skip.
        Returns inserted id or None if skipped.
        """
        table = self._get_table(table_name)
        stmt  = sqlite_insert(table).values(**data).prefix_with("OR IGNORE")
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.inserted_primary_key[0] if result.rowcount and result.inserted_primary_key else None

    def upsert(self, table_name: str, data: dict, conflict_columns: list[str]) -> None:
        """Insert or update on conflict (SQLite INSERT OR REPLACE semantics)."""
        table = self._get_table(table_name)
        stmt  = sqlite_insert(table).values(**data)
        update_cols = {
            col: stmt.excluded[col]
            for col in data
            if col not in conflict_columns
        }
        stmt = stmt.on_conflict_do_update(index_elements=conflict_columns, set_=update_cols)
        with self.engine.begin() as conn:
            conn.execute(stmt)
        log.info("Upsert complete", extra={"table": table_name})

    def update_data(self, table_name: str, criteria: dict, new_values: dict) -> int:
        """Update rows matching criteria. Returns rows affected."""
        table = self._get_table(table_name)
        stmt  = update(table)
        for key, value in criteria.items():
            stmt = stmt.where(getattr(table.c, key) == value)

        with self.engine.begin() as conn:
            result = conn.execute(stmt.values(**new_values))
        log.info("Rows updated", extra={"table": table_name, "rows_affected": result.rowcount})
        return result.rowcount

    def delete_data(self, table_name: str, criteria: dict) -> int:
        """Delete rows matching criteria. Returns rows affected."""
        table = self._get_table(table_name)
        stmt  = delete(table)
        for key, value in criteria.items():
            stmt = stmt.where(getattr(table.c, key) == value)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        log.info("Rows deleted", extra={"table": table_name, "rows_affected": result.rowcount})
        return result.rowcount

    # ──────────────────────────────────────────────────────────────────────
    # RAW SQL
    # ──────────────────────────────────────────────────────────────────────

    def execute_raw(self, query: str, params: dict | None = None) -> list[dict] | dict:
        """
        Execute a raw SQL string. Use params dict to prevent injection.
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            if result.returns_rows:
                return [dict(row._mapping) for row in result]
            return {"status": "success", "rows_affected": result.rowcount}

    # ──────────────────────────────────────────────────────────────────────
    # IMPORT / EXPORT
    # ──────────────────────────────────────────────────────────────────────

    def export_to_csv(self, table_name: str, file_path: str) -> str:
        """Export a full table to CSV."""
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        df.to_csv(file_path, index=False)
        log.info("Table exported", extra={"table": table_name, "file": file_path})
        return f"Exported {table_name} to {file_path}"

    def import_from_csv(self, table_name: str, file_path: str, if_exists: str = "append") -> None:
        """Import data from CSV into a table."""
        df = pd.read_csv(file_path)
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)  # type: ignore[arg-type]
        log.info("CSV imported", extra={"table": table_name, "file": file_path, "rows": len(df)})

    # ──────────────────────────────────────────────────────────────────────
    # INTROSPECTION
    # ──────────────────────────────────────────────────────────────────────

    def get_table_schema(self, table_name: str) -> dict[str, str]:
        """Return column names and their types for a table."""
        inspector = inspect(self.engine)
        columns   = inspector.get_columns(table_name)
        return {col['name']: str(col['type']) for col in columns}

    def get_all_table_names(self) -> list[str]:
        """Return all table names in the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()

    # ══════════════════════════════════════════════════════════════════════
    # WORKFLOW-SPECIFIC METHODS  (Resume Tailoring System)
    # ══════════════════════════════════════════════════════════════════════
    #
    # Naming convention:
    #   Stage 0  →  get_application_state, create_application
    #   Stage 1  →  save_parsed_job, upsert_skill, save_job_skills,
    #               get_skill_overlap_count, update_skill_overlap
    #   Stage 2  →  fetch_section_items_by_ids, fetch_sections_by_ids,
    #               update_similarity_score
    #   Stage 3  →  fetch_flat_sections, save_generated_output, snapshot_latex
    #   Stage 4  →  create_edit, resolve_edit, get_edit,
    #               get_accepted_edits, get_line_count_delta, revert_edit
    #   Stage 5  →  get_final_output, finalise_application
    # ══════════════════════════════════════════════════════════════════════

    # ── Stage 0 ───────────────────────────────────────────────────────────

    def get_application_state(
        self, user_id: int, raw_hash: str
    ) -> dict:
        """
        Resolve the Stage 0 duplicate-guard decision in one call.

        Returns a dict with keys:
            job          : jobs row or None
            application  : applications row or None
            has_final    : True if a is_draft=0 generated_output exists
            resume_state : one of:
                           'no_job'            → parse JD (Stage 1)
                           'no_application'    → create app, skip Stage 1 (Stage 2)
                           'draft_only'        → skip Stages 1 & 2, go to Stage 3
                           'complete'          → return final output, stop

        Example:
            state = handler.get_application_state(user_id=1, raw_hash="abc123")
            if state['resume_state'] == 'complete':
                return state['application']['id']   # already done
        """
        job = self.fetch_by_hash("jobs", "raw_hash", raw_hash)

        if job is None:
            return {"job": None, "application": None, "has_final": False,
                    "resume_state": "no_job"}

        app = self.fetch_one(
            "applications",
            filters={"user_id": user_id, "job_id": job["id"]},
        )

        if app is None:
            return {"job": job, "application": None, "has_final": False,
                    "resume_state": "no_application"}

        # Check for a finalised output
        has_final = self.exists(
            "generated_outputs",
            {"application_id": app["id"], "is_draft": 0},
        )

        if has_final:
            return {"job": job, "application": app, "has_final": True,
                    "resume_state": "complete"}

        return {"job": job, "application": app, "has_final": False,
                "resume_state": "draft_only"}

    def create_application(self, user_id: int, job_id: int) -> int:
        """
        Insert a fresh draft application row and return the new app_id.

        Example:
            app_id = handler.create_application(user_id=1, job_id=7)
        """
        return self.add_one("applications", {
            "user_id": user_id,
            "job_id":  job_id,
            "status":  "draft",
        })

    # ── Stage 1 ───────────────────────────────────────────────────────────

    def save_parsed_job(self, job_data: dict, app_id: int) -> int:
        """
        Insert a new jobs row and link it to the application.

        job_data should contain all jobs columns except id/created_at.
        Returns job_id.

        Example:
            job_id = handler.save_parsed_job(
                job_data={
                    "user_id": 1, "jd_raw": "...", "company": "Acme",
                    "role": "SWE", "jd_parsed": "{...}", "raw_hash": "abc",
                    "parsed_hash": "def", "seniority_level": "mid",
                    "location": "Remote", "is_remote": 1,
                },
                app_id=99,
            )
        """
        job_id = self.add_one("jobs", job_data)
        self.update_data("applications", {"id": app_id}, {"job_id": job_id})
        log.info("Parsed job saved and linked", extra={"job_id": job_id, "app_id": app_id})
        return job_id

    def upsert_skill(self, name: str, category: str | None = None) -> int:
        """
        Upsert a skill by name and return its id.
        If the skill already exists, return the existing id.

        Example:
            skill_id = handler.upsert_skill("python", "programming")
        """
        self.upsert(
            "skills",
            data={"name": name, "category": category},
            conflict_columns=["name"],
        )
        row = self.fetch_one("skills", filters={"name": name}, columns=["id"])
        if row is None:
            raise ValueError(f"Skill '{name}' not found after upsert")
        return row["id"]

    def save_job_skills(self, job_id: int, skills: list[dict]) -> int:
        """
        Bulk-insert job_skills rows.
        Each item in `skills` should have keys: skill_id, is_required.

        Example:
            handler.save_job_skills(job_id=7, skills=[
                {"skill_id": 3, "is_required": 1},
                {"skill_id": 9, "is_required": 0},
            ])
        """
        rows = [{"job_id": job_id, **s} for s in skills]
        return self.bulk_insert("job_skills", rows)

    def get_skill_overlap_count(self, user_id: int, job_id: int) -> int:
        """
        Count the intersection of resume_skills and job_skills for this user/job.

        Example:
            overlap = handler.get_skill_overlap_count(user_id=1, job_id=7)
        """
        result = self.execute_raw(
            """
            SELECT COUNT(*) AS overlap
            FROM resume_skills rs
            JOIN job_skills js ON rs.skill_id = js.skill_id
            WHERE rs.user_id = :user_id
              AND js.job_id  = :job_id
            """,
            {"user_id": user_id, "job_id": job_id},
        )
        return result[0]["overlap"] if result else 0

    def update_skill_overlap(self, app_id: int, overlap_count: int) -> None:
        """
        Persist the computed skill overlap count to the applications row.

        Example:
            handler.update_skill_overlap(app_id=99, overlap_count=8)
        """
        self.update_data(
            "applications",
            criteria={"id": app_id},
            new_values={"skill_overlap_count": overlap_count},
        )

    # ── Stage 2 ───────────────────────────────────────────────────────────

    def fetch_section_items_by_ids(self, sql_ids: list[int]) -> list[dict]:
        """
        Fetch resume_section_items WHERE id IN (sql_ids) AND is_master = 1.
        Returns a list of dicts ordered to match the input id order (ChromaDB rank).

        Example:
            items = handler.fetch_section_items_by_ids([5, 12, 3])
        """
        if not sql_ids:
            return []

        rows = self.execute_raw(
            f"""
            SELECT id, section_id, section_name, item_name, role_title,
                   content_latex, content_text, item_index
            FROM resume_section_items
            WHERE id IN ({','.join('?' * len(sql_ids))})
              AND is_master = 1
            """.replace("?", ":id_{}".format(0)),  # use named params below
            # SQLite doesn't support list params natively; use execute_raw with
            # a pre-built IN clause instead:
        ) or []

        # Re-issue with raw string binding (safe — values are int ids)
        rows = self.execute_raw(
            f"SELECT id, section_id, section_name, item_name, role_title, "
            f"content_latex, content_text, item_index "
            f"FROM resume_section_items "
            f"WHERE id IN ({','.join(str(i) for i in sql_ids)}) AND is_master = 1"
        ) or []

        order = {sql_id: pos for pos, sql_id in enumerate(sql_ids)}
        return sorted(rows, key=lambda r: order.get(r["id"], 9999))

    def fetch_sections_by_ids(self, section_ids: list[int]) -> list[dict]:
        """
        Fetch resume_sections WHERE id IN (section_ids).
        Returns list of dicts with id, section_name, content_latex.

        Example:
            sections = handler.fetch_sections_by_ids([2, 5])
        """
        if not section_ids:
            return []

        result = self.execute_raw(
            f"SELECT id, section_name, content_latex "
            f"FROM resume_sections "
            f"WHERE id IN ({','.join(str(i) for i in section_ids)})"
        )
        return result if isinstance(result, list) else []

    def update_similarity_score(self, app_id: int, score: float) -> None:
        """
        Persist the computed similarity score to applications.

        Example:
            handler.update_similarity_score(app_id=99, score=0.87)
        """
        self.update_data(
            "applications",
            criteria={"id": app_id},
            new_values={"similarity_score": score},
        )

    # ── Stage 3 ───────────────────────────────────────────────────────────

    def fetch_flat_sections(self, resume_id: int) -> list[dict]:
        """
        Fetch the non-experience section blobs needed for draft assembly:
        skills, education, achievements, relevant_coursework.

        Example:
            flat = handler.fetch_flat_sections(resume_id=1)
        """
        result = self.execute_raw(
            """
            SELECT id, section_name, content_latex
            FROM resume_sections
            WHERE resume_id = :resume_id
              AND section_name IN ('skills', 'education', 'achievements', 'relevant_coursework')
            """,
            {"resume_id": resume_id},
        )
        return result if isinstance(result, list) else []

    def save_generated_output(
        self,
        app_id: int,
        output_type: str,
        content: str,
        model_used: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        is_draft: int = 1,
    ) -> int:
        """
        Insert a generated_outputs row (draft or final) and return its id.

        output_type options: 'resume_ats_draft', 'resume_impact_draft',
                             'resume_final', 'cover_letter', 'hr_email'

        Example:
            out_id = handler.save_generated_output(
                app_id=99,
                output_type="resume_ats_draft",
                content="\\documentclass{...}",
                model_used="claude-sonnet-4-20250514",
                prompt_tokens=1200,
                completion_tokens=800,
                is_draft=1,
            )
        """
        return self.add_one("generated_outputs", {
            "application_id":   app_id,
            "output_type":      output_type,
            "content":          content,
            "model_used":       model_used,
            "prompt_tokens":    prompt_tokens,
            "completion_tokens": completion_tokens,
            "is_draft":         is_draft,
        })

    def snapshot_latex(self, app_id: int, latex: str) -> None:
        """
        Update applications.current_latex_snapshot to the latest LaTeX state.
        Called after every significant generation or edit step.

        Example:
            handler.snapshot_latex(app_id=99, latex="\\documentclass{...}")
        """
        self.update_data(
            "applications",
            criteria={"id": app_id},
            new_values={"current_latex_snapshot": latex},
        )

    # ── Stage 4 ───────────────────────────────────────────────────────────

    def create_edit(
        self,
        app_id: int,
        section: str,
        original_text: str,
        suggested_text: str,
        sentence_index: int,
        line_count_delta: int = 0,
        parent_edit_id: int | None = None,
    ) -> int:
        """
        Insert a pending resume_edits row and return the new edit_id.

        Example:
            edit_id = handler.create_edit(
                app_id=99,
                section="experience",
                original_text="Worked on backend services.",
                suggested_text="Built and scaled backend microservices handling 10k RPM.",
                sentence_index=2,
                line_count_delta=1,
                parent_edit_id=None,
            )
        """
        return self.add_one("resume_edits", {
            "application_id":  app_id,
            "section":         section,
            "original_text":   original_text,
            "suggested_text":  suggested_text,
            "status":          "pending",
            "sentence_index":  sentence_index,
            "line_count_delta": line_count_delta,
            "parent_edit_id":  parent_edit_id,
        })

    def resolve_edit(
        self,
        edit_id: int,
        action: str,           # 'accepted' | 'rejected'
        final_text: str | None = None,
    ) -> None:
        """
        Accept or reject a pending edit.
        When accepting, final_text should be the confirmed text (= suggested_text).

        Example:
            handler.resolve_edit(edit_id=14, action='accepted',
                                 final_text="Built and scaled backend microservices.")
            handler.resolve_edit(edit_id=15, action='rejected')
        """
        if action not in ("accepted", "rejected"):
            raise ValueError(f"action must be 'accepted' or 'rejected', got {action!r}")

        new_values: dict = {"status": action}
        if action == "accepted" and final_text is not None:
            new_values["final_text"] = final_text

        self.update_data("resume_edits", criteria={"id": edit_id}, new_values=new_values)

    def get_edit(self, edit_id: int) -> dict | None:
        """
        Fetch a single resume_edits row by id.

        Example:
            edit = handler.get_edit(edit_id=14)
        """
        return self.fetch_one("resume_edits", filters={"id": edit_id})

    def get_accepted_edits(self, app_id: int) -> list[dict]:
        """
        Fetch all accepted edits for an application, ordered by id DESC
        (newest first) — matches the priority rule in Stage 5 resolve_item_text.

        Example:
            edits = handler.get_accepted_edits(app_id=99)
        """
        result = self.execute_raw(
            """
            SELECT id, section, sentence_index, final_text, line_count_delta
            FROM resume_edits
            WHERE application_id = :app_id
              AND status         = 'accepted'
              AND sentence_index IS NOT NULL
            ORDER BY id DESC
            """,
            {"app_id": app_id},
        )
        return result if isinstance(result, list) else []

    def get_line_count_delta(self, app_id: int) -> int:
        """
        Return SUM(line_count_delta) across all accepted edits for this application.
        Used in Stage 5 to detect page overflow before running condense.

        Example:
            delta = handler.get_line_count_delta(app_id=99)
        """
        result = self.execute_raw(
            """
            SELECT COALESCE(SUM(line_count_delta), 0) AS total_delta
            FROM resume_edits
            WHERE application_id = :app_id
              AND status         = 'accepted'
            """,
            {"app_id": app_id},
        )
        return result[0]["total_delta"] if result else 0

    def revert_edit(self, edit_id: int) -> str:
        """
        Walk the parent_edit_id chain and revert the edit.

        - Root edit (parent_edit_id IS NULL)  → returns original_text
        - Child edit                          → marks current as 'rejected',
                                                returns parent's suggested_text

        Example:
            restored_text = handler.revert_edit(edit_id=18)
        """
        edit = self.get_edit(edit_id)
        if edit is None:
            raise ValueError(f"No resume_edit found with id={edit_id}")

        if edit["parent_edit_id"] is None:
            # Root — restore the original master text
            return edit["original_text"]

        # Child — reject current, restore parent's proposal
        self.resolve_edit(edit_id, action="rejected")
        parent = self.get_edit(edit["parent_edit_id"])
        if parent is None:
            raise ValueError(f"Parent edit id={edit['parent_edit_id']} not found")
        return parent["suggested_text"]

    # ── Stage 5 ───────────────────────────────────────────────────────────

    def get_final_output(
        self, app_id: int, output_type: str = "resume_final"
    ) -> dict | None:
        """
        Fetch the authoritative (is_draft=0) output for an application.

        Example:
            final = handler.get_final_output(app_id=99)
            cover = handler.get_final_output(app_id=99, output_type="cover_letter")
        """
        return self.fetch_one(
            "generated_outputs",
            filters={"application_id": app_id, "output_type": output_type, "is_draft": 0},
        )

    def finalise_application(
        self, app_id: int, final_latex: str, resume_version_path: str
    ) -> None:
        """
        Mark the application as 'applied', persist the final LaTeX snapshot
        and .tex file path. Called once at the very end of Stage 5.

        Example:
            handler.finalise_application(
                app_id=99,
                final_latex="\\documentclass{...}",
                resume_version_path="/outputs/resumes/app_99_final.tex",
            )
        """
        self.update_data(
            "applications",
            criteria={"id": app_id},
            new_values={
                "status":                  "applied",
                "current_latex_snapshot":  final_latex,
                "resume_version_path":     resume_version_path,
            },
        )
        log.info("Application finalised", extra={"app_id": app_id, "path": resume_version_path})


# ──────────────────────────────────────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    handler = SQLHandler()

    print("Tables:", handler.get_all_table_names())
    print("Jobs schema:", handler.get_table_schema("jobs"))
    print("Job count:", handler.count("jobs"))