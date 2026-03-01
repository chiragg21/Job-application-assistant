"""
utils/sqlite_handler.py
-----------------------
Centralized SQLite handler using SQLAlchemy.
All database operations across the project go through this class.

Added methods (on top of original):
    fetch_one()           - fetch a single row as dict
    exists()              - check if a row exists by criteria
    upsert()              - insert or update on conflict
    fetch_by_hash()       - fast hash-based lookup (for jd_parser duplicate check)
    bulk_insert()         - insert many rows efficiently in one transaction
    count()               - count rows with optional filters
    get_all_table_names() - list all tables in the DB
"""

import sys
import pathlib
# sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

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
        # Resolve absolute path relative to project root
        full_db_path = sqldb['sql_path']
        print(f"Initializing SQLHandler with DB path: {full_db_path}")
        db_dir = os.path.dirname(full_db_path)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.engine   = create_engine(f"sqlite:///{full_db_path}", echo=False)
        self.metadata = MetaData()

        log.info("SQLHandler initialised", extra={"db": full_db_path})

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _get_table(self, table_name: str) -> Table:
        """Load table definition from DB (cached by SQLAlchemy after first load)."""
        return Table(table_name, self.metadata, autoload_with=self.engine, extend_existing=True)

    # ------------------------------------------------------------------
    # READ operations
    # ------------------------------------------------------------------

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
        columns: list = None,
        filters: dict = None,
        limit: int = None,
    ) -> pd.DataFrame:
        """
        Fetch rows from a single table with optional column selection and filters.
        Returns a DataFrame.

        Example:
            handler.fetch_table_where("jobs", columns=["title"], filters={"status": "open"})
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
        filters: dict,
        columns: list = None,
    ) -> dict | None:
        """
        Fetch a single row as a dict. Returns None if not found.

        Example:
            handler.fetch_one("jobs", filters={"id": 42})
            handler.fetch_one("jobs", filters={"raw_hash": "abc123"}, columns=["id", "company"])
        """
        table = self._get_table(table_name)
        stmt  = select(*[getattr(table.c, col) for col in columns]) if columns else select(table)

        for key, value in filters.items():
            stmt = stmt.where(getattr(table.c, key) == value)
        stmt = stmt.limit(1)

        with self.engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        return dict(row._mapping) if row else None

    def fetch_by_hash(self, table_name: str, hash_column: str, hash_value: str) -> dict | None:
        """
        Fast single-column hash lookup. Used by jd_parser for duplicate detection.
        Equivalent to fetch_one but named explicitly for clarity at call sites.

        Example:
            handler.fetch_by_hash("jobs", "raw_hash", "sha256value...")
        """
        return self.fetch_one(table_name, filters={hash_column: hash_value}, columns=["id", hash_column])

    def exists(self, table_name: str, filters: dict) -> bool:
        """
        Check if any row matches the given filters. Returns True/False.

        Example:
            handler.exists("skills", {"name": "python"})
        """
        return self.fetch_one(table_name, filters=filters, columns=["id" if "id" in self.get_table_schema(table_name) else list(self.get_table_schema(table_name).keys())[0]]) is not None

    def count(self, table_name: str, filters: dict = None) -> int:
        """
        Count rows in a table with optional filters.

        Example:
            handler.count("jobs", filters={"user_id": 1})
            handler.count("applications")
        """
        table = self._get_table(table_name)
        stmt  = select(func.count()).select_from(table)

        if filters:
            for key, value in filters.items():
                stmt = stmt.where(getattr(table.c, key) == value)

        with self.engine.connect() as conn:
            return conn.execute(stmt).scalar()

    # ------------------------------------------------------------------
    # WRITE operations
    # ------------------------------------------------------------------

    def add_data(self, table_name: str, data: list[dict]) -> None:
        """
        Insert one or multiple rows.

        Example:
            handler.add_data("skills", [{"name": "python"}, {"name": "pytorch"}])
        """
        table = self._get_table(table_name)
        with self.engine.begin() as conn:
            conn.execute(insert(table), data)
        log.info("Rows inserted", extra={"table": table_name, "count": len(data)})

    def add_one(self, table_name: str, data: dict) -> int:
        """
        Insert a single row and return the new row's id.

        Example:
            job_id = handler.add_one("jobs", {"user_id": 1, "company": "Google", ...})
        """
        table = self._get_table(table_name)
        with self.engine.begin() as conn:
            result = conn.execute(insert(table), data)
            new_id = result.inserted_primary_key[0]
        log.info("Row inserted", extra={"table": table_name, "id": new_id})
        return new_id

    def bulk_insert(self, table_name: str, data: list[dict]) -> int:
        """
        Insert many rows in a single transaction. More efficient than
        calling add_data in a loop. Returns number of rows inserted.

        Example:
            handler.bulk_insert("job_skills", [
                {"job_id": 1, "skill_id": 3, "is_required": 1},
                {"job_id": 1, "skill_id": 7, "is_required": 0},
            ])
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

        Used for skills table where name is UNIQUE.

        Example:
            handler.insert_or_ignore("skills", {"name": "python"})
        """
        table = self._get_table(table_name)
        stmt  = sqlite_insert(table).values(**data).prefix_with("OR IGNORE")
        with self.engine.begin() as conn:
            result = conn.execute(stmt)
            return result.inserted_primary_key[0] if result.rowcount else None

    def upsert(self, table_name: str, data: dict, conflict_columns: list[str]) -> None:
        """
        Insert or update on conflict (SQLite INSERT OR REPLACE semantics).
        Useful for updating cached_jd_outputs flags after generating content.

        Example:
            handler.upsert(
                "generated_outputs",
                {"application_id": 1, "output_type": "cover_letter", "content": "..."},
                conflict_columns=["application_id", "output_type"]
            )
        """
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
        """
        Update rows matching criteria. Returns rows affected.

        Example:
            handler.update_data("applications", {"id": 1}, {"status": "interview"})
        """
        table = self._get_table(table_name)
        stmt  = update(table)
        for key, value in criteria.items():
            stmt = stmt.where(getattr(table.c, key) == value)

        with self.engine.begin() as conn:
            result = conn.execute(stmt.values(**new_values))
        log.info(
            "Rows updated",
            extra={"table": table_name, "rows_affected": result.rowcount}
        )
        return result.rowcount

    def delete_data(self, table_name: str, criteria: dict) -> int:
        """
        Delete rows matching criteria. Returns rows affected.

        Example:
            handler.delete_data("jobs", {"id": 42})
        """
        table = self._get_table(table_name)
        stmt  = delete(table)
        for key, value in criteria.items():
            stmt = stmt.where(getattr(table.c, key) == value)

        with self.engine.begin() as conn:
            result = conn.execute(stmt)
        log.info(
            "Rows deleted",
            extra={"table": table_name, "rows_affected": result.rowcount}
        )
        return result.rowcount

    # ------------------------------------------------------------------
    # RAW SQL
    # ------------------------------------------------------------------

    def execute_raw(self, query: str, params: dict = None) -> list[dict] | dict:
        """
        Execute a raw SQL string. Use params dict to prevent injection.

        Example:
            handler.execute_raw(
                "SELECT * FROM jobs WHERE user_id = :uid AND seniority_level = :level",
                {"uid": 1, "level": "junior"}
            )
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query), params or {})
            conn.commit()
            if result.returns_rows:
                return [dict(row._mapping) for row in result]
            return {"status": "success", "rows_affected": result.rowcount}

    # ------------------------------------------------------------------
    # IMPORT / EXPORT
    # ------------------------------------------------------------------

    def export_to_csv(self, table_name: str, file_path: str) -> str:
        """Export a full table to CSV."""
        df = pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
        df.to_csv(file_path, index=False)
        log.info("Table exported", extra={"table": table_name, "file": file_path})
        return f"Exported {table_name} to {file_path}"

    def import_from_csv(self, table_name: str, file_path: str, if_exists: str = "append") -> None:
        """Import data from CSV into a table."""
        df = pd.read_csv(file_path)
        df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        log.info("CSV imported", extra={"table": table_name, "file": file_path, "rows": len(df)})

    # ------------------------------------------------------------------
    # INTROSPECTION
    # ------------------------------------------------------------------

    def get_table_schema(self, table_name: str) -> dict[str, str]:
        """Return column names and their types for a table."""
        inspector = inspect(self.engine)
        columns   = inspector.get_columns(table_name)
        return {col['name']: str(col['type']) for col in columns}

    def get_all_table_names(self) -> list[str]:
        """Return all table names in the database."""
        inspector = inspect(self.engine)
        return inspector.get_table_names()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    handler = SQLHandler()

    print("Tables:", handler.get_all_table_names())
    print("Jobs schema:", handler.get_table_schema("jobs"))
    print("Job count:", handler.count("jobs"))