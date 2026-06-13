"""
db/migrate.py
-------------
Lightweight migration runner for the job-application-assistant project.

How it works:
    - Reads all .sql files from db/migrations/ in numeric order (001_, 002_ ...)
    - Tracks which migrations have already been applied in a `_migrations` table
    - Skips already-applied migrations — safe to run multiple times
    - Applies only new ones, in order

Run:
    python db/migrate.py              # apply all pending migrations
    python db/migrate.py --status     # show which migrations are applied/pending
    python db/migrate.py --dry-run    # show what would run without executing

Migration file naming convention:
    db/migrations/001_initial_schema.sql
    db/migrations/002_add_hash_columns_to_jobs.sql
    db/migrations/003_add_interview_questions.sql
    ...
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from app.utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

from config.config import get_config_dict

cfg     = get_config_dict()
DB_PATH = Path(cfg["path_dir"]["data_dir"]) / "app.db"
MIGRATIONS_DIR = pathlib.Path(__file__).parent / "migrations"


# ---------------------------------------------------------------------------
# Setup migrations tracking table
# ---------------------------------------------------------------------------

MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS _migrations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL UNIQUE,
    applied_at  TIMESTAMP NOT NULL
);
"""


def _ensure_migrations_table(conn: sqlite3.Connection) -> None:
    conn.executescript(MIGRATIONS_TABLE_SQL)
    conn.commit()


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_applied_migrations(conn: sqlite3.Connection) -> set[str]:
    """Return set of migration filenames already applied."""
    rows = conn.execute("SELECT filename FROM _migrations ORDER BY filename").fetchall()
    return {row[0] for row in rows}


def get_pending_migrations() -> list[Path]:
    """
    Read db/migrations/ and return .sql files sorted numerically.
    Files must start with a number prefix: 001_, 002_, etc.
    """
    if not MIGRATIONS_DIR.exists():
        log.warning(f"Migrations directory not found: {MIGRATIONS_DIR}")
        return []

    files = sorted(
        [f for f in MIGRATIONS_DIR.glob("*.sql")],
        key=lambda f: f.name,      # lexicographic sort works with zero-padded numbers
    )
    return files


def apply_migration(conn: sqlite3.Connection, filepath: Path, dry_run: bool = False) -> None:
    """Read and execute a single migration file, then record it."""
    sql = filepath.read_text(encoding="utf-8")

    # Strip comment lines for display
    statements = [
        line for line in sql.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]

    log.info(f"Applying migration: {filepath.name}")
    for stmt in statements:
        log.info(f"  → {stmt}")

    if dry_run:
        log.info("  [DRY RUN — not executed]")
        return

    try:
        conn.executescript(sql)
        conn.execute(
            "INSERT INTO _migrations (filename, applied_at) VALUES (?, ?)",
            (filepath.name, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        log.info(f"Migration applied: {filepath.name}")

    except sqlite3.OperationalError as e:
        # Common case: column already exists (if schema.sql was manually updated)
        if "duplicate column name" in str(e).lower():
            log.warning(
                f"Column already exists — marking as applied anyway: {filepath.name}"
            )
            conn.execute(
                "INSERT OR IGNORE INTO _migrations (filename, applied_at) VALUES (?, ?)",
                (filepath.name, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        else:
            log.error(f"Migration failed: {filepath.name} — {e}")
            raise


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def show_status(conn: sqlite3.Connection) -> None:
    applied  = get_applied_migrations(conn)
    all_files = get_pending_migrations()

    if not all_files:
        print("No migration files found in db/migrations/")
        return

    print(f"\n{'FILE':<45} {'STATUS':<12} APPLIED AT")
    print("-" * 80)

    for f in all_files:
        if f.name in applied:
            row = conn.execute(
                "SELECT applied_at FROM _migrations WHERE filename = ?", (f.name,)
            ).fetchone()
            applied_at = row[0] if row else "unknown"
            print(f"  {f.name:<43} {'✓ applied':<12} {applied_at}")
        else:
            print(f"  {f.name:<43} {'○ pending':<12}")

    pending_count = sum(1 for f in all_files if f.name not in applied)
    print(f"\n{len(applied)} applied, {pending_count} pending\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run database migrations")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show migration status without running anything",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without making changes",
    )
    args = parser.parse_args()

    if not DB_PATH.exists():
        log.error(f"Database not found at {DB_PATH}. Run init_db.py first.")
        sys.exit(1)

    with sqlite3.connect(DB_PATH) as conn:
        _ensure_migrations_table(conn)

        if args.status:
            show_status(conn)
            return

        applied  = get_applied_migrations(conn)
        all_files = get_pending_migrations()
        pending  = [f for f in all_files if f.name not in applied]

        if not pending:
            log.info("No pending migrations — database is up to date")
            show_status(conn)
            return

        log.info(f"Found {len(pending)} pending migration(s)")

        if args.dry_run:
            log.info("DRY RUN — no changes will be made")

        for migration_file in pending:
            apply_migration(conn, migration_file, dry_run=args.dry_run)

        if not args.dry_run:
            log.info(f"Done — {len(pending)} migration(s) applied")

        show_status(conn)


if __name__ == "__main__":
    main()