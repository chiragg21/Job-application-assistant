"""
scripts/reset_db.py
-------------------
Hard-reset helper for local development.

SQLite  — deletes all rows from every table and resets the
          AUTOINCREMENT counter so the next insert gets id = 1.

ChromaDB — deletes and recreates every collection so all
           documents, embeddings, and metadata are wiped.
           ChromaDB has no built-in TRUNCATE; delete+recreate
           is the only reliable way to reset internal IDs.

Usage:
    python scripts/reset_db.py                 # interactive prompt
    python scripts/reset_db.py --confirm       # skip prompt (CI / scripts)
"""

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from config.config import get_config_dict
from utils.logger import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

cfg      = get_config_dict()
path_cfg = cfg["path_dir"]
rag_cfg  = cfg["rag_config"]

DB_PATH          = Path(path_cfg["data_dir"]) / "app.db"
VECTORSTORE_PATH = Path(path_cfg["data_dir"]) / "vectorstore"
EMBEDDING_MODEL  = rag_cfg["embedding_model"]

# Tables in child → parent order so FK constraints are not violated
SQLITE_TABLES = [
    "job_skills",
    "skills",
    "jobs",
    # add more tables here (child first, parent last)
]

# ChromaDB collections to wipe
CHROMA_COLLECTIONS = [
    "jd_chunks",
    "cached_jd_outputs",
    # add more collections here
]


# ---------------------------------------------------------------------------
# SQLite reset
# ---------------------------------------------------------------------------

def reset_sqlite(db_path: Path) -> None:
    """
    For every table:
        1. DELETE all rows
        2. Remove the sqlite_sequence entry → resets AUTOINCREMENT to 0
           so the next insert gets id = 1
    """
    log.info("Resetting SQLite", extra={"db": str(db_path)})
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    try:
        cur.execute("PRAGMA foreign_keys = OFF")
        cur.execute("BEGIN")

        for table in SQLITE_TABLES:
            cur.execute(f"DELETE FROM {table}")
            deleted = cur.rowcount
            # Remove the sequence counter so next id starts at 1
            cur.execute(
                "DELETE FROM sqlite_sequence WHERE name = ?", (table,)
            )
            log.info(f"  {table}: {deleted} rows deleted, sequence reset")
            print(f"  ✓ {table}: {deleted} rows deleted, id counter reset to 0")

        con.commit()
        log.info("SQLite reset complete")
        print("\n✅ SQLite reset complete.\n")

    except Exception as e:
        con.rollback()
        log.exception("SQLite reset failed — rolled back")
        print(f"\n❌ SQLite reset failed: {e}")
        raise

    finally:
        cur.execute("PRAGMA foreign_keys = ON")
        con.close()


# ---------------------------------------------------------------------------
# ChromaDB reset
# ---------------------------------------------------------------------------

def reset_chroma(vectorstore_path: Path) -> None:
    """
    For every collection:
        1. Delete the collection entirely
        2. Recreate it with the same name + embedding function
    This is the only way to reset ChromaDB's internal segment IDs.
    """
    log.info("Resetting ChromaDB", extra={"path": str(vectorstore_path)})

    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    client = chromadb.PersistentClient(path=str(vectorstore_path))

    for col_name in CHROMA_COLLECTIONS:
        try:
            existing = client.get_collection(col_name)
            count    = existing.count()
            client.delete_collection(col_name)
            client.create_collection(col_name, embedding_function=ef)
            log.info(f"  {col_name}: {count} docs deleted, collection recreated")
            print(f"  ✓ {col_name}: {count} documents removed, collection recreated")
        except Exception as e:
            # Collection might not exist yet — create it fresh
            log.warning(f"  {col_name}: could not delete ({e}), creating fresh")
            client.create_collection(col_name, embedding_function=ef)
            print(f"  ✓ {col_name}: created fresh (did not exist)")

    log.info("ChromaDB reset complete")
    print("\n✅ ChromaDB reset complete.\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-reset SQLite + ChromaDB.")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip the interactive confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.confirm:
        print("\n⚠️  WARNING: This will permanently delete ALL data from:")
        print(f"   SQLite      → {DB_PATH}")
        print(f"   ChromaDB    → {VECTORSTORE_PATH}")
        print(f"   Tables      → {', '.join(SQLITE_TABLES)}")
        print(f"   Collections → {', '.join(CHROMA_COLLECTIONS)}")
        answer = input("\nType 'yes' to continue: ").strip().lower()
        if answer != "yes":
            print("Aborted.")
            sys.exit(0)

    print()
    reset_sqlite(DB_PATH)
    reset_chroma(VECTORSTORE_PATH)
    print("🎉 Full reset done. All IDs will restart from 1 on the next insert.\n")


if __name__ == "__main__":
    main()