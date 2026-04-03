"""
init_db.py
----------
Initializes the SQLite database from schema.sql and sets up
ChromaDB collections with correct embedding configuration.

Run once to set up fresh:   python init_db.py
Run to reset everything:    python init_db.py --reset

Directory structure expected:
    /db/schema.sql
    /data/app.db          (created here)
    /data/vectorstore/    (created here)
"""

import sys
import pathlib

# Ensure project root is in path regardless of where script is run from
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import argparse
import shutil
import sqlite3
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from utils.logger import get_logger
from config.config import get_config_dict

path_config = get_config_dict()['path_dir']
rag_config = get_config_dict()['rag_config']

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_DIR = Path(path_config['data_dir'])
DB_PATH = Path(path_config['data_dir']) / "app.db"
SCHEMA_PATH = Path(path_config['base_dir']) / "db" / "schema.sql"
VECTORSTORE_PATH = Path(path_config['data_dir']) / "vectorstore"

EMBEDDING_MODEL = rag_config['embedding_model']

COLLECTION_JD_CHUNKS = "jd_chunks"
COLLECTION_RESUME_SECTIONS = "resume_sections"
COLLECTION_CACHED_JD_OUTPUTS = "cached_jd_outputs"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

def _create_sqlite_db() -> None:
    """Read schema.sql and create all tables in a fresh DB file."""
    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(schema_sql)
        conn.commit()
    log.info(f"SQLite DB created at {DB_PATH}")
    _verify_sqlite_tables()


def _verify_sqlite_tables() -> None:
    """Log the tables that exist — sanity check after creation."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
    log.info(f"SQLite tables: {tables}")


def init_sqlite(reset: bool = False) -> sqlite3.Connection:
    """
    Ensure the SQLite DB exists and is initialized.

    - If reset=True  : delete existing DB and recreate from schema.sql
    - If DB missing  : create fresh DB from schema.sql
    - If DB exists   : connect and return as-is (no changes)
    """
    if not SCHEMA_PATH.exists():
        log.error(f"Schema file not found at {SCHEMA_PATH}")
        sys.exit(1)

    DB_DIR.mkdir(parents=True, exist_ok=True)

    # Reset mode — wipe and recreate
    if reset:
        if DB_PATH.exists():
            DB_PATH.unlink()
            log.info("Existing app.db deleted (reset mode)")
        _create_sqlite_db()

    # DB doesn't exist — create fresh
    elif not DB_PATH.exists():
        log.info("No existing DB found — creating fresh database")
        _create_sqlite_db()

    # DB already exists — nothing to do
    else:
        log.info(f"DB already exists at {DB_PATH} — skipping creation")
        _verify_sqlite_tables()

    return sqlite3.connect(DB_PATH)


# ---------------------------------------------------------------------------
# ChromaDB
# ---------------------------------------------------------------------------

def init_chromadb(reset: bool = False) -> chromadb.PersistentClient:
    """
    Create a persistent ChromaDB client and set up all collections.

    - If reset=True        : wipe vectorstore dir and recreate collections
    - If collections exist : get_or_create is safe to re-run, no data lost
    """
    if reset and VECTORSTORE_PATH.exists():
        shutil.rmtree(VECTORSTORE_PATH)
        log.info("Existing vectorstore deleted (reset mode)")

    VECTORSTORE_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(VECTORSTORE_PATH))

    # Single shared embedding function — critical that all collections use
    # the same model, otherwise similarity scores across collections break
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Collection 1: JD chunks
    jd_chunks = client.get_or_create_collection(
        name=COLLECTION_JD_CHUNKS,
        embedding_function=ef,
        metadata={
            "description": "Chunked JD text for semantic retrieval",
            "hnsw:space": "cosine",
        },
    )
    log.info(f"Collection '{COLLECTION_JD_CHUNKS}': {jd_chunks.count()} docs")

    # Collection 2: Resume sections
    resume_sections = client.get_or_create_collection(
        name=COLLECTION_RESUME_SECTIONS,
        embedding_function=ef,
        metadata={
            "description": "Resume sections embedded for RAG prompt construction",
            "hnsw:space": "cosine",
        },
    )
    log.info(f"Collection '{COLLECTION_RESUME_SECTIONS}': {resume_sections.count()} docs")

    # Collection 3: Cached JD outputs (whole-JD similarity for output reuse)
    cached_outputs = client.get_or_create_collection(
        name=COLLECTION_CACHED_JD_OUTPUTS,
        embedding_function=ef,
        metadata={
            "description": "Full JD embeddings for similarity-based output caching",
            "hnsw:space": "cosine",
        },
    )
    log.info(f"Collection '{COLLECTION_CACHED_JD_OUTPUTS}': {cached_outputs.count()} docs")

    log.info(f"ChromaDB vectorstore ready at {VECTORSTORE_PATH}")
    return client


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Initialize project databases")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing databases and recreate from scratch",
    )
    args = parser.parse_args()

    if args.reset:
        confirm = input("This will DELETE all existing data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            log.info("Reset cancelled")
            sys.exit(0)

    log.info("=== Initializing SQLite ===")
    init_sqlite(reset=args.reset)

    log.info("=== Initializing ChromaDB ===")
    init_chromadb(reset=args.reset)

    log.info("=== Setup complete ===")
    log.info(f"  SQLite  → {DB_PATH}")
    log.info(f"  Chroma  → {VECTORSTORE_PATH}")
    log.info(
        "\nNext steps:\n"
        "  1. Add a user:        db/init_user.py\n"
        "  2. Ingest a JD:       pipelines/ingest_jd.py\n"
        "  3. Embed your resume: pipelines/embed_resume.py"
    )


if __name__ == "__main__":
    main()