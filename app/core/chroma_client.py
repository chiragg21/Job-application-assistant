"""
Shared ChromaDB PersistentClient singleton.

All modules that need ChromaDB access should import `get_chroma_client()`
from here rather than creating their own PersistentClient instances.
Having a single client per process prevents the 'RustBindingsAPI has no
attribute bindings' error that occurs when multiple PersistentClient
objects are opened concurrently against the same SQLite path.
"""
import chromadb
from chromadb.config import Settings
from pathlib import Path

from config.config import get_config_dict

_client = None


def get_chroma_client():
    global _client
    if _client is None:
        config = get_config_dict()
        path = config["chromadb"]["path"]
        Path(path).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client
