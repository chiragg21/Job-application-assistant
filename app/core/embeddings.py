"""
app.core.embeddings
~~~~~~~~~~~~~~~~~~~
Single shared SentenceTransformerEmbeddingFunction instance.

All modules that need to embed text (jd_parser, resume_parser, retriever,
score_agent) import `get_ef()` from here.  The underlying model is loaded
exactly once — when `get_ef()` is first called, which happens during the
FastAPI lifespan startup via `prewarm()`.
"""
from __future__ import annotations

from chromadb.utils import embedding_functions
from config.config import get_config_dict

_ef = None


def get_ef():
    """Return the shared embedding function, creating it on first call."""
    global _ef
    if _ef is None:
        cfg   = get_config_dict()
        model = cfg.get("rag_config", {}).get("embedding_model", "all-mpnet-base-v2")
        token = cfg.get("huggingface", {}).get("token") or None
        import os
        token = token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN") or None
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model,
            token=token,
        )
    return _ef


def prewarm() -> None:
    """Load model weights and run a no-op embed so the first real request is fast."""
    get_ef()(["warmup"])
