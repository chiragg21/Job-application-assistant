"""
conftest.py — shared fixtures and early sys.modules patches.

All external stubs must be at MODULE LEVEL so they take effect before pytest
imports any test file that transitively imports project modules.
"""
import sys
from unittest.mock import MagicMock

# ── 1. Stub infrakit (internal package) ──────────────────────────────────────
_mock_infrakit_llm = MagicMock()
_mock_infrakit_llm.LLMClient = MagicMock          # class, not instance
_mock_infrakit_llm.Prompt = MagicMock             # class, not instance

_mock_infrakit_loader = MagicMock()
_mock_infrakit_loader.load = MagicMock(return_value={})
_mock_infrakit_loader.load_env = MagicMock(return_value={})

_mock_infrakit_logger_core = MagicMock()
_mock_infrakit_logger_core.setup = MagicMock()
_mock_infrakit_logger_core.get_logger = MagicMock(return_value=MagicMock())

_mock_infrakit_llm_models = MagicMock()
_mock_infrakit_llm_models.ModelStatus = MagicMock()

for _name, _mod in [
    ("infrakit",                    MagicMock()),
    ("infrakit.llm",                _mock_infrakit_llm),
    ("infrakit.llm.models",         _mock_infrakit_llm_models),
    ("infrakit.core",               MagicMock()),
    ("infrakit.core.logger",        _mock_infrakit_logger_core),
    ("infrakit.core.config",        MagicMock()),
    ("infrakit.core.config.loader", _mock_infrakit_loader),
]:
    sys.modules.setdefault(_name, _mod)

# ── 2. Stub chromadb heavy parts so no model weights are downloaded ───────────
import chromadb
import chromadb.utils.embedding_functions as _chromadb_ef

chromadb.PersistentClient = MagicMock()
_chromadb_ef.SentenceTransformerEmbeddingFunction = MagicMock()

# ── 3. Stub config module before any project module reads it ──────────────────
MOCK_CONFIG = {
    "path_dir": {
        "data_dir":   "/tmp/test_job_assistant",
        "output_dir": "/tmp/test_output",
    },
    "rag_config": {
        "embedding_model": "all-mpnet-base-v2",
        "chunk_size": 256,
    },
    "retriever_config": {
        "top_k":              5,
        "fetch_multiplier":   3,
        "weight_req_skills":  0.5,
        "weight_resp":        0.3,
        "weight_nice_to_have": 0.2,
    },
    "resume_defaults": {
        "flat_sections":       ["summary", "skills", "education", "achievements", "relevant_coursework"],
        "atomic_sections":     ["experience", "projects"],
        "section_order":       ["summary", "experience", "projects", "skills", "education"],
        "resume_order":        ["summary", "experience", "projects", "skills", "education"],
        "font_size":           11,
        "spacing":             1.0,
        "section_space":       "8pt",
        "subsection_space":    "4pt",
    },
    "sqlite":   {"sql_path": "/tmp/test_job_assistant/test.db"},
    "chromadb": {
        "vectorstore_path": "/tmp/test_job_assistant/vectorstore",
        "path":             "/tmp/test_job_assistant/vectorstore",
        "collection1":      "jd_chunks",
        "collection2":      "resume_sections",
        "collection3":      "cached_jd_outputs",
    },
    "score_weights": {
        "ats_friendliness": 0.30,
        "keyword_match":    0.40,
        "resume_quality":   0.30,
    },
}

_mock_config_mod = MagicMock()
_mock_config_mod.get_config_dict = MagicMock(return_value=MOCK_CONFIG)
sys.modules["config"]        = MagicMock()
sys.modules["config.config"] = _mock_config_mod

# ── 4. Shared pytest fixtures ─────────────────────────────────────────────────
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def cfg():
    return MOCK_CONFIG


@pytest.fixture
def sample_jd():
    from app.models.jd import ParsedJD
    return ParsedJD(
        company="DeepMind",
        role="ML Engineer",
        seniority_level="Junior",
        location="Bangalore",
        is_remote=True,
        about_company="World-leading AI research lab.",
        about_job="Own the training infrastructure.",
        responsibilities=[
            "Build and maintain ML pipelines",
            "Collaborate with data scientists to deploy models",
        ],
        required_skills=["Python", "PyTorch", "Docker"],
        nice_to_have_skills=["LangChain", "AWS"],
        perks="Competitive salary + equity",
        raw_text="About DeepMind...",
    )


@pytest.fixture
def mock_sql():
    handler = MagicMock()
    handler.fetch_one.return_value       = None
    handler.fetch_by_hash.return_value   = None
    handler.add_one.return_value         = 1
    handler.bulk_insert.return_value     = 3
    handler.insert_or_ignore.return_value = 1
    handler.update_data.return_value     = 1
    return handler


@pytest.fixture
def mock_llm_result():
    """Fake LLMClient response: schema matched, .parsed holds the model instance."""
    resp = MagicMock()
    resp.schema_matched = True
    return resp
