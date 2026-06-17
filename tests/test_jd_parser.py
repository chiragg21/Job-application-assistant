"""Tests for app/core/jd_parser.py"""
import json
import pytest
from unittest.mock import MagicMock, patch, call

from app.core.jd_parser import _hash_text, _hash_dict, JDParser
from app.models.jd import ParsedJD


# ── Hash helpers (pure functions) ─────────────────────────────────────────────

class TestHashHelpers:
    def test_hash_text_deterministic(self):
        assert _hash_text("hello") == _hash_text("hello")

    def test_hash_text_strips_whitespace(self):
        assert _hash_text("hello") == _hash_text("  hello  ")

    def test_hash_text_different_inputs(self):
        assert _hash_text("foo") != _hash_text("bar")

    def test_hash_dict_deterministic(self):
        d = {"a": 1, "b": 2}
        assert _hash_dict(d) == _hash_dict(d)

    def test_hash_dict_key_order_invariant(self):
        assert _hash_dict({"a": 1, "b": 2}) == _hash_dict({"b": 2, "a": 1})

    def test_hash_dict_different_values(self):
        assert _hash_dict({"a": 1}) != _hash_dict({"a": 2})


# ── JDParser fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def parser(mock_sql):
    """JDParser with all external I/O mocked."""
    mock_chunks_col = MagicMock()
    mock_cached_col = MagicMock()
    mock_chroma     = MagicMock()
    mock_ef         = MagicMock()

    mock_chroma.get_or_create_collection.side_effect = (
        lambda name, **kw: mock_chunks_col if "chunks" in name else mock_cached_col
    )

    with patch("app.core.jd_parser.SQLHandler",        return_value=mock_sql), \
         patch("app.core.jd_parser.get_chroma_client", return_value=mock_chroma), \
         patch("app.core.jd_parser.get_ef",            return_value=mock_ef):
        p = JDParser()
    p.db             = mock_sql
    p._col_jd_chunks = mock_chunks_col
    p._col_cached    = mock_cached_col
    return p


# ── check_duplicate ───────────────────────────────────────────────────────────

class TestCheckDuplicate:
    def test_no_match_returns_none_none(self, parser):
        parser.db.fetch_by_hash.return_value = None
        job_id, dup_type = parser.check_duplicate("some jd text")
        assert job_id is None and dup_type is None

    def test_raw_hash_match_returns_id(self, parser):
        parser.db.fetch_by_hash.return_value = {"id": 42}
        job_id, dup_type = parser.check_duplicate("some jd text")
        assert job_id == 42
        assert dup_type == "raw_hash"

    def test_parsed_hash_match(self, parser, sample_jd):
        parser.db.fetch_by_hash.return_value = {"id": 7}
        job_id, dup_type = parser.check_duplicate("jd text", parsed=sample_jd)
        assert job_id == 7
        assert dup_type == "parsed_hash"

    def test_parsed_hash_check_called_with_correct_column(self, parser, sample_jd):
        parser.db.fetch_by_hash.return_value = None
        parser.check_duplicate("text", parsed=sample_jd)
        args = parser.db.fetch_by_hash.call_args[0]
        assert args[0] == "jobs"
        assert args[1] == "parsed_hash"
        assert isinstance(args[2], str)  # the actual SHA-256 hash


# ── parse_with_llm ────────────────────────────────────────────────────────────

class TestParseWithLLM:
    def test_returns_parsed_jd(self, parser, sample_jd):
        with patch("app.core.jd_parser.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=sample_jd, error=None)
            result = parser.parse_with_llm("JD text about ML Engineer role")
        assert isinstance(result, ParsedJD)
        assert result.raw_text == "JD text about ML Engineer role"

    def test_sets_raw_text_on_parsed(self, parser, sample_jd):
        with patch("app.core.jd_parser.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=sample_jd, error=None)
            result = parser.parse_with_llm("my raw jd text")
        assert result.raw_text == "my raw jd text"

    def test_schema_mismatch_raises(self, parser):
        with patch("app.core.jd_parser.generate_for_task") as mock_gtf:
            mock_gtf.return_value = MagicMock(
                schema_matched=False, parsed=None, error=None, content=""
            )
            with pytest.raises(ValueError):
                parser.parse_with_llm("JD text")

    def test_llm_exception_propagates(self, parser):
        with patch("app.core.jd_parser.generate_for_task") as mock_gtf:
            mock_gtf.side_effect = RuntimeError("API error")
            with pytest.raises(RuntimeError, match="API error"):
                parser.parse_with_llm("JD text")


# ── save_to_sqlite ────────────────────────────────────────────────────────────

class TestSaveToSQLite:
    def test_inserts_job_row(self, parser, sample_jd):
        parser.db.add_one.return_value = 5
        parser.db.fetch_one.return_value = {"id": 1}

        job_id = parser.save_to_sqlite(sample_jd, user_id=1)
        assert job_id == 5
        parser.db.add_one.assert_called_once()

    def test_saves_required_skills(self, parser, sample_jd):
        parser.db.add_one.return_value = 1
        parser.db.fetch_one.return_value = {"id": 1}

        parser.save_to_sqlite(sample_jd, user_id=1)
        # insert_or_ignore called once per required skill + nice_to_have skill
        total_skills = len(sample_jd.required_skills) + len(sample_jd.nice_to_have_skills)
        assert parser.db.insert_or_ignore.call_count == total_skills

    def test_saves_job_skills_links_via_bulk_insert(self, parser, sample_jd):
        parser.db.add_one.return_value = 1
        parser.db.fetch_one.return_value = {"id": 1}

        parser.save_to_sqlite(sample_jd, user_id=1)
        # bulk_insert called once for required skills, once for nice_to_have
        assert parser.db.bulk_insert.call_count == 2
        tables = [call[0][0] for call in parser.db.bulk_insert.call_args_list]
        assert all(t == "job_skills" for t in tables)


# ── chunk_and_embed ───────────────────────────────────────────────────────────

class TestChunkAndEmbed:
    def test_upserts_to_jd_chunks(self, parser, sample_jd):
        parser.chunk_and_embed(sample_jd, job_id=1)
        parser._col_jd_chunks.upsert.assert_called_once()

    def test_upserts_full_jd_to_cache(self, parser, sample_jd):
        parser.chunk_and_embed(sample_jd, job_id=1)
        parser._col_cached.upsert.assert_called_once()

    def test_chunk_count_includes_responsibilities(self, parser, sample_jd):
        parser.chunk_and_embed(sample_jd, job_id=1)
        documents = parser._col_jd_chunks.upsert.call_args.kwargs["documents"]
        # At minimum: company_info + about_company + about_job + responsibilities + skills
        assert len(documents) >= len(sample_jd.responsibilities) + 2

    def test_empty_optional_fields_not_chunked(self, parser):
        from app.models.jd import ParsedJD
        minimal_jd = ParsedJD(
            company="TestCo",
            role="Engineer",
            responsibilities=["Do stuff"],
            required_skills=["Python"],
            about_company="",   # empty — should produce no chunk
            about_job="",
            perks="",
        )
        parser.chunk_and_embed(minimal_jd, job_id=2)
        chunk_types = [
            m["chunk_type"]
            for m in parser._col_jd_chunks.upsert.call_args.kwargs["metadatas"]
        ]
        assert "about_company" not in chunk_types
        assert "about_job" not in chunk_types
        assert "perks" not in chunk_types


# ── run (full pipeline) ───────────────────────────────────────────────────────

class TestRun:
    def test_duplicate_raw_hash_skips_llm(self, parser, sample_jd):
        parser.db.fetch_by_hash.return_value = {"id": 99}
        parser.db.fetch_one.return_value = {"jd_parsed": json.dumps(sample_jd.to_dict())}

        with patch.object(parser, "parse_with_llm") as mock_parse:
            result = parser.run("some jd text", user_id=1)

        assert result["duplicate"] is True
        assert result["duplicate_type"] == "raw_hash"
        assert result["job_id"] == 99
        mock_parse.assert_not_called()

    def test_new_jd_runs_full_pipeline(self, parser, sample_jd):
        parser.db.fetch_by_hash.return_value = None
        parser.db.fetch_one.return_value = {"id": 1}
        parser.db.add_one.return_value = 1

        with patch.object(parser, "parse_with_llm", return_value=sample_jd) as mock_parse:
            result = parser.run("About DeepMind...", user_id=1)

        assert result["duplicate"] is False
        assert result["job_id"] == 1
        mock_parse.assert_called_once()
        parser._col_jd_chunks.upsert.assert_called_once()
        parser._col_cached.upsert.assert_called_once()

    def test_empty_jd_raises(self, parser):
        with pytest.raises(ValueError, match="empty"):
            parser.run("   ", user_id=1)

    def test_parsed_hash_duplicate_skips_db_write(self, parser, sample_jd):
        """Step 1 raw hash misses, Step 3 parsed hash hits — no SQL insert."""
        def _fetch_by_hash(table, col, hash_val):
            if col == "raw_hash":
                return None
            return {"id": 55}

        parser.db.fetch_by_hash.side_effect = _fetch_by_hash

        with patch.object(parser, "parse_with_llm", return_value=sample_jd):
            result = parser.run("unique jd text not seen before", user_id=1)

        assert result["duplicate"] is True
        assert result["duplicate_type"] == "parsed_hash"
        assert result["job_id"] == 55
        parser.db.add_one.assert_not_called()
        parser._col_jd_chunks.upsert.assert_not_called()
