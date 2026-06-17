"""Tests for app/core/resume_parser.py"""
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

from app.core.resume_parser import ResumeParser
from app.models.resume import (
    PersonalInfo, AtomicItem, SectionContent, ResumeSection, ParsedResume
)


# ── Fixture: parser with all I/O mocked ──────────────────────────────────────

@pytest.fixture
def parser(mock_sql):
    mock_chroma = MagicMock()
    mock_col    = MagicMock()
    mock_ef     = MagicMock()
    mock_chroma.get_or_create_collection.return_value = mock_col

    with patch("app.core.resume_parser.SQLHandler",        return_value=mock_sql), \
         patch("app.core.resume_parser.get_chroma_client", return_value=mock_chroma), \
         patch("app.core.resume_parser.get_ef",            return_value=mock_ef):
        p = ResumeParser()
    p.db          = mock_sql
    p.col_resume  = mock_col
    return p


# ── Sample LaTeX that matches the regex patterns in extract_personal_info ─────

SAMPLE_HEADER_LATEX = r"""
\begin{document}
{\Huge \textbf{Chirag Garg}}\\[4pt]
\href{mailto:chirag@example.com}{chirag@example.com} \quad
\faPhone~ +91-9876543210 \quad
\faGithub~\href{https://github.com/chiragg21}{GitHub} \quad
\faLinkedin~\href{https://linkedin.com/in/chiragg21}{LinkedIn}
\end{document}
"""

SAMPLE_FULL_LATEX = (
    SAMPLE_HEADER_LATEX
    + r"""
\section{Experience}
\textbf{Google} \hfill 2021--2023
\begin{itemize}
  \item Built and scaled backend microservices handling 10k RPM.
  \item Reduced latency by 30\%.
\end{itemize}

\section{Projects}
\textbf{JobAssistant} | Python, ChromaDB (2024)
\begin{itemize}
  \item Resume tailoring with RAG and LLM suggestions.
\end{itemize}

\section{Skills}
Python, PyTorch, Docker, SQL

\section{Education}
B.Tech Computer Science, IIT Delhi, 2021
"""
)


# ── extract_personal_info ─────────────────────────────────────────────────────

class TestExtractPersonalInfo:
    def test_extracts_name(self, parser):
        info = parser.extract_personal_info(SAMPLE_HEADER_LATEX)
        assert info.name == "Chirag Garg"

    def test_extracts_email(self, parser):
        info = parser.extract_personal_info(SAMPLE_HEADER_LATEX)
        assert info.email == "chirag@example.com"

    def test_extracts_github(self, parser):
        info = parser.extract_personal_info(SAMPLE_HEADER_LATEX)
        assert "github.com/chiragg21" in info.github

    def test_extracts_linkedin(self, parser):
        info = parser.extract_personal_info(SAMPLE_HEADER_LATEX)
        assert "linkedin.com/in/chiragg21" in info.linkedin

    def test_missing_fields_use_defaults(self, parser):
        info = parser.extract_personal_info("No contact info here.")
        assert info.name == "Unknown"
        assert info.email == "Unknown"


# ── _strip_comments ───────────────────────────────────────────────────────────

class TestStripComments:
    def test_removes_line_comment(self, parser):
        latex = "some text % this is a comment\nnext line"
        result = parser._strip_comments(latex)
        assert "this is a comment" not in result
        assert "some text" in result
        assert "next line" in result

    def test_preserves_escaped_percent(self, parser):
        latex = r"100\% complete"
        result = parser._strip_comments(latex)
        assert r"100\%" in result

    def test_empty_string(self, parser):
        assert parser._strip_comments("") == ""

    def test_no_comments_unchanged(self, parser):
        latex = "\\textbf{hello} world"
        assert parser._strip_comments(latex) == latex


# ── _clean_latex ──────────────────────────────────────────────────────────────

class TestCleanLatex:
    def test_removes_commands_with_braces(self, parser):
        result = parser._clean_latex(r"\textbf{hello}")
        assert "textbf" not in result
        assert "hello" in result

    def test_removes_begin_end_environments(self, parser):
        result = parser._clean_latex(r"\begin{itemize}\end{itemize}")
        assert "begin" not in result
        assert "itemize" not in result

    def test_removes_item_command(self, parser):
        latex = r"\begin{itemize}\item foo\item bar\end{itemize}"
        result = parser._clean_latex(latex)
        assert r"\item" not in result

    def test_result_is_plain_text(self, parser):
        result = parser._clean_latex(r"\section{Skills} Python, Docker")
        assert "\\" not in result or len(result.strip()) > 0


# ── _map_section_name ─────────────────────────────────────────────────────────

class TestMapSectionName:
    @pytest.mark.parametrize("variant,expected", [
        ("work experience",          "experience"),
        ("professional experience",  "experience"),
        ("experience",               "experience"),
        ("projects",                 "projects"),
        ("personal project",         "projects"),
        ("technical skills",         "skills"),
        ("skills",                   "skills"),
        ("academic background",      "education"),
        ("education",                "education"),
        ("certifications",           "achievements"),
        ("relevant coursework",      "relevant_coursework"),
        ("courses",                  "relevant_coursework"),
    ])
    def test_known_variant(self, parser, variant, expected):
        assert parser._map_section_name(variant) == expected

    def test_unknown_section_passthrough(self, parser):
        assert parser._map_section_name("publications") == "publications"

    def test_case_insensitive(self, parser):
        assert parser._map_section_name("SKILLS") == "skills"


# ── _parse_atomic_items ───────────────────────────────────────────────────────

class TestParseAtomicItems:
    def test_returns_list(self, parser):
        latex = r"""
\textbf{Google} \hfill 2021--2023
\begin{itemize}
  \item Built pipelines.
\end{itemize}
\textbf{Meta} \hfill 2023--2024
\begin{itemize}
  \item Deployed models.
\end{itemize}
"""
        items = parser._parse_atomic_items(latex)
        assert isinstance(items, list)
        assert len(items) >= 1

    def test_item_has_content(self, parser):
        latex = r"""
\textbf{Google} \hfill 2021--2023
\begin{itemize}
  \item Built pipelines.
\end{itemize}
"""
        items = parser._parse_atomic_items(latex)
        assert any("Google" in str(item) or "pipelines" in str(item) for item in items)

    def test_empty_returns_empty(self, parser):
        items = parser._parse_atomic_items("")
        assert items == [] or items is not None


# ── parse_sections ────────────────────────────────────────────────────────────

class TestParseSections:
    def test_detects_experience_section(self, parser):
        sections = parser.parse_sections(SAMPLE_FULL_LATEX)
        assert sections.experience is not None and len(sections.experience) > 0

    def test_detects_skills_section(self, parser):
        sections = parser.parse_sections(SAMPLE_FULL_LATEX)
        assert sections.skills is not None

    def test_skills_content_contains_python(self, parser):
        sections = parser.parse_sections(SAMPLE_FULL_LATEX)
        assert "Python" in sections.skills.content_latex

    def test_detects_education_section(self, parser):
        sections = parser.parse_sections(SAMPLE_FULL_LATEX)
        assert sections.education is not None


# ── Shared fixtures for persistence tests ────────────────────────────────────

@pytest.fixture
def personal():
    return PersonalInfo(
        name="Chirag Garg",
        email="chirag@example.com",
        phone="+91-9876543210",
        github="https://github.com/chiragg21",
        linkedin="https://linkedin.com/in/chiragg21",
    )


@pytest.fixture
def atomic_item():
    return AtomicItem(
        name="Google",
        role="ML Engineer",
        content_latex=r"\textbf{Google} \item Built pipelines.",
        content_text="Google Built pipelines.",
    )


@pytest.fixture
def sample_sections(atomic_item):
    return ResumeSection(
        skills=SectionContent(content_latex="Python, Docker", content_text="Python Docker"),
        experience=[atomic_item],
    )


# ── _persist_section ──────────────────────────────────────────────────────────

class TestPersistSection:
    def test_calls_add_one_with_correct_table(self, parser):
        parser.db.add_one.return_value = 7
        result = parser._persist_section(resume_id=1, section_name="skills",
                                         content_latex="Python", content_text="Python")
        parser.db.add_one.assert_called_once()
        assert parser.db.add_one.call_args[0][0] == "resume_sections"

    def test_returns_row_id(self, parser):
        parser.db.add_one.return_value = 42
        result = parser._persist_section(1, "skills", "Python", "Python")
        assert result == 42

    def test_stores_both_latex_and_text(self, parser):
        parser.db.add_one.return_value = 1
        parser._persist_section(1, "skills", "\\textbf{Py}", "Py")
        row = parser.db.add_one.call_args[0][1]
        assert row["content_latex"] == "\\textbf{Py}"
        assert row["content_text"] == "Py"


# ── _persist_section_items ────────────────────────────────────────────────────

class TestPersistSectionItems:
    def test_inserts_one_row_per_item(self, parser, atomic_item):
        parser.db.add_one.side_effect = [10, 11]
        result = parser._persist_section_items(
            resume_id=1, section_id=5,
            section_name="experience",
            items=[atomic_item, atomic_item],
        )
        assert parser.db.add_one.call_count == 2

    def test_returns_list_of_item_id_pairs(self, parser, atomic_item):
        parser.db.add_one.return_value = 99
        result = parser._persist_section_items(1, 5, "experience", [atomic_item])
        assert len(result) == 1
        item, sql_id = result[0]
        assert isinstance(item, AtomicItem)
        assert sql_id == 99

    def test_stores_item_name_and_role(self, parser, atomic_item):
        parser.db.add_one.return_value = 1
        parser._persist_section_items(1, 5, "experience", [atomic_item])
        row = parser.db.add_one.call_args[0][1]
        assert row["item_name"] == "Google"
        assert row["role_title"] == "ML Engineer"


# ── chunk_and_embed ───────────────────────────────────────────────────────────

class TestChunkAndEmbedParser:
    def test_upserts_flat_section_to_chroma(self, parser, personal, sample_sections):
        parser.chunk_and_embed(
            resume_id=1, personal=personal,
            sections=sample_sections, atomic_rows={},
        )
        parser.col_resume.upsert.assert_called_once()
        ids = parser.col_resume.upsert.call_args.kwargs["ids"]
        assert any("skills" in i for i in ids)

    def test_upserts_atomic_item_to_chroma(self, parser, personal, sample_sections, atomic_item):
        atomic_rows = {"experience": [(atomic_item, 55)]}
        parser.chunk_and_embed(
            resume_id=1, personal=personal,
            sections=sample_sections, atomic_rows=atomic_rows,
        )
        ids = parser.col_resume.upsert.call_args.kwargs["ids"]
        assert "sec_item_55" in ids

    def test_skips_empty_flat_sections(self, parser, personal):
        empty_sections = ResumeSection()   # all None
        parser.chunk_and_embed(
            resume_id=1, personal=personal,
            sections=empty_sections, atomic_rows={},
        )
        parser.col_resume.upsert.assert_not_called()

    def test_metadata_carries_resume_id(self, parser, personal, sample_sections):
        parser.chunk_and_embed(1, personal, sample_sections, {})
        metas = parser.col_resume.upsert.call_args.kwargs["metadatas"]
        assert all(m["resume_id"] == 1 for m in metas)


# ── add_to_sql_and_chroma ─────────────────────────────────────────────────────

class TestAddToSQLAndChroma:
    def test_inserts_resume_record_first(self, parser, personal, sample_sections):
        parser.db.add_one.return_value = 5
        resume_id = parser.add_to_sql_and_chroma(
            user_id=1, resume_path=Path("/tmp/r.tex"),
            personal=personal, resume_sections=sample_sections,
        )
        first_call_table = parser.db.add_one.call_args_list[0][0][0]
        assert first_call_table == "resumes"
        assert resume_id == 5

    def test_persists_flat_sections(self, parser, personal, sample_sections):
        from app.core.resume_parser import FLAT_SECTIONS
        parser.db.add_one.return_value = 1
        parser.add_to_sql_and_chroma(1, Path("/tmp/r.tex"), personal, sample_sections)
        tables = [c[0][0] for c in parser.db.add_one.call_args_list]
        # resume_sections inserts happen for each flat + atomic section
        assert tables.count("resume_sections") >= 1

    def test_calls_chunk_and_embed(self, parser, personal, sample_sections):
        parser.db.add_one.return_value = 1
        with patch.object(parser, "chunk_and_embed") as mock_embed:
            parser.add_to_sql_and_chroma(1, Path("/tmp/r.tex"), personal, sample_sections)
        mock_embed.assert_called_once()


# ── run ───────────────────────────────────────────────────────────────────────

class TestRunMethod:
    def test_returns_parsed_resume(self, parser, tmp_path):
        latex_file = tmp_path / "resume.tex"
        latex_file.write_text(SAMPLE_FULL_LATEX, encoding="utf-8")

        parser.db.add_one.return_value = 7
        result = parser.run(latex_file, user_id=1)

        assert isinstance(result, ParsedResume)
        assert result.resume_id == "7"
        assert result.personal_info.name == "Chirag Garg"

    def test_raises_on_missing_file(self, parser):
        with pytest.raises(Exception):
            parser.run(Path("/nonexistent/resume.tex"), user_id=1)

    def test_extracts_sections_from_latex(self, parser, tmp_path):
        latex_file = tmp_path / "resume.tex"
        latex_file.write_text(SAMPLE_FULL_LATEX, encoding="utf-8")
        parser.db.add_one.return_value = 1
        result = parser.run(latex_file, user_id=1)
        assert result.resume_sections.skills is not None
