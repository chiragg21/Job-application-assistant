import json
import os
import re
from typing import List, Tuple, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from app.utils import SQLHandler, get_logger
from app.models.resume import (
    PersonalInfo, ResumeSection, ParsedResume,
    SectionContent, AtomicItem
)
from app.core.chroma_client import get_chroma_client
from app.core.embeddings import get_ef

from config.config import get_config_dict
_cfg       = get_config_dict()
resume_cnf = _cfg['resume_defaults']
_HF_TOKEN  = _cfg.get("huggingface", {}).get("token") or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_TOKEN", "")
if _HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", _HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", _HF_TOKEN)

log = get_logger(__name__)


# ── LLM output schema for the fallback parser ─────────────────────────────────

class _LLMAtomicItem(BaseModel):
    name: Optional[str] = Field(None, description="Company or project name")
    role: Optional[str] = Field(None, description="Job title or role (experience only)")
    content: str        = Field("",   description="Full plain-text content of this block")

class _LLMSection(BaseModel):
    type: str = Field(..., description=(
        "One of: experience, projects, education, skills, achievements, relevant_coursework"
    ))
    items:   List[_LLMAtomicItem] = Field(default_factory=list, description=(
        "For experience and projects: one entry per company / project block"
    ))
    content: str = Field("", description=(
        "For education, skills, achievements, relevant_coursework: full section text"
    ))

class _LLMResumeOutput(BaseModel):
    name:     str = Field("", description="Full name")
    email:    str = Field("", description="Email address")
    phone:    str = Field("", description="Phone number")
    github:   str = Field("", description="GitHub profile URL (if present)")
    linkedin: str = Field("", description="LinkedIn profile URL (if present)")
    sections: List[_LLMSection] = Field(default_factory=list)

SECTION_MAPPING = {
    "experience": ["experience", "work experience", "professional experience", "employment", "working experience"],
    "projects": ["projects", "project", "foundational machine learning projects", "personal project", "academic projects"],
    "education": ["education", "academic background", "education background"],
    "skills": ["skills", "technical skills", "programming skills"],
    "achievements": ["achievements", "certifications", "awards", "honors"],
    "relevant_coursework": ["relevant coursework", "relevant courses", "coursework", "courses"]
}

# Sections that are split into per-company/per-project AtomicItems
ATOMIC_SECTIONS = resume_cnf['atomic_sections']

# Sections stored as a single content blob
FLAT_SECTIONS = resume_cnf['flat_sections']


class ResumeParser:
    def __init__(self):
        self.db = SQLHandler()

        self.ef = get_ef()
        self.chroma_client = get_chroma_client()

        self.col_resume = self.chroma_client.get_or_create_collection(
            name="resume_sections",
            embedding_function=self.ef,
        )

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def extract_personal_info(self, latex: str) -> PersonalInfo:
        """Extracts real contact info (URLs, email) from LaTeX header."""
        info = PersonalInfo(
            name="Unknown", email="Unknown", phone="Unknown",
            github="", linkedin=""
        )

        name_match = re.search(r'\\Huge\s+\\textbf\{(.*?)\}', latex)
        if name_match:
            info.name = name_match.group(1).strip()

        email_match = re.search(r'\\href\{mailto:(.*?)\}', latex)
        if email_match:
            info.email = email_match.group(1).strip()

        phone_match = re.search(r'\\faPhone~(.*?)(?=\s+\\quad|\\\\|$)', latex)
        if phone_match:
            info.phone = re.sub(r'\\[a-z]+', '', phone_match.group(1)).strip()

        github_match = re.search(r'\\faGithub~\\href\{(.*?)\}', latex)
        if github_match:
            info.github = github_match.group(1).strip()

        linkedin_match = re.search(r'\\faLinkedin~\\href\{(.*?)\}', latex)
        if linkedin_match:
            info.linkedin = linkedin_match.group(1).strip()

        return info

    def _strip_comments(self, latex: str) -> str:
        """
        Removes LaTeX line comments (% to end of line).
        Handles escaped percent signs (\\%) correctly by only stripping
        unescaped % characters.
        """
        return re.sub(r'(?<!\\)%[^\n]*', '', latex)

    def _clean_latex(self, latex_str: str) -> str:
        """
        Strips LaTeX markup and returns clean plain text.

        Handles (in order):
        1. \\begin{env}[opts] / \\end{env}        — environment delimiters
        2. \\item                                  — bullet markers
        3. \\\\ (line break) and \\hfill           — layout commands
        4. \\vspace{}, \\hspace{} and similar      — spacing commands
        5. \\textbf{x}, \\textit{x}, \\href{}{x}  — content-wrapping commands
        6. Any remaining \\command[opts]{content}  — generic fallback
        7. Bare \\command tokens with no braces    — e.g. \\quad, \\hfill
        8. Leftover braces / special chars
        """
        text = latex_str

        # 1. Remove \begin{...}[optional] and \end{...}
        text = re.sub(r'\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?', '', text)

        # 2. Remove \item (the bullet marker itself)
        text = re.sub(r'\\item\b', '', text)

        # 3. Remove \\ (LaTeX line break) — must come before generic command strip
        text = re.sub(r'\\\\', ' ', text)

        # 4. Remove spacing/layout commands that take a length argument
        #    e.g. \vspace{-2.5mm}, \hspace{1em}
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*(?:mm|cm|em|ex|pt|in|px|\\)[^}]*\}', '', text)

        # 5. \href{url}{display} — keep only display text
        text = re.sub(r'\\href\{[^}]*\}\{([^}]*)\}', r'\1', text)

        # 6. Generic \command[opt]{content} — keep content, drop command+braces
        text = re.sub(r'\\[a-zA-Z]+(?:\[[^\]]*\])?\{([^}]*)\}', r'\1', text)

        # 7. Bare \command tokens (no braces), including \hfill, \quad, \faPhone etc.
        text = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', ' ', text)

        # 8. Leftover braces, tilde non-breaking spaces, and stray backslashes
        text = text.replace('{', '').replace('}', '')
        text = text.replace('~', ' ')
        text = text.replace('&', '&')
        text = re.sub(r'\\(?=[^a-zA-Z])', '', text)  # lone backslash before non-letter
        # 9. Collapse whitespace
        return ' '.join(text.split())

    def _map_section_name(self, title: str) -> str:
        """Maps raw LaTeX section titles to standardized field names."""
        title_lower = title.strip().lower()
        for standard, variants in SECTION_MAPPING.items():
            if title_lower in variants:
                return standard
        return title_lower.replace(" ", "_")

    def _parse_atomic_items(self, section_latex: str) -> List[AtomicItem]:
        """
        Splits a section blob into one AtomicItem per company/project block.

        A new block is detected only when \\textbf{} appears at the START OF A
        LINE (after optional whitespace). This prevents inline bold phrases
        inside bullet points (e.g. "Built a \\textbf{framework} for...") from
        being mistaken for block headers.

        Two resume layouts are supported:

        Layout A — named subsections (experience):
            \\textbf{Company Name} \\hfill \\textit{Date Range} \\\\
            \\textit{Role Title} ...
            \\begin{itemize}
                \\item ...
            \\end{itemize}

        Layout B — no subsection header (some project sections):
            \\textbf{Academic Projects} \\hfill \\textit{2023-2024}
            \\begin{itemize}
                \\item ...
            \\end{itemize}
            -> Treated as a single item with name extracted if present,
               or name=None if the section has no leading \\textbf{} at all.
        """
        # Split ONLY on \textbf that starts a line AND is followed by \hfill
        # on the same line — that combination uniquely identifies section-header
        # lines (e.g. "\textbf{Company} \hfill \textit{Date} \\").
        # Inline bold inside bullet points never has \hfill, so they are
        # never mistaken for block headers even when they start a wrapped line.
        header_pattern = re.compile(r'(?m)(?=^\s*\\textbf\{[^}]+\}[^\n]*\\hfill)')
        raw_blocks = header_pattern.split(section_latex)

        # Filter out empty fragments (e.g. whitespace before the first header)
        raw_blocks = [b.strip() for b in raw_blocks if b.strip()]

        # If no line-leading \textbf was found at all (unusual layout),
        # treat the entire section as one nameless item.
        if not raw_blocks:
            raw_blocks = [section_latex.strip()]

        items: List[AtomicItem] = []
        for block in raw_blocks:
            if not block:
                continue
            
            # --- Name: \textbf at the very start of the block ---
            name_match = re.match(r'\s*\\textbf\{(.*?)\}', block)
            name = name_match.group(1).strip() if name_match else None

            # --- Role: first \textit{} within the header region only.
            # The header is considered to end at \begin{itemize} or after
            # the first \\\\, whichever comes first — this avoids picking
            # up date-range italics or bullet-point italics as the role.
            header_end = re.search(r'\\begin\{itemize\}|\\\\', block)
            header_region = block[:header_end.start()] if header_end else block[:300]

            # Skip the first \textit{} on the same token as \textbf (that's
            # usually the date range on the same line), and grab the next one
            # which is the role title on line 2.
            all_textit = re.findall(r'\\textit\{(.*?)\}', header_region)
            # Heuristic: date ranges contain digits or '–', role titles don't
            role = None
            for candidate in all_textit:
                if not re.search(r'\d|–|-', candidate):
                    role = candidate.strip()
                    break
            
            # Fallback: role as plain text on the line after the \textbf{...} \hfill ... \\
            if role is None:
                plain_role_match = re.search(r'\\\\[\r\n]+([^\\\n\r{]+?)(?:\s*[\r\n]|\s*\\begin)', block)
                if plain_role_match:
                    candidate = plain_role_match.group(1).strip()
                    if candidate and not re.search(r'\d|–|-', candidate):
                        role = candidate

            items.append(AtomicItem(
                name=name,
                role=role,
                content_latex=block,
                content_text=self._clean_latex(block)
            ))

        return items

    # ------------------------------------------------------------------
    # File-type text extraction (fast path for non-LaTeX formats)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text_from_pdf(path: Path) -> str:
        """Extract plain text from a PDF using pymupdf (already in deps as fitz)."""
        import fitz  # pymupdf
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)

    @staticmethod
    def _extract_text_from_docx(path: Path) -> str:
        """Extract plain text from a DOCX file using python-docx."""
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    def _extract_raw_text(self, path: Path) -> str:
        """Dispatch to the right extractor based on file extension."""
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._extract_text_from_pdf(path)
        if suffix in (".docx", ".doc"):
            return self._extract_text_from_docx(path)
        # .tex or plain text — read as UTF-8
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Regex parse quality check
    # ------------------------------------------------------------------

    @staticmethod
    def _has_meaningful_sections(sections: ResumeSection) -> bool:
        """Return True if the regex parser found at least one non-empty section."""
        return bool(
            sections.experience
            or sections.projects
            or sections.education
            or sections.skills
        )

    # ------------------------------------------------------------------
    # LLM fallback parser
    # ------------------------------------------------------------------

    def _llm_parse_resume(self, raw_text: str) -> Tuple[PersonalInfo, ResumeSection]:
        """
        Use an LLM to extract structured resume data from plain text.
        Called when the regex fast-path fails (unknown template, PDF, DOCX).
        Returns (PersonalInfo, ResumeSection) ready for SQL + embedding.
        """
        from app.utils.llm import generate_for_task, Prompt

        system_prompt = (
            "You are a resume parser. Extract structured information from the resume text below.\n\n"
            "Rules:\n"
            "- For 'experience' and 'projects' sections, split into one item per company/project. "
            "Include all bullet points in that item's content.\n"
            "- For 'education', 'skills', 'achievements', 'relevant_coursework': put the full "
            "section text in content (no items array needed).\n"
            "- Use only these section types: experience, projects, education, skills, "
            "achievements, relevant_coursework.\n"
            "- If a section is absent, omit it.\n"
            "- Extract URLs exactly as written (github.com/..., linkedin.com/...). "
            "Leave fields empty string if not found."
        )
        user_prompt = f"Parse this resume:\n\n{raw_text[:12000]}"  # cap at ~12K chars

        try:
            resp = generate_for_task(
                "resume_parsing",
                Prompt(system=system_prompt, user=user_prompt),
                _LLMResumeOutput,
            )
        except (ValueError, RuntimeError) as exc:
            raise RuntimeError(f"LLM resume parser unavailable: {exc}") from exc

        if not resp.schema_matched or resp.parsed is None:
            raise ValueError("LLM resume parser returned a malformed response")

        return self._llm_output_to_resume(resp.parsed)

    def _llm_output_to_resume(
        self, out: _LLMResumeOutput
    ) -> Tuple[PersonalInfo, ResumeSection]:
        """Convert the LLM-extracted output to our domain models."""
        personal = PersonalInfo(
            name=out.name     or "Unknown",
            email=out.email   or "Unknown",
            phone=out.phone   or "Unknown",
            github=out.github   or "",
            linkedin=out.linkedin or "",
        )

        sections = ResumeSection()
        for sec in out.sections:
            stype = sec.type.lower().strip()

            if stype in ATOMIC_SECTIONS:
                items: List[AtomicItem] = []
                for it in sec.items:
                    text = (it.content or "").strip()
                    if not text:
                        continue
                    items.append(AtomicItem(
                        name=it.name,
                        role=it.role,
                        content_latex=text,   # plain text doubles as latex for non-LaTeX sources
                        content_text=text,
                    ))
                if items and hasattr(sections, stype):
                    setattr(sections, stype, items)

            elif stype in FLAT_SECTIONS:
                text = (sec.content or "").strip()
                if text and hasattr(sections, stype):
                    setattr(sections, stype, SectionContent(
                        content_latex=text,
                        content_text=text,
                    ))

        return personal, sections

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_sections(self, latex: str) -> ResumeSection:
        """
        Splits LaTeX by \\section{} commands into a single ResumeSection.
        Flat sections -> SectionContent, atomic sections -> List[AtomicItem].
        Comments are stripped first so they never pollute section content.
        """
        latex = self._strip_comments(latex)
        pattern = r'\\section\*?\{(.*?)\}(.*?)(?=\\section|\Z)'
        matches = re.findall(pattern, latex, re.DOTALL)

        result = ResumeSection()
        for title, content in matches:
            norm_name = self._map_section_name(title)
            if not hasattr(result, norm_name):
                continue

            content = content.strip()
            content = re.sub(r'\\vspace\{[^}]*\}\n?', '', content).strip()
            if norm_name in ATOMIC_SECTIONS:
                setattr(result, norm_name, self._parse_atomic_items(content))
            elif norm_name in FLAT_SECTIONS:
                setattr(result, norm_name, SectionContent(
                    content_latex=content,
                    content_text=self._clean_latex(content)
                ))

        return result

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_section(
        self,
        resume_id: int,
        section_name: str,
        content_latex: Optional[str],
        content_text: Optional[str],
        section_type: str = "flat",
    ) -> int:
        """
        Inserts one row into resume_sections and returns its id.
        Called for every section — flat and atomic alike — so every
        resume_section_items row can carry a valid section_id FK.
        """
        return self.db.add_one("resume_sections", {
            "resume_id":    resume_id,
            "section_name": section_name,
            "content_latex": content_latex,
            "content_text":  content_text,
            "section_type":  section_type,
        })

    def _persist_section_items(
        self,
        resume_id: int,
        section_id: int,
        section_name: str,
        items: List[AtomicItem],
    ) -> List[Tuple[AtomicItem, int]]:
        """
        Inserts one row per AtomicItem into resume_section_items.
        Returns List[(item, sql_row_id)] for use in chunk_and_embed.

        role_title is NULL for projects — the column is nullable so both
        experience and project items live in the same table without a dummy value.
        """
        rows: List[Tuple[AtomicItem, int]] = []
        for idx, item in enumerate(items):
            row_id = self.db.add_one("resume_section_items", {
                "resume_id": resume_id,
                "section_id": section_id,       # FK -> resume_sections.id
                "section_name": section_name,   # denormalized for fast filtering
                "item_name": item.name,
                "role_title": item.role,        # NULL for projects
                "content_latex": item.content_latex,
                "content_text": item.content_text,
                "item_index": idx,
                "is_master": 1,
            })
            rows.append((item, row_id))
        return rows

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def chunk_and_embed(
        self,
        resume_id: int,
        personal: PersonalInfo,
        sections: ResumeSection,
        atomic_rows: dict,  # {section_name: List[Tuple[AtomicItem, int]]}
    ):
        """
        Upserts all section content into ChromaDB.

        Flat sections   -> one document per section  (id: res{resume_id}_{section_name})
        Atomic sections -> one document per item      (id: sec_item_{sql_row_id})

        Atomic item metadata carries sql_id so a vector result can be traced:
            ChromaDB -> resume_section_items.id -> resume_sections.id -> resumes.id
        """
        log.info(f"Embedding resume sections for resume_id={resume_id}")

        documents, metadatas, ids = [], [], []

        base_meta = {
            "resume_id": resume_id,
            "user_name": personal.name,
            "email": personal.email,
        }

        # 1. Flat sections — one chunk each
        for sec_name in FLAT_SECTIONS:
            value: SectionContent = getattr(sections, sec_name, None)
            if value and value.content_text and value.content_text.strip():
                documents.append(value.content_text)
                metadatas.append({**base_meta, "section_type": sec_name})
                ids.append(f"res{resume_id}_{sec_name}")

        # 2. Atomic sections — one chunk per item, keyed by SQL row id
        for sec_name, rows in atomic_rows.items():
            for item, sql_id in rows:
                if not (item.content_text and item.content_text.strip()):
                    continue
                documents.append(item.content_text)
                metadatas.append({
                    **base_meta,
                    "section_type": sec_name,
                    "sql_table": "resume_section_items",
                    "sql_id": sql_id,
                    "item_name": item.name or "",
                    "role_title": item.role or "",  # empty string for projects
                })
                ids.append(f"sec_item_{sql_id}")

        if documents:
            self.col_resume.upsert(documents=documents, metadatas=metadatas, ids=ids)
            log.info(f"Successfully embedded {len(documents)} chunks for resume_id={resume_id}")

    def add_to_sql_and_chroma(self, user_id: int, resume_path: Path, personal: PersonalInfo, resume_sections: ResumeSection):
        # 1. Insert master record into resumes
        resume_id = self.db.add_one("resumes", {
            "user_id": user_id,
            "resume_path": str(resume_path),
            "name": personal.name,
            "email": personal.email,
            "phone": personal.phone,
            "github_url": personal.github,
            "linkedin_url": personal.linkedin,
        })

        # 2. Persist all sections into resume_sections (one row each).
        #    Atomic sections additionally fan out into resume_section_items.
        atomic_rows = {}

        # Flat sections — just the parent row, no child items
        for sec_name in FLAT_SECTIONS:
            sc: Optional[SectionContent] = getattr(resume_sections, sec_name, None)
            self._persist_section(
                resume_id, sec_name,
                content_latex=sc.content_latex if sc else None,
                content_text=sc.content_text if sc else None,
            )

        # Atomic sections — parent row (full blob) + child item rows
        for sec_name in ATOMIC_SECTIONS:
            items: List[AtomicItem] = getattr(resume_sections, sec_name, [])

            # Reconstruct full section blob from individual items
            full_latex = "\n\n".join(i.content_latex for i in items) or None
            full_text = " ".join(i.content_text for i in items) or None

            section_id = self._persist_section(
                resume_id, sec_name,
                content_latex=full_latex,
                content_text=full_text,
                section_type="atomic",
            )

            atomic_rows[sec_name] = self._persist_section_items(
                resume_id, section_id, sec_name, items
            )

        # 3. Embed everything into ChromaDB
        self.chunk_and_embed(resume_id, personal, resume_sections, atomic_rows)

        # 4. Create a default layout row for this resume
        try:
            self.db.add_one("resume_layouts", {
                "resume_id":   resume_id,
                "template_id": "classic",
                "section_order": json.dumps(resume_cnf.get("resume_order", [])),
            })
        except Exception:
            pass  # ignore — layout row may already exist

        return resume_id
    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self, resume_path: Path, user_id: int) -> ParsedResume:
        """
        Parse a resume and persist to SQL + ChromaDB.

        Fast path: .tex files that match the known FontAwesome template are parsed
        with regexes (zero LLM calls, instant).

        Fallback: PDFs, DOCX files, and .tex files from unknown templates are
        converted to plain text and sent to the LLM resume parser once.
        """
        try:
            suffix = resume_path.suffix.lower()
            personal: PersonalInfo
            resume_sections: ResumeSection

            if suffix == ".tex":
                latex_code = resume_path.read_text(encoding="utf-8")
                personal        = self.extract_personal_info(self._strip_comments(latex_code))
                resume_sections = self.parse_sections(latex_code)

                if not self._has_meaningful_sections(resume_sections):
                    log.info("[parser] regex found no sections in %s — using LLM fallback", resume_path.name)
                    raw_text        = self._clean_latex(latex_code)
                    personal, resume_sections = self._llm_parse_resume(raw_text)
                else:
                    log.info("[parser] regex fast-path succeeded for %s", resume_path.name)
            else:
                log.info("[parser] non-LaTeX file (%s) — extracting text and using LLM parser", suffix)
                raw_text        = self._extract_raw_text(resume_path)
                personal, resume_sections = self._llm_parse_resume(raw_text)

            resume_id = None
            if user_id is not None:
                log.info("[parser] storing resume for user_id=%s name=%s", user_id, personal.name)
                resume_id = self.add_to_sql_and_chroma(user_id, resume_path, personal, resume_sections)

            return ParsedResume(
                resume_id=str(resume_id) if resume_id else "N/A",
                resume_path=str(resume_path),
                personal_info=personal,
                resume_sections=resume_sections,
            )

        except Exception as e:
            log.error("[parser] error processing %s: %s", resume_path, e)
            raise