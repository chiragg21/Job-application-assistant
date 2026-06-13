import os
import re
from typing import List, Tuple, Optional
from pathlib import Path
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
    ) -> int:
        """
        Inserts one row into resume_sections and returns its id.
        Called for every section — flat and atomic alike — so every
        resume_section_items row can carry a valid section_id FK.
        """
        return self.db.add_one("resume_sections", {
            "resume_id": resume_id,
            "section_name": section_name,
            "content_latex": content_latex,
            "content_text": content_text,
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
            )

            atomic_rows[sec_name] = self._persist_section_items(
                resume_id, section_id, sec_name, items
            )

        # 3. Embed everything into ChromaDB
        self.chunk_and_embed(resume_id, personal, resume_sections, atomic_rows)
        return resume_id
    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self, resume_path: Path, user_id: int) -> ParsedResume:
        """Parses a LaTeX resume, persists to SQL, and embeds into ChromaDB."""
        try:
            with open(resume_path, "r", encoding="utf-8") as f:
                latex_code = f.read()

            personal = self.extract_personal_info(self._strip_comments(latex_code))
            resume_sections = self.parse_sections(latex_code)  # strips internally


            if user_id is not None:
                log.info(f"Storing resume for user_id={user_id}, name={personal.name}")
                resume_id = self.add_to_sql_and_chroma(user_id, resume_path, personal, resume_sections)
                
            return ParsedResume(
                resume_id=str(resume_id) if resume_id else "N/A",
                resume_path=str(resume_path),
                personal_info=personal,
                resume_sections=resume_sections,
            )

        except Exception as e:
            log.error(f"Error processing resume {resume_path}: {e}")
            raise