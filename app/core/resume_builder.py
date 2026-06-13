import json
import re
import os
from pathlib import Path

import jinja2

from app.utils.sqlite_handler import SQLHandler
from app.models.resume_builder import ResumeBuilderInput
from config.config import get_config_dict

resume_defaults = get_config_dict()['resume_defaults']

DEFAULT_RESUME_SECTION_ORDER = resume_defaults['resume_order']
DEFAULT_RESUME_FONT_SIZE = resume_defaults['font_size'] 
DEFAULT_RESUME_SECTION_VSPACE = resume_defaults['section_space']
DEFAULT_RESUME_SUBSECTION_VSPACE = resume_defaults['subsection_space']
FLAT_SECTIONS = resume_defaults['flat_sections']
ATOMIC_SECTIONS = resume_defaults['atomic_sections']

SECTION_HEADINGS: dict[str, str] = {
    "education":           "Education",
    "achievements":        "Achievements",
    "skills":              "Technical Skills",
    "relevant_coursework": "Relevant Coursework",
    "experience":          "Experience",
    "projects":            "Projects",
}

DEFAULT_PERSONAL_INFO = {
    "name": "Chirag Garg",
    "email": "chiraggarg18382@gmail.com",
    "linkedin": 'https://www.linkedin.com/in/chirag-garg-382462247',
    'linkedin_name': "Chirag Garg",
    "github": 'https://github.com/chiragg21',
    "github_handle": "chiragg21",
    "phone_number": "+91 62943 38143"
}

class ResumeBuilder:
    def __init__(
        self,
        user_id: int | None,
        sections: ResumeBuilderInput,
        font_size: int | None = None,
    ) -> None:
        self.db = SQLHandler()
        self.personal_info = {}

        personal_info = self.db.fetch_one("users", {'id': user_id})
        if personal_info is None:
            personal_info = {}
        for key in DEFAULT_PERSONAL_INFO.keys():
            self.personal_info[key] = personal_info.get(key, DEFAULT_PERSONAL_INFO.get(key, None))

        self.sections    = sections
        self.font_size   = font_size if font_size is not None else DEFAULT_RESUME_FONT_SIZE
        self.current_resume = ""
        

    def _create_header(self):
        header = f"""
\\documentclass[{self.font_size}pt,a4paper]{{article}}
\\usepackage[a4paper,margin=0.65in]{{geometry}}
\\usepackage{{titlesec}}
\\usepackage{{enumitem}}
\\usepackage{{hyperref}}
\\usepackage{{fontawesome}}
\\usepackage{{xcolor}}
\\usepackage{{parskip}}

\\hypersetup{{colorlinks=true, urlcolor=blue}}

\\setlength{{\\parindent}}{{0pt}}
\\setlength{{\\parskip}}{{0pt}}
\\titleformat{{\\section}}{{\\bfseries\\large}}{{}}{{0em}}{{}}[\\titlerule]
\\setlist[itemize]{{leftmargin=1.5em, itemsep=2pt, topsep=2pt}}

\\pagestyle{{empty}}
\\begin{{document}}

%---------------------- HEADER ----------------------%
\\begin{{center}}
    {{\\Huge \\textbf{{{self.personal_info['name']}}}}}\\\\
    \\vspace{{1mm}}
    \\faEnvelope~\\href{{mailto:{self.personal_info['email']}}}{{{self.personal_info['email']}}} \\quad
    \\faPhone~{self.personal_info['phone_number']} \\quad
    \\faGithub~\\href{{{self.personal_info['github']}}}{{{self.personal_info['github_handle']}}} \\quad
    \\faLinkedin~\\href{{{self.personal_info['linkedin']}}}{{{self.personal_info['linkedin_name']}}}
\\end{{center}}

"""
        return header
    
    def _add_space(self, space):
        return f"\\vspace{{{space}mm}}"

    def build(self, section_order: list[str] = DEFAULT_RESUME_SECTION_ORDER, section_space: float = DEFAULT_RESUME_SECTION_VSPACE, subsection_space: float = DEFAULT_RESUME_SUBSECTION_VSPACE):

        resume = self._create_header()

        for sec in section_order:
            heading = SECTION_HEADINGS.get(sec, sec.replace("_", " ").title())
            heading_latex = f"\\section{{{heading}}}\n"
            if sec in FLAT_SECTIONS:
                content = (getattr(self.sections, sec, "") or "").replace("\\end{document}", "")
                if content.strip():
                    resume += self._add_space(section_space) + heading_latex + content + "\n"
            else:
                items = getattr(self.sections, sec, []) or []
                non_empty = [s.replace("\\end{document}", "") for s in items if s and s.strip()]
                if non_empty:
                    resume += self._add_space(section_space) + heading_latex
                    for i, subsec in enumerate(non_empty):
                        if i > 0:
                            resume += self._add_space(subsection_space)
                        resume += subsec + "\n"

        resume += "\n\\end{document}"

        self.current_resume = resume
        return resume
    
    def count_lines(self, resume_latex: str | None = None) -> dict:
        """
        Compiles the LaTeX to PDF, then uses pdfinfo to get page count,
        and a line-counting TeX package to measure actual used lines.
        Falls back to page-based estimation if compilation fails.
        """
        import tempfile, subprocess

        if resume_latex is None:
            resume_latex = self.current_resume
        
        if not resume_latex:
            raise ValueError("resume_latex cannot be empty")

        # A4 at given font size — empirical lines per page
        LINES_PER_PAGE_BY_FONT = {
            10: 56,
            11: 52,
            12: 47,
        }
        font_size = int(DEFAULT_RESUME_FONT_SIZE)
        lines_per_page = LINES_PER_PAGE_BY_FONT.get(font_size, 52)

        with tempfile.TemporaryDirectory() as tmpdir:
            tex_path = os.path.join(tmpdir, "resume.tex")
            pdf_path = os.path.join(tmpdir, "resume.pdf")

            with open(tex_path, "w") as f:
                f.write(resume_latex)

            try:
                result = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", tmpdir, tex_path],
                    capture_output=True, text=True, timeout=30
                )
                if not os.path.exists(pdf_path):
                    raise RuntimeError(f"pdflatex failed:\n{result.stdout[-1000:]}")

                # Get page dimensions and content height via pdfinfo
                pdfinfo = subprocess.run(
                    ["pdfinfo", pdf_path],
                    capture_output=True, text=True, timeout=10
                )
                
                total_pages = 1
                for line in pdfinfo.stdout.splitlines():
                    if line.startswith("Pages:"):
                        total_pages = int(line.split(":")[1].strip())
                        break

                total_capacity = total_pages * lines_per_page

                # Get actual used height using ghostscript bbox
                gs = subprocess.run(
                    ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=bbox", pdf_path],
                    capture_output=True, text=True, timeout=15
                )

                # Parse %%BoundingBox from last page to estimate used lines on it
                bboxes = re.findall(r'%%BoundingBox: (\d+) (\d+) (\d+) (\d+)', gs.stderr)
                
                # Points per line approximation at given font size (1pt = 1/72 inch)
                POINTS_PER_LINE = font_size * 1.2  # standard leading is 1.2x font size
                PAGE_HEIGHT_POINTS = 841  # A4 in points
                MARGIN_POINTS = int(0.65 * 72 * 2)  # top+bottom margins

                usable_height = PAGE_HEIGHT_POINTS - MARGIN_POINTS

                if bboxes:
                    last_bbox = bboxes[-1]
                    y_min, y_max = int(last_bbox[1]), int(last_bbox[3])
                    used_height_last_page = y_max - y_min
                    used_lines_last_page = used_height_last_page / POINTS_PER_LINE
                    used_lines = (total_pages - 1) * lines_per_page + round(used_lines_last_page, 1)
                else:
                    used_lines = total_pages * lines_per_page  # conservative fallback

                remaining_lines = max(0, round(total_capacity - used_lines, 1))

                return {
                    "total_pages":      total_pages,
                    "total_capacity":   total_capacity,
                    "used_lines":       round(used_lines, 1),
                    "remaining_lines":  remaining_lines,
                    "font_size":        font_size,
                    "lines_per_page":   lines_per_page,
                }

            except (subprocess.TimeoutExpired, FileNotFoundError, RuntimeError) as e:
                # Fallback: rough page-count-only estimate
                return {
                    "total_pages":      None,
                    "total_capacity":   lines_per_page,
                    "used_lines":       None,
                    "remaining_lines":  None,
                    "font_size":        font_size,
                    "lines_per_page":   lines_per_page,
                    "error":            str(e)
                }


# ---------------------------------------------------------------------------
# Template-based renderer (Jinja2 + DB data)
# ---------------------------------------------------------------------------

class TemplateRenderer:
    """
    Render a resume from DB data using a Jinja2 LaTeX template.

    Unlike ResumeBuilder (which accepts pre-assembled strings), this class
    reads all data directly from SQLite using a resume_id, applies layout
    preferences from the resume_layouts table, and renders the chosen template.

    Available templates live in templates/resume/*.tex.j2.
    """

    TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates" / "resume"

    def __init__(self) -> None:
        self._env: jinja2.Environment | None = None

    def _get_env(self) -> jinja2.Environment:
        if self._env is None:
            self._env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(str(self.TEMPLATES_DIR)),
                keep_trailing_newline=True,
                undefined=jinja2.Undefined,
            )
            self._env.filters["enumerate"] = lambda lst: list(enumerate(lst or []))
        return self._env

    def render(
        self,
        resume_id: int,
        template_id: str | None = None,
        section_order: list[str] | None = None,
        font_size: int | None = None,
        margin: float | None = None,
        section_space: float | None = None,
        subsection_space: float | None = None,
    ) -> str:
        """
        Render `resume_id` using the given (or stored) template and layout.

        Caller overrides take precedence over stored layout, which takes
        precedence over global defaults.
        """
        db = SQLHandler()

        resume_row = db.fetch_one("resumes", filters={"id": resume_id})
        if not resume_row:
            raise ValueError(f"Resume {resume_id} not found in DB")

        user_row: dict = db.fetch_one("users", filters={"id": resume_row.get("user_id")}) or {}

        # Layout from DB (resume_layouts table may not exist yet)
        layout: dict = {}
        try:
            layout = db.fetch_one("resume_layouts", filters={"resume_id": resume_id}) or {}
        except Exception:
            pass

        # Resolve params: caller > DB layout > global defaults
        tpl_id        = template_id   or layout.get("template_id")       or "classic"
        sec_order     = section_order or self._parse_json_list(layout.get("section_order")) or DEFAULT_RESUME_SECTION_ORDER
        fsize         = font_size     or layout.get("font_size")          or DEFAULT_RESUME_FONT_SIZE
        marg          = margin        or layout.get("margin")             or 0.65
        sec_sp        = section_space or DEFAULT_RESUME_SECTION_VSPACE
        sub_sp        = subsection_space or DEFAULT_RESUME_SUBSECTION_VSPACE

        # Personal info
        github_url    = user_row.get("github") or DEFAULT_PERSONAL_INFO["github"]
        github_handle = github_url.rstrip("/").split("/")[-1] if github_url else DEFAULT_PERSONAL_INFO["github_handle"]
        linkedin_url  = user_row.get("linkedin") or DEFAULT_PERSONAL_INFO["linkedin"]
        display_name  = user_row.get("name") or resume_row.get("name") or DEFAULT_PERSONAL_INFO["name"]

        personal = {
            "name":          display_name,
            "email":         user_row.get("email") or resume_row.get("email") or DEFAULT_PERSONAL_INFO["email"],
            "phone":         user_row.get("phone_number") or DEFAULT_PERSONAL_INFO["phone_number"],
            "github":        github_url,
            "github_handle": github_handle,
            "linkedin":      linkedin_url,
            "linkedin_name": display_name,
        }

        # Sections
        sections: list[dict] = []
        for sec_key in sec_order:
            heading = SECTION_HEADINGS.get(sec_key, sec_key.replace("_", " ").title())
            if sec_key in FLAT_SECTIONS:
                row     = db.fetch_one("resume_sections", filters={"resume_id": resume_id, "section_name": sec_key})
                content = ((row or {}).get("content_latex") or "").replace("\\end{document}", "").strip()
                if content:
                    sections.append({"key": sec_key, "heading": heading, "type": "flat",
                                     "content": content, "items": None})
            elif sec_key in ATOMIC_SECTIONS:
                rows  = db.execute_raw(
                    "SELECT content_latex FROM resume_section_items "
                    "WHERE resume_id = :rid AND section_name = :sec AND is_latest = 1 "
                    "ORDER BY item_index",
                    {"rid": resume_id, "sec": sec_key},
                ) or []
                items = [r["content_latex"].replace("\\end{document}", "")
                         for r in rows if r.get("content_latex")]
                if items:
                    sections.append({"key": sec_key, "heading": heading, "type": "atomic",
                                     "content": None, "items": items})

        env      = self._get_env()
        template = env.get_template(f"{tpl_id}.tex.j2")
        return template.render(
            font_size=fsize,
            margin=marg,
            section_space=sec_sp,
            subsection_space=sub_sp,
            personal=personal,
            sections=sections,
        )

    @staticmethod
    def _parse_json_list(val) -> list | None:
        if not val:
            return None
        if isinstance(val, list):
            return val
        try:
            return json.loads(val)
        except Exception:
            return None

    def list_templates(self) -> list[str]:
        """Return template IDs (without .tex.j2) available in templates/resume/."""
        return [p.name.replace(".tex.j2", "") for p in self.TEMPLATES_DIR.glob("*.tex.j2")]
