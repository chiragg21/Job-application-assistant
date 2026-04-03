import re
import os
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
    
    def count_lines(self, resume_latex: str|None = None) -> dict:
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
