"""
streamlit_app.py — Job Assistant UI

Run with:
    uv run streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests
import streamlit as st

API_BASE = os.getenv("JOB_ASSISTANT_API", "http://localhost:8000")

# ── Resume layout defaults (mirror .env / config.ini) ─────────────────────────
_ALL_SECTIONS = ["education", "achievements", "experience", "projects", "skills", "relevant_coursework"]
_SECTION_LABELS = {
    "education":           "Education",
    "achievements":        "Achievements",
    "experience":          "Experience",
    "projects":            "Projects",
    "skills":              "Technical Skills",
    "relevant_coursework": "Relevant Coursework",
}
_DEFAULT_SECTION_ORDER = ["education", "achievements", "experience", "projects", "skills", "relevant_coursework"]


def _present_sections() -> list[str]:
    """Return only sections that have content in the current parsed resume."""
    pr = st.session_state.get("parsed_resume") or {}
    rs = pr.get("resume_sections") or {}
    present = []
    for sec in _DEFAULT_SECTION_ORDER:
        val = rs.get(sec)
        if val is None:
            continue
        if isinstance(val, list) and len(val) > 0:        # atomic (experience, projects)
            present.append(sec)
        elif isinstance(val, dict) and val.get("content_latex"):  # flat (skills, etc.)
            present.append(sec)
    return present if present else _DEFAULT_SECTION_ORDER.copy()


def _default_filename(ext: str) -> str:
    """Build resume_{candidate}_{company}.{ext} from session state."""
    pr  = st.session_state.get("parsed_resume") or {}
    pi  = pr.get("personal_info") or {}
    pj  = st.session_state.get("parsed_jd") or {}
    raw_name    = pi.get("name", "candidate") or "candidate"
    raw_company = pj.get("company", "company") or "company"
    slug = lambda s: re.sub(r"[^a-zA-Z0-9]+", "_", s.strip()).strip("_") or "unknown"
    return f"resume_{slug(raw_name)}_{slug(raw_company)}.{ext}"

# ── page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Job Assistant",
    page_icon="📄",
    layout="wide",
)

# ── CSS tweaks ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .diff-old  { background:#ffd7d7; padding:4px 8px; border-radius:4px;
                 font-family:monospace; font-size:0.82rem; word-break:break-all; }
    .diff-new  { background:#d7ffd7; padding:4px 8px; border-radius:4px;
                 font-family:monospace; font-size:0.82rem; word-break:break-all; }
    .score-bar { height:12px; border-radius:6px; background:#e0e0e0; margin:4px 0; }
    .score-fill{ height:12px; border-radius:6px; }
    .section-progress { font-size:0.8rem; color:#888; margin-bottom:4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── HTTP session (connection reuse) ────────────────────────────────────────────

@st.cache_resource
def _http_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"Content-Type": "application/json"})
    return s


def api(method: str, path: str, **kwargs):
    """Call the FastAPI backend. Returns (data, error_str)."""
    url = f"{API_BASE}{path}"
    try:
        resp = getattr(_http_session(), method)(url, timeout=180, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return None, detail
    except Exception as exc:
        return None, str(exc)


# ── LaTeX helpers ──────────────────────────────────────────────────────────────

def try_compile_latex(latex: str) -> bytes | None:
    """Compile LaTeX → PDF and return raw bytes, or None on failure."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tex = Path(tmpdir) / "resume.tex"
            tex.write_text(latex, encoding="utf-8")
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode",
                 "-output-directory", tmpdir, str(tex)],
                capture_output=True, timeout=30,
            )
            pdf = Path(tmpdir) / "resume.pdf"
            if pdf.exists():
                return pdf.read_bytes()
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def _compile_latex_cached(latex: str) -> bytes | None:
    """
    Cached wrapper around try_compile_latex.
    Streamlit hashes `latex` as the cache key — identical LaTeX skips pdflatex entirely.
    """
    return try_compile_latex(latex)


def pdf_to_png(pdf_bytes: bytes, dpi: int = 150) -> bytes | None:
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pix = doc[0].get_pixmap(dpi=dpi)
        return pix.tobytes("png")
    except Exception:
        return None


def latex_to_png(latex: str, dpi: int = 150) -> bytes | None:
    pdf = _compile_latex_cached(latex)
    return pdf_to_png(pdf, dpi=dpi) if pdf else None


def latex_panel(label: str, latex: str, key: str):
    if label:
        st.markdown(f"**{label}**")
    if not latex:
        st.info("(no content)")
        return
    png = latex_to_png(latex)
    if png:
        st.image(png, use_container_width=True)
    else:
        st.code(latex, language="latex")


def build_proposed(current: str, lines_to_change: list, suggested_changes: list) -> str:
    proposed = current
    for old, new in zip(lines_to_change, suggested_changes):
        proposed = proposed.replace(old, new, 1)
    return proposed


# ── Score UI helpers ───────────────────────────────────────────────────────────

def score_color(score: float) -> str:
    if score >= 75:
        return "#2ecc71"
    if score >= 50:
        return "#f39c12"
    return "#e74c3c"


def score_bar(label: str, score: float):
    color = score_color(score)
    st.markdown(
        f"""
        <div style="margin-bottom:10px">
          <span style="font-size:0.9rem">{label}</span>
          <span style="float:right;font-weight:bold">{score:.1f}/100</span>
          <div class="score-bar">
            <div class="score-fill" style="width:{score}%;background:{color}"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Parsed JD panel ───────────────────────────────────────────────────────────

def jd_panel(parsed_jd: dict):
    """Render parsed JD in structured expandable sections."""
    if not parsed_jd:
        return
    with st.expander("📋 Parsed Job Description", expanded=False):
        role = parsed_jd.get("job_title") or parsed_jd.get("role") or ""
        company = parsed_jd.get("company") or ""
        if role or company:
            st.markdown(f"**{role}** {'@ ' + company if company else ''}".strip())
            st.divider()

        req = parsed_jd.get("required_skills", [])
        if req:
            st.markdown("**Required Skills**")
            st.markdown(", ".join(req) if isinstance(req, list) else str(req))

        resp = parsed_jd.get("responsibilities", [])
        if resp:
            st.markdown("**Responsibilities**")
            for r in (resp if isinstance(resp, list) else [resp]):
                st.markdown(f"- {r}")

        nth = parsed_jd.get("nice_to_have_skills", [])
        if nth:
            st.markdown("**Nice to Have**")
            st.markdown(", ".join(nth) if isinstance(nth, list) else str(nth))


# ── Generation panel (reusable) ───────────────────────────────────────────────

def generation_panel(thread_id: str, key_prefix: str = "gen"):
    """Generate cover letter / email / outreach using the tailored resume."""
    st.subheader("Generate Documents")

    _type_labels = {
        "coverletter":     "📝 Cover Letter",
        "email":           "📧 HR Email",
        "outreachmessage": "💬 Outreach Message",
    }

    gen_types = st.multiselect(
        "Documents to generate",
        list(_type_labels.keys()),
        default=["coverletter"],
        format_func=lambda x: _type_labels[x],
        key=f"{key_prefix}_types",
    )

    no_jd = st.checkbox(
        "Cold-outreach mode (no specific JD)",
        value=False,
        key=f"{key_prefix}_no_jd",
        help="Use this for cold emails / outreach when you don't have a job posting — "
             "the LLM won't anchor to any JD and will highlight your general strengths.",
    )

    custom_instruction = st.text_area(
        "Custom instructions (optional)",
        value="",
        height=80,
        placeholder="e.g. 'Keep it under 150 words', 'Address the email to Sarah at Acme', "
                    "'Emphasise my ML experience', 'Write in a casual tone'…",
        key=f"{key_prefix}_custom_instruction",
    )

    if st.button("✨ Generate", type="primary", key=f"{key_prefix}_btn"):
        if not gen_types:
            st.warning("Select at least one document type.")
        else:
            with st.spinner("Generating documents (this may take a moment)…"):
                gdata, err = api(
                    "post",
                    f"/edit/{thread_id}/generate",
                    json={
                        "generation_types":   gen_types,
                        "custom_instruction": custom_instruction.strip() or None,
                        "no_jd":              no_jd,
                    },
                )
            if err:
                st.error(err)
            else:
                st.session_state[f"{key_prefix}_results"] = gdata.get("results", {})
                if gdata.get("errors"):
                    st.error(f"Generation errors: {gdata['errors']}")

    results = st.session_state.get(f"{key_prefix}_results", {})
    for doc_type, content in results.items():
        with st.expander(_type_labels.get(doc_type, doc_type), expanded=True):
            st.markdown(content)
            st.download_button(
                f"⬇️ Download {doc_type}.txt",
                data=content,
                file_name=f"{doc_type}.txt",
                mime="text/plain",
                key=f"{key_prefix}_dl_{doc_type}",
            )


# ── Session state initialisation ──────────────────────────────────────────────

def _init():
    defaults: dict = {
        "thread_id":          None,
        "interrupt":          None,
        "stage":              "setup",
        "parsed_jd":          None,
        "parsed_resume":      None,
        "score_result":       None,
        "generation_results": {},
        "preview_latex":      "",
        "error":              None,
        "resume_path":        None,
        "resume_tex":         "",
        "existing_resume_id": None,
        "jd_text":            "",
        "user_id":            1,
        # inline preview shown during section_review
        "inline_preview_latex": "",
        "quick_gen_results":    {},
        # new-upload user-creation flow
        "personal_info_preview":   None,   # PersonalInfo dict from extract-info
        "new_user_created":        False,  # True once create-user succeeds for this upload
        "uploaded_filename":       None,   # tracks which file triggered the current form
        # layout customization (used in refine_or_finish)
        "custom_section_order":    None,   # None → use server default
        "custom_font_size":        11,
        "custom_section_space":    -2.5,
        "custom_subsection_space": 1.0,
        "custom_preview_latex":    "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


_init()

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Job Assistant")

    @st.cache_data(ttl=30)
    def _health():
        return api("get", "/health")

    health, err = _health()
    if health:
        st.success("API online", icon="✅")
    else:
        st.error(f"API offline — {err}")
        st.info("Start the API:\n```\nuv run uvicorn app.api.main:app --reload\n```")
        st.stop()

    st.divider()

    # ── 1. Resume source ───────────────────────────────────────────────────
    st.subheader("1. Resume")

    # Fetch all user profiles (distinct candidates stored in the DB)
    db_users, _  = api("get", "/resume/users")
    db_users     = db_users or []
    db_resumes   = []   # populated below when a user is selected

    resume_source = st.radio(
        "Resume source",
        ["Upload new .tex file"] + (["Use profile from DB"] if db_users else []),
        label_visibility="collapsed",
        key="resume_source_radio",
    )

    if resume_source == "Upload new .tex file":
        uploaded = st.file_uploader(
            "LaTeX resume (.tex)", type=["tex"], label_visibility="collapsed"
        )
        if uploaded:
            # Only process when a genuinely different file is selected
            if uploaded.name != st.session_state.get("uploaded_filename"):
                raw_bytes = uploaded.read()
                tex_str   = raw_bytes.decode("utf-8", errors="replace")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".tex", mode="wb") as tmp:
                    tmp.write(raw_bytes)
                    tmp_resume_path = tmp.name
                st.session_state["resume_path"]        = tmp_resume_path
                st.session_state["existing_resume_id"] = None
                st.session_state["resume_tex"]         = tex_str
                st.session_state["new_user_created"]   = False
                st.session_state["uploaded_filename"]  = uploaded.name
                with st.spinner("Extracting personal info…"):
                    info_data, info_err = api(
                        "post", "/resume/extract-info",
                        json={"resume_path": tmp_resume_path},
                    )
                st.session_state["personal_info_preview"] = info_data if not info_err else {}

        # ── New-user form ──────────────────────────────────────────────────
        if st.session_state.get("resume_path") and not st.session_state.get("new_user_created"):
            pi = st.session_state.get("personal_info_preview") or {}
            st.markdown("**Your profile** — confirm or complete the details below:")
            with st.form("new_user_form", border=True):
                col_a, col_b = st.columns(2)
                with col_a:
                    f_name  = st.text_input("Full name *", value=pi.get("name") or "")
                    f_email = st.text_input("Email",       value=pi.get("email") or "")
                    f_phone = st.text_input("Phone",       value=pi.get("phone") or "")
                with col_b:
                    f_github   = st.text_input("GitHub URL",  value=pi.get("github") or "")
                    f_linkedin = st.text_input("LinkedIn URL", value=pi.get("linkedin") or "")
                    f_role     = st.text_input("Current role (optional)", value="")
                    f_company  = st.text_input("Current company (optional)", value="")
                submitted = st.form_submit_button("✅ Create profile & continue", type="primary")
                if submitted:
                    if not f_name.strip():
                        st.error("Full name is required.")
                    else:
                        user_data, user_err = api(
                            "post", "/resume/create-user",
                            json={
                                "name":         f_name.strip(),
                                "email":        f_email.strip() or None,
                                "phone":        f_phone.strip() or None,
                                "github":       f_github.strip() or None,
                                "linkedin":     f_linkedin.strip() or None,
                                "current_role": f_role.strip() or None,
                                "company":      f_company.strip() or None,
                            },
                        )
                        if user_err:
                            st.error(f"Could not create profile: {user_err}")
                        else:
                            st.session_state["user_id"]        = user_data["user_id"]
                            st.session_state["new_user_created"] = True
                            st.rerun()

        if st.session_state.get("new_user_created"):
            st.success(f"Profile created — user ID {st.session_state['user_id']}")

    else:
        # ── User (candidate) selector ──────────────────────────────────
        user_options = {
            u["user_id"]: (
                f"{u.get('name') or 'User ' + str(u['user_id'])}"
                + (f"  ·  {u['email']}" if u.get("email") else "")
                + f"  ({u.get('resume_count', 1)} resume version{'s' if u.get('resume_count', 1) != 1 else ''})"
            )
            for u in db_users
        }
        chosen_user_id = st.selectbox(
            "Select candidate profile",
            options=list(user_options.keys()),
            format_func=lambda uid: user_options[uid],
            label_visibility="collapsed",
            key="chosen_user_id_select",
        )

        if chosen_user_id != st.session_state.get("user_id"):
            st.session_state["user_id"]            = chosen_user_id
            st.session_state["existing_resume_id"] = None
            st.session_state["resume_path"]        = None
            st.session_state["resume_tex"]         = ""

        # Show all resume versions for this user so they can optionally
        # pick which one to use for personal info (name/email/phone/links).
        db_resumes, _ = api("get", f"/resume/list?user_id={chosen_user_id}")
        db_resumes    = db_resumes or []

        if db_resumes:
            st.caption(
                f"**{len(db_resumes)} resume version(s)** found for this profile. "
                "The system will automatically pick the most JD-relevant content "
                "from all versions.  \n"
                "Choose below to override which version supplies the personal info "
                "(name, email, phone, links)."
            )
            version_options = {
                r["resume_id"]: (
                    f"#{r['resume_id']}"
                    + (f"  ·  {r['created_at'][:10]}" if r.get("created_at") else "")
                    + "  (latest)" if r == db_resumes[0] else f"  #{r['resume_id']}"
                    + (f"  ·  {r['created_at'][:10]}" if r.get("created_at") else "")
                )
                for r in db_resumes
            }
            # Simpler label
            version_options = {
                r["resume_id"]: (
                    ("✦ Latest  " if i == 0 else f"  v{r['resume_id']}  ")
                    + (r["created_at"][:10] if r.get("created_at") else "")
                )
                for i, r in enumerate(db_resumes)
            }
            chosen_ver = st.selectbox(
                "Personal info source",
                options=list(version_options.keys()),
                format_func=lambda rid: version_options[rid],
                label_visibility="collapsed",
                key="chosen_resume_ver_select",
            )
            if chosen_ver != st.session_state.get("existing_resume_id"):
                st.session_state["existing_resume_id"] = chosen_ver
                st.session_state["resume_path"]        = None
                chosen_row = next((r for r in db_resumes if r["resume_id"] == chosen_ver), None)
                if chosen_row:
                    rp = chosen_row.get("resume_path", "")
                    try:
                        st.session_state["resume_tex"] = Path(rp).read_text(encoding="utf-8")
                    except Exception:
                        st.session_state["resume_tex"] = ""

    if st.session_state.get("resume_tex"):
        with st.expander("Preview resume source", expanded=False):
            png = latex_to_png(st.session_state["resume_tex"])
            if png:
                st.image(png, use_container_width=True)
            else:
                st.code(st.session_state["resume_tex"], language="latex")

    st.divider()

    # ── 2. Paste JD ────────────────────────────────────────────────────────
    st.subheader("2. Job Description")
    st.session_state["jd_text"] = st.text_area(
        "Paste the full JD", value=st.session_state["jd_text"],
        height=220, label_visibility="collapsed",
    )

    st.divider()

    # ── Start / Reset ──────────────────────────────────────────────────────
    if st.button("🔍 Parse JD & Resume", type="primary", use_container_width=True):
        jt          = st.session_state.get("jd_text", "").strip()
        rp          = st.session_state.get("resume_path")
        existing_id = st.session_state.get("existing_resume_id")
        uid         = st.session_state.get("user_id", 1)

        if not jt:
            st.error("Paste a job description first.")
        elif not rp and not existing_id and not db_users:
            st.error("Upload a .tex resume or select a saved profile.")
        elif rp and not st.session_state.get("new_user_created"):
            st.error("Complete the profile form above before continuing.")
        else:
            payload: dict = {"user_id": uid, "jd_text": jt}
            if rp:
                payload["resume_path"] = rp
            elif existing_id:
                payload["existing_resume_id"] = existing_id
            # else: user-profile mode — graph loads latest resume automatically

            with st.spinner("Parsing JD + resume, retrieving relevant sections…"):
                data, err = api("post", "/edit/start", json=payload)

            if err:
                st.error(err)
            else:
                # Reset all session state for the new session
                for k in ("inline_preview_latex", "gen_results", "refine_gen_results",
                          "done_gen_results", "preview_latex", "score_result",
                          "generation_results", "error"):
                    st.session_state[k] = "" if k in ("inline_preview_latex", "preview_latex") else (
                        {} if "results" in k or k == "generation_results" else None
                    )
                st.session_state.update(
                    thread_id=data["thread_id"],
                    interrupt=data.get("interrupt"),
                    stage="running",
                )
                st.rerun()

    if st.session_state["thread_id"]:
        st.caption(f"Session `{st.session_state['thread_id'][:8]}…`")


# ── Main content ───────────────────────────────────────────────────────────────

thread_id = st.session_state["thread_id"]
interrupt  = st.session_state["interrupt"]


def send_resume(response: dict):
    """POST the user's response and advance the graph."""
    data, err = api("post", f"/edit/{thread_id}/resume", json={"response": response})
    if err:
        st.session_state["error"] = err
        st.rerun()
        return
    new_interrupt = data.get("interrupt")
    st.session_state["interrupt"] = new_interrupt
    st.session_state["stage"]     = data.get("stage") or ""

    # Only pay the extra GET cost when the session is finishing
    # (no more interrupts) or when we don't yet have parsed_jd cached.
    need_state = (new_interrupt is None) or (not st.session_state.get("parsed_jd"))
    if need_state:
        state_data, _ = api("get", f"/edit/{thread_id}/state")
        if state_data:
            st.session_state["score_result"]       = state_data.get("score_result")
            st.session_state["generation_results"] = state_data.get("generation_results") or {}
            st.session_state["parsed_jd"]          = state_data.get("parsed_jd")
            st.session_state["parsed_resume"]      = state_data.get("parsed_resume")
    st.rerun()


# ── Error banner ───────────────────────────────────────────────────────────────

if st.session_state.get("error"):
    st.error(st.session_state["error"])
    if st.button("Dismiss"):
        st.session_state["error"] = None
        st.rerun()

# ── Welcome screen ─────────────────────────────────────────────────────────────

if not thread_id:
    st.title("Resume Tailoring Assistant")

    # ── Quick Generate tab is available whenever any DB profiles exist ────────
    _jd_ready   = bool(st.session_state.get("jd_text", "").strip())
    _can_quick  = bool(db_users)

    if _can_quick:
        st.markdown("---")
        tab_quick, tab_edit_info = st.tabs(["✨ Quick Generate", "📖 How it works"])
    else:
        tab_quick     = None
        tab_edit_info = None

    # ── Quick Generate tab ────────────────────────────────────────────────────
    if _can_quick and tab_quick is not None:
        with tab_quick:
            st.markdown(
                "Generate a cover letter, HR email, or outreach message **right now** "
                "without going through the full editing flow.  \n"
                "The system uses JD-embedding similarity to automatically pick the best "
                "version of each section from your saved resume history."
            )
            st.info(
                "To tailor your resume section-by-section first, click **🚀 Start / Reset Session** "
                "in the sidebar instead."
            )

            _qg_type_labels = {
                "coverletter":     "📝 Cover Letter",
                "email":           "📧 HR Email",
                "outreachmessage": "💬 Outreach Message",
            }
            _qg_types = st.multiselect(
                "Documents to generate",
                list(_qg_type_labels.keys()),
                default=["coverletter"],
                format_func=lambda x: _qg_type_labels[x],
                key="welcome_qg_types",
            )
            _qg_no_jd = st.checkbox(
                "Cold-outreach mode (no specific JD)",
                value=False,
                key="welcome_qg_no_jd",
                help="Generate without anchoring to a job description.",
            )
            _qg_custom = st.text_area(
                "Custom instructions (optional)", value="", height=68,
                placeholder="e.g. 'Keep it under 150 words', 'Address Sarah at Acme'…",
                key="welcome_qg_custom",
            )

            _jd_ready_or_cold = _jd_ready or _qg_no_jd
            if not _jd_ready_or_cold:
                st.info("Paste a job description in the sidebar, or enable cold-outreach mode.")

            if st.button("✨ Generate Now", type="primary", use_container_width=True, disabled=not _jd_ready_or_cold):
                if not _qg_types:
                    st.warning("Select at least one document type.")
                else:
                    with st.spinner("Parsing JD · selecting best resume sections · generating…"):
                        gdata, err = api(
                            "post",
                            "/edit/quick-generate",
                            json={
                                "user_id":            st.session_state["user_id"],
                                "jd_text":            st.session_state["jd_text"] if not _qg_no_jd else None,
                                "resume_id":          st.session_state.get("existing_resume_id"),
                                "generation_types":   _qg_types,
                                "custom_instruction": _qg_custom.strip() or None,
                                "no_jd":              _qg_no_jd,
                            },
                        )
                    if err:
                        st.error(err)
                    else:
                        st.session_state["quick_gen_results"] = gdata.get("results", {})
                        if gdata.get("parsed_jd"):
                            st.session_state["parsed_jd"] = gdata["parsed_jd"]
                        if gdata.get("errors"):
                            st.warning(f"Partial errors: {gdata['errors']}")
                        st.rerun()

            # Show results
            _qg_results = st.session_state.get("quick_gen_results", {})
            if _qg_results:
                st.markdown("---")
                st.subheader("Generated Documents")
                _type_labels = {
                    "coverletter":     "📝 Cover Letter",
                    "email":           "📧 HR Email",
                    "outreachmessage": "💬 Outreach Message",
                }
                for _doc_type, _content in _qg_results.items():
                    _title = _type_labels.get(_doc_type, _doc_type)
                    with st.expander(_title, expanded=True):
                        st.markdown(_content)
                        st.download_button(
                            f"⬇️ Download {_doc_type}.txt",
                            data=_content,
                            file_name=f"{_doc_type}.txt",
                            mime="text/plain",
                            key=f"qg_dl_{_doc_type}",
                        )

    # ── How it works (always visible) ─────────────────────────────────────────
    _how_it_works = """
### How it works

1. **Upload** your `.tex` resume **or select a saved profile** in the sidebar
2. **Paste** the job description
3. **Quick Generate** → get cover letter / email / outreach message instantly using JD-based section selection
   — OR —
   **Start Session** → tailor each section interactively with AI suggestions
4. **Review** each LLM suggestion side-by-side with your current text
5. See your **score**, refine if needed, then **download** the final resume
6. Generate documents at any point during or after editing
"""
    _latex_reqs = """
### LaTeX Preview

The app will try to compile your resume to PDF and show a rendered preview.

**Requirements:**
- [MiKTeX](https://miktex.org/) or [TeX Live](https://tug.org/texlive/) must be installed
  and `pdflatex` must be on your `PATH`
- If not available, the raw LaTeX source is shown with syntax highlighting instead
"""
    if _can_quick and tab_edit_info is not None:
        with tab_edit_info:
            col_l, col_r = st.columns(2)
            col_l.markdown(_how_it_works)
            col_r.markdown(_latex_reqs)
    else:
        col_l, col_r = st.columns(2)
        col_l.markdown(_how_it_works)
        col_r.markdown(_latex_reqs)

    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# INTERRUPT: item_selection
# ═══════════════════════════════════════════════════════════════════════════════

if interrupt and interrupt.get("type") == "item_selection":
    st.title("Sections Retrieved")
    st.markdown(
        "These sections were selected as most relevant to the job description. "
        "Un-check any you want to exclude, then choose what to do next."
    )

    # Show parsed JD
    jd_panel(st.session_state.get("parsed_jd") or {})

    ranked   = interrupt.get("ranked_items", [])
    selected: list[dict] = []

    for i, item in enumerate(ranked):
        sec       = item.get("section_name", "")
        name      = item.get("item_name") or ""
        rows      = item.get("rows", [])
        top_score = rows[0].get("score", 0.0) if rows else 0.0
        content   = rows[0].get("content_latex", "") if rows else ""
        label     = f"**{sec}**" + (f"  —  {name}" if name else "")

        c1, c2, c3 = st.columns([0.5, 5, 1])
        with c1:
            checked = st.checkbox("", value=True, key=f"sel_{i}", label_visibility="collapsed")
        with c2:
            st.markdown(label)
        with c3:
            bar_color = score_color(top_score * 100)
            st.markdown(
                f'<span style="color:{bar_color};font-weight:bold">{top_score:.2f}</span>',
                unsafe_allow_html=True,
            )

        with st.expander("Preview content", expanded=False):
            st.code(content or "(no content)", language="latex")

        if checked:
            selected.append(item)

        st.divider()

    # ── Choose: Quick Generate or Start Editing ────────────────────────────
    st.markdown("### What would you like to do?")
    tab_qg, tab_edit = st.tabs(["✨ Quick Generate", "📝 Start Editing Session"])

    with tab_qg:
        st.markdown(
            "Generate a cover letter, HR email, or outreach message right now "
            "using the checked sections above — no full editing session needed."
        )

        _is_type_labels = {
            "coverletter":     "📝 Cover Letter",
            "email":           "📧 HR Email",
            "outreachmessage": "💬 Outreach Message",
        }
        _is_qg_types = st.multiselect(
            "Documents to generate",
            list(_is_type_labels.keys()),
            default=["coverletter"],
            format_func=lambda x: _is_type_labels[x],
            key="is_qg_types",
        )
        _is_no_jd = st.checkbox(
            "Cold-outreach mode (no specific JD)",
            value=False,
            key="is_qg_no_jd",
            help="Generate without anchoring to a job description.",
        )
        _is_custom = st.text_area(
            "Custom instructions (optional)", value="", height=68,
            placeholder="e.g. 'Keep it under 150 words', 'Address Sarah at Acme'…",
            key="is_qg_custom",
        )

        if st.button("✨ Generate Now", type="primary", key="is_qg_btn"):
            if not _is_qg_types:
                st.warning("Select at least one document type.")
            elif not selected:
                st.warning("Check at least one section above.")
            else:
                with st.spinner("Building resume · generating documents…"):
                    gdata, err = api(
                        "post",
                        f"/edit/{thread_id}/generate-from-items",
                        json={
                            "generation_types":   _is_qg_types,
                            "selected_items":     selected,
                            "custom_instruction": _is_custom.strip() or None,
                            "no_jd":              _is_no_jd,
                        },
                    )
                if err:
                    st.error(err)
                else:
                    st.session_state["quick_gen_results"] = gdata.get("results", {})
                    if gdata.get("errors"):
                        st.warning(f"Partial errors: {gdata['errors']}")
                    st.rerun()

        _is_qg_results = st.session_state.get("quick_gen_results", {})
        if _is_qg_results:
            st.markdown("---")
            st.subheader("Generated Documents")
            _type_labels = {
                "coverletter":     "📝 Cover Letter",
                "email":           "📧 HR Email",
                "outreachmessage": "💬 Outreach Message",
            }
            for _doc_type, _content in _is_qg_results.items():
                with st.expander(_type_labels.get(_doc_type, _doc_type), expanded=True):
                    st.markdown(_content)
                    st.download_button(
                        f"⬇️ Download {_doc_type}.txt",
                        data=_content,
                        file_name=f"{_doc_type}.txt",
                        mime="text/plain",
                        key=f"is_qg_dl_{_doc_type}",
                    )

    with tab_edit:
        st.markdown(
            "Walk through each section with AI suggestions, review changes, "
            "refine as needed, and download the final resume."
        )
        if st.button("📝 Start Editing Session", type="primary", key="is_confirm_edit"):
            with st.spinner("Building edit state & generating suggestions…"):
                send_resume({"selected_items": selected})


# ═══════════════════════════════════════════════════════════════════════════════
# INTERRUPT: section_review
# ═══════════════════════════════════════════════════════════════════════════════

elif interrupt and interrupt.get("type") == "section_review":
    section    = interrupt.get("section", "")
    item       = interrupt.get("item") or ""
    current    = interrupt.get("current_latex", "")
    lines      = interrupt.get("lines_to_change", [])
    changes    = interrupt.get("suggested_changes", [])
    can_back   = interrupt.get("can_go_back", False)
    sec_idx    = interrupt.get("section_index", 0)
    total_secs = interrupt.get("total_sections", 1)
    history    = interrupt.get("history", [])   # [{node_id, action, content}, ...]

    # _wkey scopes all widget keys to this section+history snapshot so that
    # widgets reset automatically when a new LLM result arrives.
    _wkey = f"{sec_idx}_{len(history)}"

    # Read which individual changes the user has checked (default: all selected).
    # We read from session_state here — before the checkboxes are rendered —
    # so that the proposed preview below already reflects the current selection.
    selected_indices = [
        i for i in range(len(lines))
        if st.session_state.get(f"change_{_wkey}_{i}", True)
    ]

    # Build proposed preview using only the currently selected changes.
    proposed = build_proposed(
        current,
        [lines[i] for i in selected_indices],
        [changes[i] for i in selected_indices],
    )

    heading = f"Review: **{section}**" + (f"  —  {item}" if item else "")

    # ── Progress bar ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="section-progress">Section {sec_idx + 1} of {total_secs}</div>',
        unsafe_allow_html=True,
    )
    st.progress((sec_idx) / max(total_secs, 1))
    st.title("Section Review")
    st.markdown(heading)

    # ── Full-resume inline preview + JD ───────────────────────────────────
    col_pv, col_jd = st.columns([1, 1])
    with col_pv:
        if st.button("👁️ Load Full Resume Preview", key="load_preview_btn"):
            with st.spinner("Building preview…"):
                prev_data, prev_err = api("get", f"/edit/{thread_id}/preview")
            if prev_err:
                st.error(prev_err)
            else:
                st.session_state["inline_preview_latex"] = prev_data.get("latex", "")
                st.rerun()
    with col_jd:
        parsed_jd = st.session_state.get("parsed_jd") or {}
        if parsed_jd:
            jd_panel(parsed_jd)

    inline_latex = st.session_state.get("inline_preview_latex", "")
    if inline_latex:
        with st.expander("Full Resume Preview (current state)", expanded=True):
            c_pv, c_close = st.columns([6, 1])
            with c_close:
                if st.button("✕ Close", key="close_preview_btn"):
                    st.session_state["inline_preview_latex"] = ""
                    st.rerun()
            with c_pv:
                latex_panel("", inline_latex, key="inline_preview")

    st.divider()

    # ── Edit History (versions for this section) ──────────────────────────
    # history[0] = original, history[-1] = current after last edit
    if len(history) > 1:
        with st.expander(f"📜 Edit History ({len(history)} versions — click to restore any)", expanded=False):
            action_labels = {
                "root":               "Original",
                "resume_suggestion":  "AI Suggestion",
                "paraphrase":         "Paraphrase",
                "reject":             "Reverted",
                "another_suggestion": "Alternative",
                "custom_instruction": "Custom edit",
            }
            for entry in history:
                nid     = entry["node_id"]
                act_raw = entry.get("action", "root")
                label   = next((v for k, v in action_labels.items() if k in act_raw), act_raw)
                is_current = (entry == history[-1])

                col_lbl, col_content, col_btn = st.columns([1, 5, 1])
                with col_lbl:
                    badge = "🟢 **Current**" if is_current else f"v{nid}"
                    st.markdown(badge)
                with col_content:
                    st.markdown(f"_{label}_")
                    st.code(entry["content"][:400] + ("…" if len(entry["content"]) > 400 else ""),
                            language="latex")
                with col_btn:
                    if not is_current:
                        if st.button(f"↩ Restore", key=f"restore_{nid}"):
                            with st.spinner("Restoring…"):
                                send_resume({"proposal": "restore", "node_id": nid})

    # ── Side-by-side preview ───────────────────────────────────────────────
    col_cur, col_new = st.columns(2, gap="large")

    with col_cur:
        st.markdown("### Current Version")
        st.code(current, language="latex")

    with col_new:
        st.markdown("### Proposed Changes")
        if proposed != current:
            st.code(proposed, language="latex")
        else:
            st.info("No changes suggested for this section.")

    # ── Diff table with per-change checkboxes ─────────────────────────────
    if lines:
        st.markdown("#### Suggested Changes — check the ones to apply")
        st.caption(f"{len(selected_indices)} of {len(lines)} changes selected")
        hdr0, hdr1, hdr2 = st.columns([0.5, 3, 3])
        hdr0.markdown("**Apply?**")
        hdr1.markdown("**Before**")
        hdr2.markdown("**After**")
        for i, (old_l, new_l) in enumerate(zip(lines, changes)):
            c0, c1, c2 = st.columns([0.5, 3, 3])
            with c0:
                st.checkbox(
                    "",
                    value=True,
                    key=f"change_{_wkey}_{i}",
                    label_visibility="collapsed",
                )
            with c1:
                st.markdown(f'<div class="diff-old">{old_l}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="diff-new">{new_l}</div>', unsafe_allow_html=True)
    else:
        st.info("No specific line suggestions — accept keeps the current version.")

    # ── Decision controls ──────────────────────────────────────────────────
    st.divider()

    # Detect if the current state is a result of a previous LLM action by
    # checking whether history has more than 2 entries (root + suggestion + ≥1 edit)
    has_prior_llm_edit = len(history) > 2

    st.markdown("### Your Decision")
    st.caption(
        "**Paraphrase**, **Custom instruction**, and **Another suggestion** call the LLM and "
        "show the result here for review — they do **not** advance to the next section."
    )

    options = ["accept", "reject", "paraphrase", "custom_instruction"]
    if has_prior_llm_edit:
        options.append("another_suggestion")

    action = st.radio(
        "Action",
        options,
        format_func=lambda x: {
            "accept":             "✅ Accept — move to next section",
            "reject":             "↩️ Revert to original",
            "paraphrase":         "🔄 Paraphrase (improve wording)",
            "custom_instruction": "✏️ Custom instruction to LLM",
            "another_suggestion": "🔁 Ask for another suggestion",
        }[x],
        horizontal=True,
        key=f"section_action_{_wkey}",
    )

    special_instruction = None
    rejection_reason    = None

    if action == "custom_instruction":
        special_instruction = st.text_area(
            "Custom instruction",
            placeholder="e.g. 'Make this more concise and quantify achievements'",
            key=f"custom_instr_{_wkey}",
        )
    elif action == "reject":
        rejection_reason = st.text_input(
            "Reason (optional — helps the LLM if you refine later)",
            key=f"rej_reason_{_wkey}",
        )
    elif action == "another_suggestion":
        rejection_reason = st.text_input(
            "What was wrong with the last suggestion? (optional)",
            key=f"another_reason_{_wkey}",
        )

    # Submit row: main action button + back navigation
    col_submit, col_back = st.columns([3, 1])
    with col_submit:
        if st.button("Submit →", type="primary", use_container_width=True):
            response: dict = {
                "proposal":            action,
                "special_instruction": special_instruction,
                "rejection_reason":    rejection_reason,
            }
            # For accept, tell the graph exactly which changes to apply.
            # Omit the key entirely when all changes are selected (full accept)
            # so the graph takes the fast path with no rebuild.
            if action == "accept" and lines and len(selected_indices) < len(lines):
                response["selected_indices"] = selected_indices
            with st.spinner("Applying edit…"):
                send_resume(response)
    with col_back:
        if st.button(
            "← Previous section",
            disabled=not can_back,
            use_container_width=True,
            help="Go back to re-review the previous section" if can_back else "Already at first section",
        ):
            with st.spinner("Going back…"):
                send_resume({"proposal": "back"})


# ═══════════════════════════════════════════════════════════════════════════════
# INTERRUPT: refine_or_finish
# ═══════════════════════════════════════════════════════════════════════════════

elif interrupt and interrupt.get("type") == "refine_or_finish":
    st.title("Editing Complete")
    score_data     = interrupt.get("score_result") or {}
    score_feedback = interrupt.get("score_feedback") or ""

    # ── Score dashboard ────────────────────────────────────────────────────
    if score_data:
        overall = score_data.get("overall_score", 0)

        st.markdown(f"## Overall Score: {overall:.1f} / 100")
        overall_color = score_color(overall)
        st.markdown(
            f'<div class="score-bar"><div class="score-fill" style="width:{overall}%;'
            f'background:{overall_color}"></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("")

        c1, c2, c3 = st.columns(3)
        c1.metric("Keyword Match",    f"{score_data.get('keyword_match_score', 0):.1f}")
        c2.metric("ATS Friendliness", f"{score_data.get('ats_friendliness_score', 0):.1f}")
        c3.metric("Resume Quality",   f"{score_data.get('resume_quality_score', 0):.1f}")

        with st.expander("Dimension details", expanded=True):
            for dim in score_data.get("dimension_scores", []):
                score_bar(dim.get("dimension", ""), dim.get("score", 0))
                st.markdown(f"_{dim.get('reasoning', '')}_")
                for s in dim.get("suggestions", []):
                    st.markdown(f"- {s}")
                st.markdown("")

        if score_data.get("missing_keywords"):
            st.warning("**Missing keywords:** " + ", ".join(score_data["missing_keywords"]))

        if score_data.get("overall_feedback"):
            st.info(score_data["overall_feedback"])

    # ── Preview + Generate at this stage ──────────────────────────────────
    tab_choice, tab_preview, tab_layout, tab_generate = st.tabs(
        ["🎯 Refine / Finish", "👁️ Resume Preview", "🎨 Customize Layout", "✉️ Generate Documents"]
    )

    with tab_choice:
        st.divider()
        st.markdown(
            "**Refine** re-runs the LLM with score feedback and walks you through "
            "all sections again. **Finish** finalises the resume."
        )
        col_r, col_f = st.columns(2)
        with col_r:
            if st.button("🔄 Refine (re-run with score feedback)", use_container_width=True):
                with st.spinner("Generating improved suggestions…"):
                    send_resume({"choice": "refine"})
        with col_f:
            if st.button("✅ Finish", type="primary", use_container_width=True):
                with st.spinner("Finishing…"):
                    send_resume({"choice": "finish"})

    with tab_preview:
        if st.button("Build Preview", key="refine_build_preview"):
            with st.spinner("Compiling LaTeX…"):
                prev, err = api("get", f"/edit/{thread_id}/preview")
            if err:
                st.error(err)
            else:
                st.session_state["preview_latex"] = prev.get("latex", "")

        latex = st.session_state.get("preview_latex", "")
        if latex:
            _fname = st.text_input(
                "File name", value=_default_filename("tex"),
                key="dl_fname_preview",
                help="Change the file name before downloading",
            )
            _base = _fname.rsplit(".", 1)[0] if "." in _fname else _fname
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "⬇️ Download .tex", data=latex,
                    file_name=f"{_base}.tex", mime="text/plain",
                )
            with col_dl2:
                pdf_bytes = _compile_latex_cached(latex)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download .pdf", data=pdf_bytes,
                        file_name=f"{_base}.pdf", mime="application/pdf",
                    )
            png = latex_to_png(latex)
            if png:
                st.image(png, use_container_width=True)
            else:
                st.code(latex, language="latex")
        else:
            st.info("Click **Build Preview** to compile the resume.")

    with tab_layout:
        st.markdown(
            "Adjust layout settings and click **Build Preview** to see changes "
            "rendered live.  These settings apply only to the preview / download "
            "— they do not affect the stored edit state."
        )
        st.divider()

        # ── Initialise custom order from present sections on first visit ──────
        if st.session_state["custom_section_order"] is None:
            st.session_state["custom_section_order"] = _present_sections()

        _order = st.session_state["custom_section_order"]

        # ── Section order controls ─────────────────────────────────────────
        st.markdown("**Section Order**")
        st.caption("Use ↑ / ↓ to reorder sections.")
        for _i, _sec in enumerate(_order):
            _col_lbl, _col_up, _col_dn = st.columns([5, 0.6, 0.6])
            _col_lbl.markdown(f"{_i + 1}. {_SECTION_LABELS.get(_sec, _sec)}")
            if _i > 0 and _col_up.button("↑", key=f"lo_up_{_i}", help="Move up"):
                _order[_i], _order[_i - 1] = _order[_i - 1], _order[_i]
                st.session_state["custom_section_order"] = _order
                st.rerun()
            if _i < len(_order) - 1 and _col_dn.button("↓", key=f"lo_dn_{_i}", help="Move down"):
                _order[_i], _order[_i + 1] = _order[_i + 1], _order[_i]
                st.session_state["custom_section_order"] = _order
                st.rerun()

        if st.button("↺ Reset order", key="lo_reset_order"):
            st.session_state["custom_section_order"] = _present_sections()
            st.rerun()

        st.divider()

        # ── Spacing + font controls ────────────────────────────────────────
        st.markdown("**Font & Spacing**")
        _lo_col1, _lo_col2, _lo_col3 = st.columns(3)
        with _lo_col1:
            st.session_state["custom_font_size"] = st.select_slider(
                "Font size (pt)", options=[10, 11, 12],
                value=st.session_state["custom_font_size"],
                key="lo_font_size",
            )
        with _lo_col2:
            st.session_state["custom_section_space"] = st.slider(
                "Section spacing (mm)", min_value=-5.0, max_value=5.0,
                value=float(st.session_state["custom_section_space"]),
                step=0.5, key="lo_sec_space",
            )
        with _lo_col3:
            st.session_state["custom_subsection_space"] = st.slider(
                "Item spacing (mm)", min_value=0.0, max_value=8.0,
                value=float(st.session_state["custom_subsection_space"]),
                step=0.5, key="lo_subsec_space",
            )

        st.divider()

        # ── Build preview with custom settings ────────────────────────────
        if st.button("🔄 Build Preview with these settings", type="primary", key="lo_build_preview"):
            with st.spinner("Compiling LaTeX…"):
                _lo_prev, _lo_err = api(
                    "get",
                    f"/edit/{thread_id}/preview",
                    params={
                        "section_order":    ",".join(st.session_state["custom_section_order"]),
                        "font_size":        st.session_state["custom_font_size"],
                        "section_space":    st.session_state["custom_section_space"],
                        "subsection_space": st.session_state["custom_subsection_space"],
                    },
                )
            if _lo_err:
                st.error(_lo_err)
            else:
                st.session_state["custom_preview_latex"] = _lo_prev.get("latex", "")
                st.rerun()

        _lo_latex = st.session_state.get("custom_preview_latex", "")
        if _lo_latex:
            _lo_fname = st.text_input(
                "File name", value=_default_filename("tex"),
                key="dl_fname_layout",
                help="Change the file name before downloading",
            )
            _lo_base = _lo_fname.rsplit(".", 1)[0] if "." in _lo_fname else _lo_fname
            _lo_dl_col1, _lo_dl_col2 = st.columns(2)
            with _lo_dl_col1:
                st.download_button(
                    "⬇️ Download .tex", data=_lo_latex,
                    file_name=f"{_lo_base}.tex", mime="text/plain",
                    key="lo_dl_tex",
                )
            with _lo_dl_col2:
                _lo_pdf = _compile_latex_cached(_lo_latex)
                if _lo_pdf:
                    st.download_button(
                        "⬇️ Download .pdf", data=_lo_pdf,
                        file_name=f"{_lo_base}.pdf", mime="application/pdf",
                        key="lo_dl_pdf",
                    )
            _lo_png = latex_to_png(_lo_latex)
            if _lo_png:
                st.image(_lo_png, use_container_width=True)
            else:
                st.code(_lo_latex, language="latex")
        else:
            st.info("Adjust settings above and click **Build Preview** to see the result.")

    with tab_generate:
        generation_panel(thread_id, key_prefix="refine_gen")


# ═══════════════════════════════════════════════════════════════════════════════
# DONE / no interrupt
# ═══════════════════════════════════════════════════════════════════════════════

elif not interrupt and thread_id:
    stage = st.session_state.get("stage", "")

    if stage and stage not in ("finish", "done", "error"):
        st.title("Processing…")
        st.markdown(f"Current stage: **{stage}**")
        if st.button("🔄 Refresh"):
            state_data, _ = api("get", f"/edit/{thread_id}/state")
            if state_data:
                st.session_state["stage"]         = state_data.get("stage") or ""
                st.session_state["parsed_jd"]     = state_data.get("parsed_jd")
                st.session_state["parsed_resume"] = state_data.get("parsed_resume")
            st.rerun()
        st.stop()

    # ── Session finished ───────────────────────────────────────────────────
    st.title("Session Complete 🎉")

    tab_preview, tab_score, tab_generate = st.tabs(
        ["📄 Resume Preview", "📊 Score", "✉️ Generate Documents"]
    )

    # ── Preview tab ─────────────────────────────────────────────────────────
    with tab_preview:
        st.subheader("Tailored Resume Preview")
        col_load, col_dl = st.columns([1, 1])

        with col_load:
            if st.button("Build Preview"):
                with st.spinner("Compiling LaTeX…"):
                    prev, err = api("get", f"/edit/{thread_id}/preview")
                if err:
                    st.error(err)
                else:
                    st.session_state["preview_latex"] = prev.get("latex", "")

        latex = st.session_state.get("preview_latex", "")

        with col_dl:
            if latex:
                _done_fname = st.text_input(
                    "File name", value=_default_filename("tex"),
                    key="dl_fname_done",
                    help="Change the file name before downloading",
                )
                _done_base = _done_fname.rsplit(".", 1)[0] if "." in _done_fname else _done_fname
                st.download_button(
                    "⬇️ Download .tex", data=latex,
                    file_name=f"{_done_base}.tex", mime="text/plain",
                )
                pdf_bytes = _compile_latex_cached(latex)
                if pdf_bytes:
                    st.download_button(
                        "⬇️ Download .pdf", data=pdf_bytes,
                        file_name=f"{_done_base}.pdf", mime="application/pdf",
                    )

        if latex:
            view_col, src_col = st.columns([3, 2], gap="large")
            with view_col:
                st.markdown("**Rendered Preview**")
                png = latex_to_png(latex)
                if png:
                    st.image(png, use_container_width=True)
                else:
                    st.info(
                        "pdflatex not found — install MiKTeX or TeX Live to see a "
                        "rendered preview. The source is shown on the right."
                    )
            with src_col:
                st.markdown("**LaTeX Source**")
                st.code(latex, language="latex")
        else:
            st.info("Click **Build Preview** to compile the resume.")

    # ── Score tab ────────────────────────────────────────────────────────────
    with tab_score:
        st.subheader("Resume Score")
        score_result = st.session_state.get("score_result")

        btn_label = "Re-score" if score_result else "Score Resume"
        if st.button(btn_label):
            with st.spinner("Scoring against JD…"):
                sdata, err = api("post", f"/edit/{thread_id}/score", json={})
            if err:
                st.error(err)
            else:
                st.session_state["score_result"] = sdata.get("score_result")
                score_result = st.session_state["score_result"]
                st.rerun()

        if score_result:
            overall = score_result.get("overall_score", 0)
            st.markdown(f"## Overall: {overall:.1f} / 100")
            score_bar("Overall Score", overall)

            c1, c2, c3 = st.columns(3)
            km  = score_result.get("keyword_match_score", 0)
            ats = score_result.get("ats_friendliness_score", 0)
            rq  = score_result.get("resume_quality_score", 0)
            c1.metric("Keyword Match",    f"{km:.1f}")
            c2.metric("ATS Friendliness", f"{ats:.1f}")
            c3.metric("Resume Quality",   f"{rq:.1f}")

            score_bar("Keyword Match",    km)
            score_bar("ATS Friendliness", ats)
            score_bar("Resume Quality",   rq)

            for dim in score_result.get("dimension_scores", []):
                with st.expander(f"{dim.get('dimension', '')} — {dim.get('score', 0):.1f}/100"):
                    st.write(dim.get("reasoning", ""))
                    for s in dim.get("suggestions", []):
                        st.write(f"- {s}")

            if score_result.get("overall_feedback"):
                st.info(score_result["overall_feedback"])
            if score_result.get("missing_keywords"):
                st.warning("Missing keywords: " + ", ".join(score_result["missing_keywords"]))

    # ── Generate tab ─────────────────────────────────────────────────────────
    with tab_generate:
        generation_panel(thread_id, key_prefix="done_gen")
