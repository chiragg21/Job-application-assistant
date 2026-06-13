from typing import Union
from app.models.generation import Email, CoverLetter, OutreachMessage
from app.models.jd import ParsedJD
from app.models.resume import ParsedResume

from app.utils.llm import generate_for_task, Prompt

_TRUTHFULNESS_GUARDRAIL = """
### IMPORTANT TRUTHFULNESS CONSTRAINTS:
- DO NOT exaggerate the candidate's skills or experience.
- DO NOT invent projects, roles, companies, or achievements.
- Only use information present in the provided resume.
- If something is not in the resume, do not imply it.
- It is okay to rephrase or summarize, but remain factually accurate.
- Maintain a realistic and honest tone.
"""

_NO_JD_NOTE = (
    "No specific job description was provided. "
    "Write for a general cold-outreach context — highlight the candidate's "
    "strongest skills and experience without anchoring to a particular role."
)


def render_email(email: Union[Email, dict]) -> str:
    if isinstance(email, dict):
        email = Email(**email)

    lines = []
    lines.append(f"Subject: {email.subject}")
    lines.append("")
    lines.append(email.greeting)
    lines.append("")
    for para in email.body:
        lines.append(para.strip())
        lines.append("")
    lines.append(email.closing)
    lines.append(email.signature)

    return "\n".join(lines).strip()


def render_cover_letter(cl: Union[CoverLetter, dict]) -> str:
    if isinstance(cl, dict):
        cl = CoverLetter(**cl)

    parts = []
    parts.append(cl.greeting)
    parts.append("")
    parts.append(cl.opening_paragraph.strip())
    parts.append("")
    for para in cl.body_paragraphs:
        parts.append(para.strip())
        parts.append("")
    parts.append(cl.closing_paragraph.strip())
    parts.append("")
    parts.append(cl.closing)
    parts.append(cl.signature)

    return "\n".join(parts).strip()


def render_outreach_message(msg: Union[OutreachMessage, dict]) -> str:
    if isinstance(msg, dict):
        msg = OutreachMessage(**msg)
    return msg.full_message.strip()


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _custom_instruction_block(custom_instruction: str | None) -> str:
    if custom_instruction and custom_instruction.strip():
        return f"""
### CUSTOM INSTRUCTIONS (follow carefully):
{custom_instruction.strip()}
"""
    return ""


def generate_cover_letter_prompt(
    job_description,
    resume_content,
    custom_instruction: str | None = None,
    no_jd: bool = False,
):
    jd_note = _NO_JD_NOTE if no_jd else ""

    system_prompt = f"""
You are an expert career coach and professional resume writer.

Your task is to generate a tailored, compelling cover letter using the provided
{"candidate information" if no_jd else "job description and candidate resume"}.

### Instructions:
- {"Write for a general cold-outreach context — do not assume a specific role." if no_jd else "Personalize the letter specifically for the role and company (avoid generic language)."}
- Highlight the candidate's most relevant achievements and skills{"." if no_jd else " that match the job description."}
- Focus on impact, results, and alignment{"." if no_jd else " with the role."}
- Maintain a confident but professional tone.
- Avoid repeating the resume verbatim — synthesize and reframe it.
- Keep it concise (300-400 words).
- Don't try to add every information present on candidate, use only those which are relevant.

{jd_note}

---

{_TRUTHFULNESS_GUARDRAIL}

{_custom_instruction_block(custom_instruction)}

### OUTPUT SCHEMA (return JSON matching this exactly):
{CoverLetter.model_json_schema()}
"""

    prompt = f"""
Generate a Cover Letter for the candidate below.

{"### Context: " + _NO_JD_NOTE if no_jd else "### Job Description:"}
{"{}" if no_jd else str(job_description)}

---

### Candidate Resume:
{resume_content}
"""
    return system_prompt, prompt


def generate_outreach_message_prompt(
    job_description,
    resume_content,
    custom_instruction: str | None = None,
    no_jd: bool = False,
):
    jd_note = _NO_JD_NOTE if no_jd else ""

    system_prompt = f"""
You are an experienced career coach helping candidates write high-response outreach messages.

Your task is to generate a concise, personalized outreach message to a recruiter or hiring manager.

### Instructions:
- Keep it short (4-6 sentences max).
- Friendly, confident, and natural tone (not overly formal).
- Mention:
  - Who the candidate is (1 line intro)
  - Key strength{"" if no_jd else " relevant to the role"}
  - {"General value proposition" if no_jd else "Alignment with the job description"}
  - A polite call-to-action (e.g., open to connecting or referral)
- Avoid buzzwords and generic phrasing.
- Optimize for LinkedIn or cold outreach.
- Don't try to add every information present on candidate, use only those which are relevant.

{jd_note}

---

{_TRUTHFULNESS_GUARDRAIL}

{_custom_instruction_block(custom_instruction)}

### OUTPUT SCHEMA (return JSON matching this exactly):
{OutreachMessage.model_json_schema()}
"""

    prompt = f"""
Generate an Outreach Message for the candidate below.

{"### Context: " + _NO_JD_NOTE if no_jd else "### Job Description:"}
{"{}" if no_jd else str(job_description)}

---

### Candidate Resume:
{resume_content}
"""
    return system_prompt, prompt


def generate_email_prompt(
    job_description,
    resume_content,
    custom_instruction: str | None = None,
    no_jd: bool = False,
):
    jd_note = _NO_JD_NOTE if no_jd else ""

    system_prompt = f"""
You are an expert career coach helping candidates write professional job outreach emails.

Your task is to generate a polished email that the CANDIDATE will send to a recruiter or hiring manager.

This must be written in first person (I, my experience, etc.).

---

Rules:
- subject → compelling but professional
- greeting → e.g., "Dear Hiring Manager,"
- body → list of 2-4 short paragraphs
- closing → e.g., "Best regards,"
- signature → candidate name only
- No markdown
- No extra keys
- No explanation text

{jd_note}

{_TRUTHFULNESS_GUARDRAIL}

{_custom_instruction_block(custom_instruction)}

### OUTPUT SCHEMA (return JSON matching this exactly):
{Email.model_json_schema()}
"""

    prompt = f"""
Generate an Email for the candidate below.

{"### Context: " + _NO_JD_NOTE if no_jd else "### Job Description:"}
{"{}" if no_jd else str(job_description)}

---

### Candidate Resume:
{resume_content}
"""
    return system_prompt, prompt


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate(
    resume: ParsedResume,
    jd: ParsedJD | None,
    type: str,
    custom_instruction: str | None = None,
    no_jd: bool = False,
):
    from app.core.gen_cache import generation_cache

    resume_dict = {key: resume.model_dump()[key] for key in ["personal_info", "resume_sections"]}
    # Use None (not {}) when JD is absent or intentionally ignored so that
    # the cache key is unambiguous — {} would conflate "no JD" with "empty JD".
    jd_dict     = jd.model_dump() if (jd is not None and not no_jd) else None

    action_map = {
        "coverletter":     (generate_cover_letter_prompt,     CoverLetter,     render_cover_letter),
        "outreachmessage": (generate_outreach_message_prompt, OutreachMessage, render_outreach_message),
        "email":           (generate_email_prompt,            Email,           render_email),
    }

    type = type.lower().replace("_", "")
    if type not in action_map:
        raise ValueError(f"Unknown generation type: '{type}'")

    # ── Cache check — skip LLM entirely on identical inputs ───────────────
    cache_key = generation_cache.make_key(jd_dict, resume_dict, type, custom_instruction, no_jd)
    cached    = generation_cache.get(cache_key)
    if cached is not None:
        return cached

    # ── LLM call ──────────────────────────────────────────────────────────
    action, model, render_fn = action_map[type]
    system_prompt, prompt = action(
        jd_dict, resume_dict,
        custom_instruction=custom_instruction,
        no_jd=no_jd,
    )

    response = generate_for_task(
        "generation",
        Prompt(system=system_prompt, user=prompt),
        model,
    )
    if not response.schema_matched or response.parsed is None:
        raise ValueError(f"LLM response did not match {model.__name__} schema")

    result = render_fn(response.parsed)
    generation_cache.set(cache_key, type, result)
    return result
