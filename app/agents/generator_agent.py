from typing import Union
from app.models.generation import Email, CoverLetter, OutreachMessage
from app.models.jd import ParsedJD
from app.models.resume import ParsedResume

from app.utils.llm import llm as _llm, Prompt

_TRUTHFULNESS_GUARDRAIL = """
### IMPORTANT TRUTHFULNESS CONSTRAINTS:
- DO NOT exaggerate the candidate's skills or experience.
- DO NOT invent projects, roles, companies, or achievements.
- Only use information present in the provided resume.
- If something is not in the resume, do not imply it.
- It is okay to rephrase or summarize, but remain factually accurate.
- Maintain a realistic and honest tone.
"""


def render_email(email: Union[Email, dict]) -> str:
    if isinstance(email, dict):
        email = Email(**email)

    lines = []

    # Subject
    lines.append(f"Subject: {email.subject}")
    lines.append("")

    # Greeting
    lines.append(email.greeting)
    lines.append("")

    # Body paragraphs
    for para in email.body:
        lines.append(para.strip())
        lines.append("")

    # Closing
    lines.append(email.closing)
    lines.append(email.signature)

    return "\n".join(lines).strip()


def render_cover_letter(cl: Union[CoverLetter, dict]) -> str:
    if isinstance(cl, dict):
        cl = CoverLetter(**cl)

    parts = []

    # Greeting
    parts.append(cl.greeting)
    parts.append("")

    # Opening
    parts.append(cl.opening_paragraph.strip())
    parts.append("")

    # Body paragraphs
    for para in cl.body_paragraphs:
        parts.append(para.strip())
        parts.append("")

    # Closing paragraph
    parts.append(cl.closing_paragraph.strip())
    parts.append("")

    # Sign-off
    parts.append(cl.closing)
    parts.append(cl.signature)

    return "\n".join(parts).strip()


def render_outreach_message(msg: Union[OutreachMessage, dict]) -> str:
    if isinstance(msg, dict):
        msg = OutreachMessage(**msg)

    return msg.full_message.strip()


def generate_cover_letter_prompt(job_description, resume_content):
    system_prompt = f"""
You are an expert career coach and professional resume writer.

Your task is to generate a tailored, compelling cover letter using the provided job description and candidate resume.

### Instructions:
- Personalize the letter specifically for the role and company (avoid generic language).
- Highlight the candidate's most relevant achievements and skills that match the job description.
- Focus on impact, results, and alignment with the role.
- Maintain a confident but professional tone.
- Avoid repeating the resume verbatim — synthesize and reframe it.
- Keep it concise (300-400 words).
- Don't try to add every information present on candidate, use only those which are relevant.

---

{_TRUTHFULNESS_GUARDRAIL}

---

"""

    prompt = f""""
        Generate Cover Letter for the below given, job description, resume content, and strictly follow the output schema.

### Job Description:
{job_description}

---

### Candidate Resume:
{resume_content}

---

### OUTPUT FORMAT (STRICT JSON)

Return JSON matching this schema:

{CoverLetter.model_json_schema()}

    """

    return system_prompt, prompt


def generate_outreach_message_prompt(job_description, resume_content):
    system_prompt = f"""
You are an experienced career coach helping candidates write high-response outreach messages.

Your task is to generate a concise, personalized outreach message to a recruiter or hiring manager.

### Instructions:
- Keep it short (4-6 sentences max).
- Friendly, confident, and natural tone (not overly formal).
- Mention:
- Who the candidate is (1 line intro)
- Key strength relevant to the role
- Alignment with the job description
- A polite call-to-action (e.g., open to connecting or referral)
- Avoid buzzwords and generic phrasing.
- Optimize for LinkedIn or cold outreach.
- Don't try to add every information present on candidate, use only those which are relevant.
---

{_TRUTHFULNESS_GUARDRAIL}

"""

    prompt = f""""
Generate Outreach Letter for the below given, job description, resume content, and strictly follow the output schema.


### Job Description:
{job_description}

---

### Candidate Resume:
{resume_content}

---

### OUTPUT FORMAT (STRICT JSON)

Return JSON matching this schema:

{OutreachMessage.model_json_schema()}

    """
    return system_prompt, prompt


def generate_email_prompt(job_description, resume_content):
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

{_TRUTHFULNESS_GUARDRAIL}

"""
    prompt = f"""
    Generate Email for the below given, job description, resume content, and strictly follow the output schema.

    ---

    ### Job Description:
    {job_description}

    ---

    ### Candidate Resume:
    {resume_content}

    ---

    ### OUTPUT FORMAT (STRICT JSON)

    Return JSON matching this schema exactly:

    {Email.model_json_schema()}
    """
    return system_prompt, prompt


def generate(resume: ParsedResume, jd: ParsedJD, type: str):
    resume_dict = {key: resume.model_dump()[key] for key in ["personal_info", "resume_sections"]}
    jd_dict = jd.model_dump()

    action_map = {
        "coverletter": (generate_cover_letter_prompt, CoverLetter, render_cover_letter),
        "outreachmessage": (generate_outreach_message_prompt, OutreachMessage, render_outreach_message),
        "email": (generate_email_prompt, Email, render_email),
    }

    type = type.lower().replace("_", "")
    if type in action_map:
        action, model, render = action_map[type]
        system_prompt, prompt = action(resume_dict, jd_dict)
    else:
        raise Exception

    response = _llm.generate(Prompt(system=system_prompt, user=prompt), provider="gemini", response_model=model)
    if not response.schema_matched or response.parsed is None:
        raise ValueError(f"LLM response did not match {model.__name__} schema")

    return render(response.parsed)
