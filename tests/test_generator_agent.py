"""Tests for app/agents/generator_agent.py"""
import pytest
from unittest.mock import MagicMock, patch

from app.agents.generator_agent import (
    render_email,
    render_cover_letter,
    render_outreach_message,
    generate_cover_letter_prompt,
    generate_outreach_message_prompt,
    generate_email_prompt,
    generate,
)
from app.models.generation import Email, CoverLetter, OutreachMessage
from app.models.jd import ParsedJD
from app.models.resume import ParsedResume, PersonalInfo, ResumeSection


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_email():
    return Email(
        subject="Application for ML Engineer",
        greeting="Dear Hiring Manager,",
        body=["I am writing to express my interest.", "My background in ML aligns well."],
        closing="Best regards,",
        signature="Chirag Garg",
    )


@pytest.fixture
def sample_cover_letter():
    return CoverLetter(
        greeting="Dear Hiring Manager,",
        opening_paragraph="I am excited to apply for the ML Engineer role at DeepMind.",
        body_paragraphs=["My experience with PyTorch is extensive.", "I have deployed models at scale."],
        closing_paragraph="I look forward to discussing this opportunity.",
        closing="Sincerely,",
        signature="Chirag Garg",
    )


@pytest.fixture
def sample_outreach():
    return OutreachMessage(
        intro="Hi, I'm Chirag — an ML engineer.",
        highlight="I've built production RAG systems.",
        call_to_action="Would love to chat about the ML Engineer role.",
        full_message="Hi, I'm Chirag — an ML engineer. I've built production RAG systems. "
                     "Would love to chat about the ML Engineer role.",
    )


@pytest.fixture
def sample_parsed_resume():
    return ParsedResume(
        resume_id="1",
        resume_path="/tmp/resume.tex",
        personal_info=PersonalInfo(
            name="Chirag Garg",
            email="chirag@example.com",
            phone="+91-9876543210",
            github="github.com/chiragg21",
            linkedin="linkedin.com/in/chiragg21",
        ),
        resume_sections=ResumeSection(),
    )


# ── render_email ──────────────────────────────────────────────────────────────

class TestRenderEmail:
    def test_subject_line_present(self, sample_email):
        rendered = render_email(sample_email)
        assert "Subject: Application for ML Engineer" in rendered

    def test_greeting_present(self, sample_email):
        assert "Dear Hiring Manager," in render_email(sample_email)

    def test_body_lines_present(self, sample_email):
        rendered = render_email(sample_email)
        assert "I am writing to express my interest." in rendered

    def test_signature_present(self, sample_email):
        assert "Chirag Garg" in render_email(sample_email)

    def test_accepts_dict_input(self, sample_email):
        rendered_from_dict  = render_email(sample_email.model_dump())
        rendered_from_model = render_email(sample_email)
        assert rendered_from_dict == rendered_from_model


# ── render_cover_letter ───────────────────────────────────────────────────────

class TestRenderCoverLetter:
    def test_greeting_present(self, sample_cover_letter):
        assert "Dear Hiring Manager," in render_cover_letter(sample_cover_letter)

    def test_opening_present(self, sample_cover_letter):
        assert "excited to apply" in render_cover_letter(sample_cover_letter)

    def test_body_paragraphs_present(self, sample_cover_letter):
        rendered = render_cover_letter(sample_cover_letter)
        assert "PyTorch" in rendered
        assert "deployed models" in rendered

    def test_signature_present(self, sample_cover_letter):
        assert "Chirag Garg" in render_cover_letter(sample_cover_letter)

    def test_accepts_dict_input(self, sample_cover_letter):
        rendered_from_dict  = render_cover_letter(sample_cover_letter.model_dump())
        rendered_from_model = render_cover_letter(sample_cover_letter)
        assert rendered_from_dict == rendered_from_model


# ── render_outreach_message ───────────────────────────────────────────────────

class TestRenderOutreachMessage:
    def test_uses_full_message(self, sample_outreach):
        rendered = render_outreach_message(sample_outreach)
        assert sample_outreach.full_message in rendered

    def test_accepts_dict_input(self, sample_outreach):
        rendered_from_dict  = render_outreach_message(sample_outreach.model_dump())
        rendered_from_model = render_outreach_message(sample_outreach)
        assert rendered_from_dict == rendered_from_model


# ── Prompt builders ───────────────────────────────────────────────────────────

class TestPromptBuilders:
    @pytest.fixture
    def resume_dict(self, sample_parsed_resume):
        return {
            "personal_info":    sample_parsed_resume.personal_info.model_dump(),
            "resume_sections":  sample_parsed_resume.resume_sections.model_dump(),
        }

    @pytest.fixture
    def jd_dict(self, sample_jd):
        return sample_jd.model_dump()

    def test_cover_letter_prompt_returns_two_strings(self, resume_dict, jd_dict):
        sys_p, user_p = generate_cover_letter_prompt(jd_dict, resume_dict)
        assert isinstance(sys_p, str) and isinstance(user_p, str)

    def test_cover_letter_prompt_contains_role(self, resume_dict, jd_dict):
        sys_p, _ = generate_cover_letter_prompt(jd_dict, resume_dict)
        assert "cover letter" in sys_p.lower() or "ML Engineer" in sys_p

    def test_outreach_prompt_returns_two_strings(self, resume_dict, jd_dict):
        sys_p, user_p = generate_outreach_message_prompt(jd_dict, resume_dict)
        assert isinstance(sys_p, str) and isinstance(user_p, str)

    def test_email_prompt_returns_two_strings(self, resume_dict, jd_dict):
        sys_p, user_p = generate_email_prompt(jd_dict, resume_dict)
        assert isinstance(sys_p, str) and isinstance(user_p, str)


# ── generate ──────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_generate_cover_letter(self, sample_parsed_resume, sample_jd, sample_cover_letter):
        with patch("app.agents.generator_agent.generate_for_task") as mock_gtf, \
             patch("app.core.gen_cache.generation_cache.get", return_value=None):
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=sample_cover_letter)
            result = generate(sample_parsed_resume, sample_jd, "coverletter")

        assert "Chirag Garg" in result or "excited" in result

    def test_generate_email(self, sample_parsed_resume, sample_jd, sample_email):
        with patch("app.agents.generator_agent.generate_for_task") as mock_gtf, \
             patch("app.core.gen_cache.generation_cache.get", return_value=None):
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=sample_email)
            result = generate(sample_parsed_resume, sample_jd, "email")

        assert "Subject:" in result

    def test_generate_outreach_message(self, sample_parsed_resume, sample_jd, sample_outreach):
        with patch("app.agents.generator_agent.generate_for_task") as mock_gtf, \
             patch("app.core.gen_cache.generation_cache.get", return_value=None):
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=sample_outreach)
            result = generate(sample_parsed_resume, sample_jd, "outreachmessage")

        assert sample_outreach.full_message in result

    def test_type_normalised_with_underscore(self, sample_parsed_resume, sample_jd, sample_cover_letter):
        with patch("app.agents.generator_agent.generate_for_task") as mock_gtf, \
             patch("app.core.gen_cache.generation_cache.get", return_value=None):
            mock_gtf.return_value = MagicMock(schema_matched=True, parsed=sample_cover_letter)
            result = generate(sample_parsed_resume, sample_jd, "cover_letter")
        assert isinstance(result, str)

    def test_unknown_type_raises(self, sample_parsed_resume, sample_jd):
        with pytest.raises(Exception):
            generate(sample_parsed_resume, sample_jd, "unknown_type")

    def test_schema_mismatch_raises(self, sample_parsed_resume, sample_jd):
        with patch("app.agents.generator_agent.generate_for_task") as mock_gtf, \
             patch("app.core.gen_cache.generation_cache.get", return_value=None):
            mock_gtf.return_value = MagicMock(schema_matched=False, parsed=None)
            with pytest.raises(ValueError, match="schema"):
                generate(sample_parsed_resume, sample_jd, "email")
