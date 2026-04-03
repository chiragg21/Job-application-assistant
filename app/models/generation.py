from pydantic import BaseModel, Field
from typing import List

class Email(BaseModel):
    subject: str = Field(..., description="Email subject line")
    greeting: str = Field(..., description="Greeting line, e.g., Dear Hiring Manager,")
    body: List[str] = Field(
        ...,
        description="List of body paragraphs (each paragraph as a string)"
    )
    closing: str = Field(..., description="Closing line, e.g., Best regards,")
    signature: str = Field(..., description="Sender name or placeholder")

class CoverLetter(BaseModel):
    greeting: str = Field(..., description="Opening greeting")
    opening_paragraph: str = Field(..., description="Introduction paragraph")
    body_paragraphs: List[str] = Field(
        ...,
        description="Main paragraphs highlighting experience and fit"
    )
    closing_paragraph: str = Field(..., description="Final paragraph with enthusiasm")
    closing: str = Field(..., description="Closing phrase, e.g., Sincerely,")
    signature: str = Field(..., description="Candidate name")

class OutreachMessage(BaseModel):
    intro: str = Field(..., description="1-line introduction")
    highlight: str = Field(..., description="Key strength or alignment")
    call_to_action: str = Field(..., description="Polite ask (connect, referral, etc.)")
    full_message: str = Field(..., description="Combined ready-to-send message")
