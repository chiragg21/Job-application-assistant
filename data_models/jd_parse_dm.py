# Correct — Pydantic BaseModel
from pydantic import BaseModel, Field

class ParsedJD(BaseModel):
    company:             str       = Field(default="")
    role:                str       = Field(default="")
    seniority_level:     str       = Field(default="unknown")
    location:            str       = Field(default="")
    is_remote:           bool      = Field(default=False)
    responsibilities:    list[str] = Field(default_factory=list)
    required_skills:     list[str] = Field(default_factory=list)
    nice_to_have_skills: list[str] = Field(default_factory=list)
    raw_text:            str       = Field(default="", exclude=True)

    def to_dict(self) -> dict:
        return self.model_dump(exclude={"raw_text"})