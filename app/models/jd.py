from pydantic import BaseModel, Field


class ParsedJD(BaseModel):
    company:             str       = Field(default="",      description="Legal or trading name of the hiring company")
    role:                str       = Field(default="",      description="Exact job title as stated in the posting")
    seniority_level:     str       = Field(default="unknown", description="Seniority level of the role (e.g. Junior, Mid, Senior, Staff, Principal, Lead)")
    location:            str       = Field(default="",      description="Office location in 'City, State, Country' format; empty string if fully remote with no office listed")
    is_remote:           bool      = Field(default=False,   description="True if the role is fully remote, False if on-site or hybrid")
    about_company:       str       = Field(default="",      description="Summary of the company — what it does, its products or services, industry, size, mission, and culture if mentioned")
    about_job:           str       = Field(default="",      description="Overview of the role — the team it sits in, the problem it solves, why the position exists, and who the person will work with")
    responsibilities:    list[str] = Field(default_factory=list, description="Concrete duties and day-to-day tasks the candidate is expected to own, as a list of action-oriented statements")
    required_skills:     list[str] = Field(default_factory=list, description="Skills, tools, or technologies explicitly listed as required or mandatory")
    nice_to_have_skills: list[str] = Field(default_factory=list, description="Skills, tools, or technologies listed as preferred, a bonus, or nice to have")
    perks:               str       = Field(default="",      description="Benefits and perks offered — e.g. salary range, equity, health insurance, PTO, learning budget, flexible hours; empty string if none stated")
    others:              str       = Field(default="",      description="Any remaining relevant information from the JD that does not fit the above fields — e.g. visa sponsorship, hiring process, EEO statement, application instructions")
    raw_text:            str       = Field(default="",      exclude=True, description="Complete raw JD text, stored locally and excluded from schema / serialisation")

    def to_dict(self) -> dict:
        return self.model_dump(exclude={"raw_text"})
