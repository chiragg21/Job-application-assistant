from typing import Optional, List
from pydantic import BaseModel, Field

class PersonalInfo(BaseModel):
    name: str = Field(..., description="Full name of the candidate")
    email: str = Field(..., description="Email address")
    phone: str = Field(..., description="Phone number")
    linkedin: str = Field(..., description="LinkedIn profile URL")
    github: str = Field(..., description="GitHub profile URL") 

# NEW: Helper class to ensure every flat section has both formats
class SectionContent(BaseModel):
    content_latex: Optional[str] = Field(None, description="Raw LaTeX content")
    content_text: Optional[str] = Field(None, description="Cleaned plain text content")

# NEW: Model for atomic work/project blocks
class AtomicItem(BaseModel):
    name: Optional[str] = Field(None, description="Company or Project name")
    role: Optional[str] = Field(None, description="Job title or specific role")
    content_latex: str = Field(..., description="Original LaTeX block for this item")
    content_text: str = Field(..., description="Cleaned text for this item")

class ResumeSection(BaseModel):
    # Updated: Flat sections now use the dual-format helper
    education: Optional[SectionContent] = Field(None, description="Education details")
    achievements: Optional[SectionContent] = Field(None, description="Achievements details")
    skills: Optional[SectionContent] = Field(None, description="Skills details")
    relevant_coursework: Optional[SectionContent] = Field(None, description="Relevant coursework details")
    
    # Updated: Complex sections now use a List of AtomicItems
    experience: List[AtomicItem] = Field(default_factory=list, description="List of work experience items")
    projects: List[AtomicItem] = Field(default_factory=list, description="List of project items")

class ParsedResume(BaseModel):
    resume_id: str = Field(..., description="Unique identifier for the resume")
    resume_path: str = Field(..., description="File path to the original resume")
    personal_info: PersonalInfo = Field(..., description="Parsed personal information")
    resume_sections: ResumeSection = Field(..., description="Parsed resume sections")
    