from pydantic import BaseModel
from typing import List

class ResumeBuilderInput(BaseModel):
    education: str
    achievements: str
    experience: List[str]
    projects: List[str]
    skills: str
    relevant_coursework: str