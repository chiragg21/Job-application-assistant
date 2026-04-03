from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────

class SectionItem(BaseModel):
    """Maps to one row in resume_section_items."""
    id: int
    section_id: int
    section_name: str
    item_name: Optional[str]
    role_title: Optional[str]
    content_latex: str
    content_text: str
    item_index: int

class Section(BaseModel):
    """Maps to one row in resume_sections (parent blob)."""
    id: int
    section_name: str
    content_latex: str

class RetrievalResult(BaseModel):
    """Everything Stage 3 needs from Stage 2."""
    items: List[SectionItem]
    sections: List[Section]
    similarity_score: float