# data_models/resume_edit_dm.py

from pydantic import BaseModel, Field
from typing import Optional, List
from data_models.jd_parse_dm import ParsedJD


class LlmOutput(BaseModel):
    lines_to_change: List[str] = Field(..., description="list of specific lines to be changed from latex content")
    suggested_changes: List[str] = Field(..., description="list of suggestions to replace original lines, also in latex")


class SectionEditState(BaseModel):
    section_name: str
    section_previous_state: str
    lines_to_change: List[str] = Field(default_factory=list)
    suggested_changes: List[str] = Field(default_factory=list)
    updated_section: str


class ResumeEditState(BaseModel):
    education:           Optional[SectionEditState] = None
    achievements:        Optional[SectionEditState] = None
    experience:          Optional[SectionEditState] = None
    projects:            Optional[SectionEditState] = None
    skills:              Optional[SectionEditState] = None
    relevant_coursework: Optional[SectionEditState] = None

    def get_section(self, section_name: str) -> Optional[SectionEditState]:
        return getattr(self, section_name, None)

    def set_section(self, section_name: str, state: SectionEditState) -> "ResumeEditState":
        return self.model_copy(update={section_name: state})


class ResumeStateNode(BaseModel):
    node_id: int
    parent_id: Optional[int] = None
    resume_state: ResumeEditState
    action: Optional[str] = None
    section_name: Optional[str] = None


class ResumeEditCycle(BaseModel):
    cycle_id: int
    jd: ParsedJD
    nodes: dict[int, ResumeStateNode] = Field(default_factory=dict)
    current_node_id: int = 0
    next_id: int = 0                            # renamed from _next_id (pydantic cant serialize private)

    def init_root(self, resume: ResumeEditState) -> None:
        root = ResumeStateNode(
            node_id=0,
            parent_id=None,
            resume_state=resume,
            action="root",
        )
        self.nodes[0] = root
        self.current_node_id = 0
        self.next_id = 1

    def push(self, resume_state: ResumeEditState, action: str, section_name: str) -> int:
        new_id = self.next_id
        self.next_id += 1

        self.nodes[new_id] = ResumeStateNode(
            node_id=new_id,
            parent_id=self.current_node_id,
            resume_state=resume_state,
            action=action,
            section_name=section_name,
        )
        self.current_node_id = new_id
        return new_id

    def checkout(self, node_id: int) -> ResumeEditState:
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} does not exist.")
        self.current_node_id = node_id
        return self.nodes[node_id].resume_state

    def revert(self) -> ResumeEditState:
        current = self.nodes[self.current_node_id]
        if current.parent_id is None:
            raise ValueError("Already at root — cannot revert further.")
        return self.checkout(current.parent_id)

    @property
    def current_state(self) -> ResumeEditState:
        return self.nodes[self.current_node_id].resume_state

    @property
    def previous_state(self) -> ResumeEditState:
        current = self.nodes[self.current_node_id]
        if current.parent_id is None:
            return current.resume_state
        return self.nodes[current.parent_id].resume_state

    def history(self) -> list[ResumeStateNode]:
        path = []
        node = self.nodes[self.current_node_id]
        while node is not None:
            path.append(node)
            node = self.nodes.get(node.parent_id) if node.parent_id is not None else None
        return list(reversed(path))

    def children_of(self, node_id: int) -> list[ResumeStateNode]:
        return [n for n in self.nodes.values() if n.parent_id == node_id]