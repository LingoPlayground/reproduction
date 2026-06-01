"""Data models for the video regeneration pipeline.

v3.0: LLM-first architecture. Semantic decisions live in planner_models.py.
This module contains only deterministic execution models.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

MIN_MODIFIED_DURATION = 4.0
MAX_MODIFIED_DURATION = 30.0


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


@dataclass
class CutPoint:
    time_sec: float


@dataclass
class CanvasNode:
    node_id: str
    prompt: str
    video_url: str
    reference_images: List[str] = field(default_factory=list)
    duration_sec: Optional[float] = None


@dataclass
class AtomLine:
    line_id: str
    speaker: str
    original: str
    rewritten: str
    start_sec: float
    end_sec: float
    shot_scene: str = ""

    @property
    def is_rewritten(self) -> bool:
        return _normalize_text(self.original) != _normalize_text(self.rewritten)


@dataclass
class EditAtom:
    atom_id: str
    shot_numbers: List[int] = field(default_factory=list)
    primary_shot_number: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0
    scene_description: str = ""
    lines: List[AtomLine] = field(default_factory=list)
    matched_node_id: Optional[str] = None
    match_confidence: Optional[float] = None
    match_reasoning: str = ""
    boundary_reason: str = ""
    source_cut_times: List[float] = field(default_factory=list)

    @property
    def rewritten_lines(self) -> List[AtomLine]:
        return [l for l in self.lines if l.is_rewritten]

    @property
    def has_rewritten_lines(self) -> bool:
        return any(l.is_rewritten for l in self.lines)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class WindowPlanDraft:
    """LLM-planned generation intent before deterministic materialization."""
    draft_id: str
    atom_ids: List[str] = field(default_factory=list)
    node_id: Optional[str] = None
    confidence: Optional[float] = None
    reasoning: str = ""
    fallback_reason: str = ""

    @property
    def is_fallback(self) -> bool:
        return not self.node_id


@dataclass
class GenerationWindow:
    window_id: str
    start_sec: float
    end_sec: float
    atoms: List[EditAtom] = field(default_factory=list)
    matched_node_id: Optional[str] = None
    match_confidence: Optional[float] = None
    rewritten_prompt: Optional[str] = None
    ref_images: List[str] = field(default_factory=list)
    degradation_level: int = 0
    degradation_reason: str = ""

    @property
    def covered_line_ids(self) -> List[str]:
        ids: List[str] = []
        for atom in self.atoms:
            for line in atom.rewritten_lines:
                ids.append(line.line_id)
        return sorted(set(ids))

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class TimelinePlanItem:
    shot_id: str
    shot_number: int
    source: Literal["original", "modified"]
    start_sec: float
    end_sec: float
    scene_description: str
    ref_images: List[str] = field(default_factory=list)
    rewritten_prompt: Optional[str] = None
    matched_node_id: Optional[str] = None
    match_confidence: Optional[float] = None
    degradation_level: int = 0
    original_duration: Optional[float] = None
    covered_line_ids: List[str] = field(default_factory=list)
    source_node_ids: List[str] = field(default_factory=list)
    degradation_reason: str = ""

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class TimelinePlan:
    title: str
    level: str
    original_video_path: str = ""
    total_duration_sec: float = 0.0
    items: List[TimelinePlanItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3Input:
    script_output: Any
    video_cut_points: List[CutPoint] = field(default_factory=list)
    rewrite_json: Dict[str, Any] = field(default_factory=dict)
    canvas_nodes: List[CanvasNode] = field(default_factory=list)
    level: str = "B2"
