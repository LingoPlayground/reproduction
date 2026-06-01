"""Data models for the video regeneration pipeline.

v3.0: LLM-first architecture. Semantic decisions live in planner_models.py.
This module contains only deterministic execution models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

MIN_MODIFIED_DURATION = 4.0
MAX_MODIFIED_DURATION = 30.0


@dataclass
class CutPoint:
    time_sec: float
    confidence: float = 1.0


@dataclass
class CanvasNode:
    node_id: str
    prompt: str
    video_url: str
    reference_images: List[str] = field(default_factory=list)
    duration_sec: Optional[float] = None


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
