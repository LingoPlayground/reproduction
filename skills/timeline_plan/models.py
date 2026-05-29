"""Data models for the ASR timeline-driven video regeneration pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class CutPoint:
    """A scene cut point detected by PySceneDetect."""
    time_sec: float
    confidence: float = 1.0


@dataclass
class KeyFrame:
    """A keyframe extracted from video at a cut point boundary."""
    time_sec: float
    image_path: str
    shot_number: int


@dataclass
class CanvasNode:
    """Canvas node data fetched from LibLib API."""
    node_id: str
    prompt: str
    video_url: str
    reference_images: List[str] = field(default_factory=list)
    duration_sec: Optional[float] = None


@dataclass
class TimelinePlanItem:
    """A single segment in the final video assembly plan."""
    shot_id: str
    shot_number: int
    source: Literal["original", "seedance"]
    start_sec: float
    end_sec: float
    scene_description: str
    ref_images: List[str] = field(default_factory=list)
    rewritten_prompt: Optional[str] = None
    matched_node_id: Optional[str] = None
    match_confidence: Optional[float] = None
    degradation_level: int = 0
    seedance_duration: Optional[int] = None
    original_duration: Optional[float] = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class TimelinePlan:
    """Complete assembly plan for a single episode at one CEFR level."""
    title: str
    level: str
    pipeline_version: str = "2.0"
    original_video_path: str = ""
    total_duration_sec: float = 0.0
    items: List[TimelinePlanItem] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3Input:
    """Bundled input for Stage 3 timeline plan generation."""
    script_output: Any  # VideoScriptOutput from lingolens
    video_cut_points: List[CutPoint] = field(default_factory=list)
    keyframes: List[KeyFrame] = field(default_factory=list)
    node_cut_points: Dict[str, List[CutPoint]] = field(default_factory=dict)
    rewrite_json: Dict[str, Any] = field(default_factory=dict)
    canvas_nodes: List[CanvasNode] = field(default_factory=list)
    level: str = "B2"


def normalize_seedance_duration(target_sec: float) -> int:
    """Map shot duration to seedance duration parameter.
    
    Clamped to [4, 30] seconds to match seedance API constraints and ensure
    the generated video fills the TimelinePlan time slot without gaps.
    """
    return max(4, min(30, round(target_sec)))
