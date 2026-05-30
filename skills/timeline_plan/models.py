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
    seedance_duration: Optional[float] = None
    original_duration: Optional[float] = None
    # v3 tracking fields (Phase 1)
    operation_type: Optional[str] = None
    duration_strategy: Optional[str] = None
    covered_line_ids: List[str] = field(default_factory=list)
    borrowed_line_ids: List[str] = field(default_factory=list)
    source_node_ids: List[str] = field(default_factory=list)
    degradation_reason: str = ""

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


@dataclass
class PromptPatchPlan:
    """Layered prompt editing plan: style + visual context + dialogue patches."""
    operation_type: Literal[
        "literal_replace", "fuzzy_replace", "semantic_insert",
        "section_reconstruct", "style_preserving_fallback", "full_fallback"
    ]
    global_style: str
    local_visual_context: str
    dialogue_patches: List[Dict[str, str]] = field(default_factory=list)
    discarded_sections: List[str] = field(default_factory=list)
    final_prompt: str = ""


@dataclass
class CoveragePlan:
    """Time coverage plan: what interval to generate and what strategy was used."""
    start_sec: float
    end_sec: float
    included_rewritten_line_ids: List[str]
    borrowed_original_line_ids: List[str]
    duration_strategy: Literal[
        "direct", "pad_after", "pad_before", "snap_to_cut",
        "hold_reaction", "borrow_neighbor", "merge_same_node_group",
        "cross_node_merge", "forced_min_duration"
    ]
    duration_expansion_sec: float = 0.0


@dataclass
class MatchEvidence:
    """A single matching signal between a line group and a canvas node."""
    signal: Literal[
        "quoted_dialogue", "fuzzy_dialogue", "speaker_presence",
        "visual_action", "shot_scene_similarity", "temporal_order",
        "reference_image_match", "implicit_visual_scene"
    ]
    detail: str
    confidence: float


MIN_SEEDANCE_DURATION = 4.0


def normalize_seedance_duration(target_sec: float) -> int:
    """Map shot duration to seedance duration parameter.
    
    After merge-up, all items should be >= MIN_SEEDANCE_DURATION.
    Returns -1 for smart duration (seedance auto-determines best length).
    """
    if target_sec < MIN_SEEDANCE_DURATION:
        return max(4, round(target_sec))
    return -1
