"""Data models for the video regeneration pipeline — v4.0.

Deterministic execution models: EditAtom, GenerationWindow,
TimelinePlanItem, TimelinePlan. LLM-facing drafts: WindowPlanDraft.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from skills.common.models import CutPoint  # noqa: F401  # backward-compat re-export

MIN_MODIFIED_DURATION = 4.0
MAX_MODIFIED_DURATION = 30.0


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


@dataclass
class CanvasNode:
    node_id: str
    prompt: str
    video_url: str
    reference_images: list[str] = field(default_factory=list)
    duration_sec: float | None = None


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
    shot_numbers: list[int] = field(default_factory=list)
    primary_shot_number: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0
    scene_description: str = ""
    lines: list[AtomLine] = field(default_factory=list)
    matched_node_id: str | None = None
    match_confidence: float | None = None
    match_reasoning: str = ""
    boundary_reason: str = ""
    source_cut_times: list[float] = field(default_factory=list)

    @property
    def rewritten_lines(self) -> list[AtomLine]:
        return [l for l in self.lines if l.is_rewritten]

    @property
    def has_rewritten_lines(self) -> bool:
        return any(l.is_rewritten for l in self.lines)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class WindowPlanDraft:
    """LLM-planned generation intent; resolver computes executable timing."""
    draft_id: str
    atom_ids: list[str] = field(default_factory=list)
    node_id: str | None = None
    confidence: float | None = None
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
    atoms: list[EditAtom] = field(default_factory=list)
    matched_node_id: str | None = None
    match_confidence: float | None = None
    rewritten_prompt: str | None = None
    ref_images: list[str] = field(default_factory=list)
    degradation_level: int = 0
    degradation_reason: str = ""

    @property
    def covered_line_ids(self) -> list[str]:
        ids: list[str] = []
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
    ref_images: list[str] = field(default_factory=list)
    rewritten_prompt: str | None = None
    matched_node_id: str | None = None
    match_confidence: float | None = None
    degradation_level: int = 0
    original_duration: float | None = None
    covered_line_ids: list[str] = field(default_factory=list)
    source_node_ids: list[str] = field(default_factory=list)
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
    items: list[TimelinePlanItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage3Input:
    script_output: Any
    video_cut_points: list[CutPoint] = field(default_factory=list)
    rewrite_json: dict[str, Any] = field(default_factory=dict)
    canvas_nodes: list[CanvasNode] = field(default_factory=list)
    level: str = "B2"
