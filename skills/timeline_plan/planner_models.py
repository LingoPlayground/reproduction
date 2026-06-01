"""Planner models: LLM output schemas for two-stage pipeline.

Stage A1 (Map): LLM maps each rewritten line to a canvas node (1-to-1).
  Code then deterministically groups by node + time proximity.
Stage B (Rewrite): Per-group LLM rewrites the prompt.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SourceTimeRange:
    start_sec: float = 0.0
    end_sec: float = 0.0


@dataclass
class LineNodeMatch:
    """One rewritten line mapped to a node (or None if unmatched)."""
    line_id: str = ""
    original_line: str = ""
    rewritten_line: str = ""
    node_id: Optional[str] = None
    match_reasoning: str = ""
    original_dialogue_in_prompt: Optional[str] = None
    confidence: float = 0.0


@dataclass
class NodeGeneration:
    """One canvas node → one generation call. Built by deterministic grouping."""
    group_id: str = ""
    covered_line_ids: List[str] = field(default_factory=list)
    matched_node_ids: List[str] = field(default_factory=list)
    source_time_range: Optional[SourceTimeRange] = None
    line_matches: List[LineNodeMatch] = field(default_factory=list)
    grouping_reasoning: str = ""
    confidence: float = 0.0
    rewritten_prompt: str = ""

    @property
    def has_prompt(self) -> bool:
        return bool(self.rewritten_prompt and self.rewritten_prompt.strip())


@dataclass
class UnmatchedLine:
    line_id: str = ""
    reason: str = ""
    original: str = ""
    rewritten: str = ""
    # For building degraded fallback items in normalizer
    start_sec: float = 0.0
    end_sec: float = 0.0
    speaker: str = ""
    shot_scene: str = ""
    shot_number: int = 0


@dataclass
class MatchResult:
    """LLM output + deterministic grouping result."""
    plan_version: str = "llm_planner_v1"
    line_matches: List[LineNodeMatch] = field(default_factory=list)
    unmatched_lines: List[UnmatchedLine] = field(default_factory=list)
    node_generations: List[NodeGeneration] = field(default_factory=list)


@dataclass
class RewriteInput:
    group_id: str
    original_prompt: str
    covered_lines: List[dict] = field(default_factory=list)
