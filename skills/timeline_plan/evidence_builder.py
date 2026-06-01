"""Evidence Builder: packages all inputs for LLM consumption.

Constructs the unified evidence dict. Does NOT extract dialogue from
node prompts — the LLM planner receives complete prompts and identifies
dialogue itself.

Key insight from rewrite JSON: each line already has `shot_scene` with
rich scene descriptions (from Stage 1 multimodal extraction). These are
far better video context than keyframe paths (which the LLM can't see).
"""
# DEPRECATED in v4.0: replaced by edit_atom_builder.py + segment_matcher.py.
# Kept for reference; will be removed in a future version.

from __future__ import annotations

from typing import Any, Dict, List

import re


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def build_evidence(
    script_shots: List[Any],
    rewrite_lines_all: List[Dict],
    canvas_nodes: List[Any],
    cut_points: List[Any],
    level: str = "B2",
) -> Dict[str, Any]:
    """Build the unified evidence dict for LLM consumption.

    Returns:
        Dict with:
        {
            "rewrite_lines": [...],     # lines where original != rewritten
            "neighbor_lines": [...],    # nearby unchanged lines for context
            "canvas_nodes": [...],      # complete node prompts + metadata
            "scene_context": {...},     # shot-level scene descriptions
            "timeline": {...},          # scene cuts, video duration
            "constraints": {...}        # hard constraints
        }
    """
    rewritten_lines: List[Dict] = []
    unchanged_lines: List[Dict] = []
    for rl in rewrite_lines_all:
        entry = {
            "line_id": str(rl.get("line_id", "")),
            "original": str(rl.get("original", "")),
            "rewritten": str(rl.get("rewritten", "")),
            "speaker": str(rl.get("speaker", "")),
            "start_sec": float(rl.get("start_seconds", 0.0)),
            "end_sec": float(rl.get("end_seconds", 0.0)),
            "shot_number": int(rl.get("shot_number", 0)),
            "shot_scene": str(rl.get("shot_scene", "")),
        }
        is_rewritten = _normalize_text(entry["original"]) != _normalize_text(entry["rewritten"])
        if is_rewritten:
            rewritten_lines.append(entry)
        else:
            unchanged_lines.append(entry)

    nodes = []
    for node in canvas_nodes:
        prompt = getattr(node, "prompt", "") or ""
        if len(prompt) > 800:
            prompt = prompt[:800] + "..."
        nodes.append({
            "node_id": getattr(node, "node_id", ""),
            "prompt": prompt,
        })

    # Build scene-level context from shot_scene descriptions
    # Each unique shot_scene description is already rich text from
    # Stage 1 multimodal extraction
    scene_map: Dict[int, str] = {}
    for rl in rewrite_lines_all:
        scene = str(rl.get("shot_scene", ""))
        sn = int(rl.get("shot_number", 0))
        if scene and sn not in scene_map:
            scene_map[sn] = scene
    scene_context = [
        {"shot_number": sn, "description": desc}
        for sn, desc in sorted(scene_map.items())
    ]

    video_duration = max(
        (rl.get("end_seconds", 0.0) for rl in rewrite_lines_all),
        default=60.0,
    )

    return {
        "rewrite_lines": rewritten_lines,
        "neighbor_lines": unchanged_lines[:50],
        "canvas_nodes": nodes,
        "scene_context": scene_context,
        "timeline": {
            "scene_cuts": [getattr(c, "time_sec", 0.0) for c in cut_points] if cut_points else [],
            "video_duration_sec": video_duration,
        },
        "constraints": {
            "must_cover_every_rewritten_line": True,
            "must_not_duplicate_lines": True,
            "must_preserve_environment_action_style": True,
            "min_modified_duration_sec": 4.0,
        },
    }
