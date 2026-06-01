#!/usr/bin/env python3
"""Stage 3: Timeline plan generator — LLM-first pipeline.

v3.0: Evidence Builder → LLM Planner (single-pass) → Verifier →
Timeline Normalizer (pure geometry). LLM handles all semantics.
Deterministic code handles only validation and geometry.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, Stage3Input,
    AtomLine, EditAtom,
)
from skills.timeline_plan.edit_atom_builder import build_edit_atoms
from skills.timeline_plan.segment_matcher import match_atoms_to_nodes
from skills.timeline_plan.generation_window_resolver import resolve_generation_windows
from skills.timeline_plan.prompt_rewriter import rewrite_prompts_for_windows
from skills.timeline_plan.plan_finalizer import finalize_timeline_plan
from skills.timeline_plan.cut_fusion import determine_cut_points
import logging

logger = logging.getLogger(__name__)


def _collect_all_atom_lines(rewrite_lines_all: list[dict]) -> list[AtomLine]:
    """Collect all lines as AtomLine objects for window resolution."""
    lines = []
    for rl in rewrite_lines_all:
        lines.append(AtomLine(
            line_id=str(rl.get("line_id", "")),
            speaker=str(rl.get("speaker", "")),
            original=str(rl.get("original", "")),
            rewritten=str(rl.get("rewritten", "")),
            start_sec=float(rl.get("start_seconds", 0.0)),
            end_sec=float(rl.get("end_seconds", 0.0)),
            shot_scene=str(rl.get("shot_scene", "")),
        ))
    return lines


def _resolve_video_duration(shots: list, rewrite_lines: list[dict]) -> float:
    """Compute video duration from shot boundaries and ASR timing."""
    shot_end = max(
        (s.end_seconds for s in shots if hasattr(s, 'end_seconds')),
        default=60.0,
    )
    asr_end = max(
        (float(rl.get("end_seconds", 0.0)) for rl in rewrite_lines),
        default=0.0,
    )
    return max(shot_end, asr_end)


def _build_all_original_plan(
    shots: list,
    scene_cuts: list[CutPoint],
    video_duration: float,
    title: str = "Untitled",
    level: str = "B2",
) -> TimelinePlan:
    """Fallback: build an all-original plan when no lines are rewritten."""
    cut_boundaries = determine_cut_points(shots, scene_cuts, video_duration)
    items = []
    for idx, shot in enumerate(shots):
        start_s, end_s = cut_boundaries[idx]
        items.append(TimelinePlanItem(
            shot_id=f"shot_{getattr(shot, 'shot_number', idx)}",
            shot_number=getattr(shot, "shot_number", idx),
            source="original",
            start_sec=start_s, end_sec=end_s,
            scene_description=getattr(shot, "scene_description", "") or "",
            original_duration=end_s - start_s,
        ))
    items.sort(key=lambda i: i.start_sec)
    return TimelinePlan(
        title=title, level=level,
        total_duration_sec=video_duration, items=items,
        metadata={"num_shots": len(shots), "num_items": len(items),
                   "num_modified": 0, "num_original": len(items)},
    )


def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    script_output = input_data.script_output
    shots = list(script_output.script.shots) if script_output else []
    rewrite_lines_all = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    scene_cuts = input_data.video_cut_points
    level = input_data.level

    video_duration = _resolve_video_duration(shots, rewrite_lines_all)
    title = getattr(script_output, "title", "Untitled") if script_output else "Untitled"

    logger.info("Building edit atoms...")
    atoms = build_edit_atoms(
        script_shots=shots, rewrite_lines=rewrite_lines_all,
        scene_cuts=scene_cuts, video_duration=video_duration,
    )
    logger.info("Built %d atoms from %d rewrite lines", len(atoms), len(rewrite_lines_all))

    target_atoms = [a for a in atoms if a.has_rewritten_lines]
    if not target_atoms:
        logger.info("No rewritten lines — all-original plan")
        return _build_all_original_plan(shots, scene_cuts, video_duration, title, level)

    logger.info("Target atoms: %d (with rewritten lines)", len(target_atoms))

    logger.info("Matching atoms to canvas nodes...")
    match_atoms_to_nodes(target_atoms, canvas_nodes)
    matched = sum(1 for a in target_atoms if a.matched_node_id)
    logger.info("Matcher: %d/%d atoms matched", matched, len(target_atoms))

    logger.info("Resolving generation windows...")
    all_lines = _collect_all_atom_lines(rewrite_lines_all)
    windows = resolve_generation_windows(
        atoms=target_atoms, all_lines=all_lines,
        canvas_nodes=canvas_nodes, video_duration=video_duration,
    )
    logger.info("Windows: %d generation windows", len(windows))

    logger.info("Rewriting prompts per window...")
    rewrite_prompts_for_windows(windows, canvas_nodes, level)
    ok = sum(1 for w in windows if w.rewritten_prompt)
    logger.info("Rewriter: %d/%d windows have prompts", ok, len(windows))

    logger.info("Finalizing timeline plan...")
    plan = finalize_timeline_plan(
        windows=windows, shots=shots,
        video_duration=video_duration, title=title, level=level,
    )

    logger.info("Timeline plan: %d items (%d modified, %d original)",
                len(plan.items),
                sum(1 for i in plan.items if i.source == "modified"),
                sum(1 for i in plan.items if i.source == "original"))
    return plan


def main():
    from skills.common.env import load_pipeline_env
    load_pipeline_env()
    import argparse
    p = argparse.ArgumentParser(description="Stage 3: Generate timeline plan")
    p.add_argument("--script", required=True)
    p.add_argument("--rewrite", required=True)
    p.add_argument("--canvas")
    p.add_argument("--cuts")
    p.add_argument("--output", required=True)
    p.add_argument("--level", default="B2")
    args = p.parse_args()

    with open(args.script, encoding="utf-8") as f:
        script_data = json.load(f)
    with open(args.rewrite, encoding="utf-8") as f:
        rewrite_data = json.load(f)

    canvas_nodes: List[CanvasNode] = []
    if args.canvas and Path(args.canvas).exists():
        with open(args.canvas, encoding="utf-8") as f:
            for n in json.load(f):
                canvas_nodes.append(CanvasNode(
                    node_id=str(n.get("nodeId") or n.get("node_id", "")),
                    prompt=str(n.get("prompt") or n.get("data_obj", {}).get("prompt", "")),
                    video_url=str(n.get("video_url") or n.get("data_obj", {}).get("url", "")),
                    reference_images=n.get("reference_images") or n.get("data_obj", {}).get("images", []),
                ))

    cuts: List[CutPoint] = []
    if args.cuts and Path(args.cuts).exists():
        with open(args.cuts, encoding="utf-8") as f:
            cuts = [CutPoint(time_sec=c["time_sec"]) for c in json.load(f)]

    class _SW:
        class _S:
            def __init__(s, d):
                s.shots = [_SW._Sh(s) for s in d.get("script", {}).get("shots", [])]
        class _Sh:
            def __init__(s, d):
                s.shot_number = d.get("shot_number", 0)
                s.start_seconds = d.get("start_seconds", 0.0)
                s.end_seconds = d.get("end_seconds", 0.0)
                s.scene_description = d.get("scene_description", "")
                s.lines = [_SW._L(l) for l in d.get("lines", [])]
        class _L:
            def __init__(s, d):
                s.line_id = d.get("line_id", "")
                s.dialogue = d.get("dialogue", "")
                s.speaker = d.get("speaker", "")
                s.start_seconds = d.get("start_seconds", 0.0)
                s.end_seconds = d.get("end_seconds", 0.0)
        def __init__(s, d):
            s.script = s._S(d)
            s.title = d.get("title", "Untitled")

    inp = Stage3Input(
        script_output=_SW(script_data) if script_data else None,
        video_cut_points=cuts,
        rewrite_json=rewrite_data, canvas_nodes=canvas_nodes, level=args.level,
    )
    plan = generate_timeline_plan(inp)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2, ensure_ascii=False)
    print(f"Timeline plan: {len(plan.items)} items -> {output_path}")
    for item in plan.items:
        deg = f" (L{item.degradation_level})" if item.degradation_level > 0 else ""
        node_info = f" node={item.matched_node_id[:12]}..." if item.matched_node_id else ""
        print(f"  [{item.source}] Shot {item.shot_number}: {item.start_sec:.1f}s-{item.end_sec:.1f}s{deg}{node_info}")


if __name__ == "__main__":
    main()
