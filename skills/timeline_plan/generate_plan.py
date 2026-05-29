#!/usr/bin/env python3
"""Stage 3: Timeline plan generator — orchestrates cut fusion, canvas matching, prompt extraction."""
from __future__ import annotations

import json
import math
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, KeyFrame, Stage3Input,
)
from skills.timeline_plan.cut_fusion import determine_cut_points
from skills.timeline_plan.canvas_matcher import match_canvas_node_for_shot
from skills.timeline_plan.prompt_extractor import extract_and_rewrite_prompt


def _shot_needs_rewrite(shot: Any, rewrite_lines: List[Dict]) -> Tuple[bool, List[Dict]]:
    shot_line_ids = {str(getattr(line, "line_id", "")) for line in (shot.lines or [])}
    matching = []
    has_change = False
    for rl in rewrite_lines:
        lid = str(rl.get("line_id", ""))
        if lid in shot_line_ids:
            matching.append(rl)
            if str(rl.get("original", "")) != str(rl.get("rewritten", "")):
                has_change = True
    return has_change, matching


def normalize_seedance_duration(target_sec: float) -> int:
    return max(5, min(30, round(target_sec)))


def _collect_ref_images(matched_node: Optional[CanvasNode], keyframes: List[KeyFrame], shot_number: int) -> List[str]:
    if matched_node and matched_node.reference_images:
        return list(matched_node.reference_images)
    shot_kfs = [kf.image_path for kf in keyframes if kf.shot_number == shot_number]
    if shot_kfs:
        return shot_kfs
    return []


def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    script_output = input_data.script_output
    shots = list(script_output.script.shots) if script_output else []
    rewrite_lines = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    video_cuts = input_data.video_cut_points
    keyframes = input_data.keyframes
    level = input_data.level

    video_duration = max(
        [s.end_seconds for s in shots if hasattr(s, 'end_seconds')],
        default=60.0,
    )
    cut_boundaries = determine_cut_points(shots, video_cuts, video_duration)

    items: List[TimelinePlanItem] = []
    for idx, shot in enumerate(shots):
        start_s, end_s = cut_boundaries[idx]
        shot_duration = end_s - start_s
        scene_desc = getattr(shot, "scene_description", "") or ""
        needs_rewrite, matching_lines = _shot_needs_rewrite(shot, rewrite_lines)

        if not needs_rewrite or not matching_lines:
            items.append(TimelinePlanItem(
                shot_id=f"shot_{shot.shot_number}", shot_number=shot.shot_number,
                source="original", start_sec=start_s, end_sec=end_s,
                scene_description=scene_desc, original_duration=shot_duration,
            ))
            continue

        degradation_level = 0
        matching_objects = [SimpleNamespace(**rl) for rl in matching_lines]
        matched_node, confidence = match_canvas_node_for_shot(shot, canvas_nodes, matching_objects)

        if matched_node:
            rewritten_prompt = extract_and_rewrite_prompt(matched_node.prompt, shot, matching_objects)
        else:
            degradation_level = max(degradation_level, 1)
            rewritten_prompt = extract_and_rewrite_prompt("", shot, matching_objects)

        ref_images = _collect_ref_images(matched_node, keyframes, shot.shot_number)
        if not ref_images:
            degradation_level = max(degradation_level, 1)

        seedance_dur = normalize_seedance_duration(shot_duration)
        items.append(TimelinePlanItem(
            shot_id=f"shot_{shot.shot_number}", shot_number=shot.shot_number,
            source="seedance", start_sec=start_s, end_sec=end_s,
            scene_description=scene_desc, ref_images=ref_images,
            rewritten_prompt=rewritten_prompt,
            matched_node_id=matched_node.node_id if matched_node else None,
            match_confidence=confidence if matched_node else None,
            degradation_level=degradation_level, seedance_duration=seedance_dur,
            original_duration=shot_duration,
        ))

    return TimelinePlan(
        title=getattr(script_output, "title", "Untitled") if script_output else "Untitled",
        level=level, total_duration_sec=video_duration, items=items,
        metadata={"num_shots": len(shots), "num_rewritten": sum(1 for i in items if i.source == "seedance"),
                   "num_original": sum(1 for i in items if i.source == "original")},
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Stage 3: Generate timeline plan")
    p.add_argument("--script", required=True)
    p.add_argument("--rewrite", required=True)
    p.add_argument("--canvas", help="Path to canvas nodes JSON")
    p.add_argument("--cuts", help="Path to video cut points JSON")
    p.add_argument("--keyframes", help="Path to keyframes JSON")
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

    cuts = [CutPoint(time_sec=c["time_sec"], confidence=c.get("confidence", 1.0)) for c in json.load(open(args.cuts))] if args.cuts else []
    kfs = [KeyFrame(time_sec=k["time_sec"], image_path=k["image_path"], shot_number=k["shot_number"]) for k in json.load(open(args.keyframes))] if args.keyframes else []

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
                s.start_seconds = d.get("start_seconds", 0.0)
                s.end_seconds = d.get("end_seconds", 0.0)
        def __init__(s, d):
            s.script = s._S(d)
            s.title = d.get("title", "Untitled")

    inp = Stage3Input(script_output=_SW(script_data) if script_data else None, video_cut_points=cuts, keyframes=kfs, rewrite_json=rewrite_data, canvas_nodes=canvas_nodes, level=args.level)
    plan = generate_timeline_plan(inp)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(plan), f, indent=2, ensure_ascii=False)
    print(f"Timeline plan: {len(plan.items)} shots -> {output_path}")
    for item in plan.items:
        deg = f" (L{item.degradation_level})" if item.degradation_level > 0 else ""
        print(f"  [{item.source}] Shot {item.shot_number}: {item.start_sec:.1f}s-{item.end_sec:.1f}s{deg}")

if __name__ == "__main__":
    main()
