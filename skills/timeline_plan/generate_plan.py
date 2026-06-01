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
from typing import Any, Dict, List

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, Stage3Input,
)
from skills.timeline_plan.cut_fusion import determine_cut_points
from skills.timeline_plan.evidence_builder import build_evidence
from skills.timeline_plan.llm_planner import generate_plan_draft
from skills.timeline_plan.timeline_normalizer import normalize_plan
from skills.timeline_plan.validator import validate_timeline_item, validate_timeline_items
import logging

logger = logging.getLogger(__name__)


def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    script_output = input_data.script_output
    shots = list(script_output.script.shots) if script_output else []
    rewrite_lines_all = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    video_cuts = input_data.video_cut_points
    level = input_data.level

    video_duration = max(
        [s.end_seconds for s in shots if hasattr(s, 'end_seconds')],
        default=60.0,
    )
    max_asr_end = max(
        (rl.get("end_seconds", 0.0) for rl in rewrite_lines_all),
        default=0.0,
    )
    video_duration = max(video_duration, max_asr_end)
    title = getattr(script_output, "title", "Untitled") if script_output else "Untitled"

    for shot in shots:
        sn = getattr(shot, "shot_number", 0)
        asr_starts = [rl.get("start_seconds", 0.0) for rl in rewrite_lines_all
                       if rl.get("shot_number") == sn and rl.get("start_seconds", 0) > 0]
        if asr_starts:
            offset = abs(min(asr_starts) - getattr(shot, "start_seconds", 0.0))
            if offset > 10:
                logger.warning("Shot %d: script vs ASR offset %.0fs", sn, offset)

    # ── Stage 3A: Evidence ──
    logger.info("Building evidence...")
    evidence = build_evidence(shots, rewrite_lines_all, canvas_nodes, video_cuts, level)
    logger.info("Evidence: %d rewrite lines, %d canvas nodes, %d shots with scene context",
                len(evidence["rewrite_lines"]), len(evidence["canvas_nodes"]),
                len(evidence["scene_context"]))

    # Quick path: no rewritten lines
    if not evidence["rewrite_lines"]:
        logger.info("No rewritten lines — all-original plan")
        cut_boundaries = determine_cut_points(shots, video_cuts, video_duration)
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

    # ── Stage 3B: LLM Planner (match + rewrite) ──
    logger.info("Running LLM planner...")
    try:
        draft = generate_plan_draft(evidence, canvas_nodes=canvas_nodes)
    except ValueError as e:
        logger.error("LLM planner failed: %s", e)
        raise

    logger.info("Planner: %d generations, %d unmatched",
                len(draft.node_generations), len(draft.unmatched_lines))

    # ── Stage 3C: Normalizer ──
    logger.info("Normalizing timeline...")
    plan = normalize_plan(
        draft=draft, script_shots=shots, canvas_nodes=canvas_nodes,
        cut_points=video_cuts, video_duration=video_duration,
        title=title, level=level,
    )

    # ── Post-normalization coverage ──
    # Only check that ALL rewritten lines are covered (neighbor lines are allowed extras)
    all_rewrite_ids = {rl["line_id"] for rl in evidence["rewrite_lines"]}
    final_covered: Dict[str, List[str]] = {}
    for item in plan.items:
        for lid in (item.covered_line_ids or []):
            final_covered.setdefault(lid, []).append(item.shot_id)

    missing = all_rewrite_ids - set(final_covered)
    dups = {lid: shots for lid, shots in final_covered.items() if len(shots) > 1 and lid in all_rewrite_ids}

    post_errors: List[str] = []
    if missing:
        post_errors.append(f"{len(missing)} lines missing: {sorted(missing)[:10]}")
    if dups:
        post_errors.append(f"{len(dups)} lines duplicated")
    if post_errors:
        raise ValueError("Post-normalization coverage FAILED:\n" + "\n".join(post_errors))

    # ── Final validation ──
    validation_errors: List[str] = []
    for item in plan.items:
        validation_errors.extend(validate_timeline_item(item))
    validation_errors.extend(validate_timeline_items(plan.items, video_duration))

    blocking = [
        e for e in validation_errors
        if any(kw in e.lower() for kw in (
            "overlap", "start_sec (", "missing start_sec", ">= end_sec",
            "empty rewritten_prompt", "covered by both",
            "gap at start", "gap at end", "empty shot_id", "invalid source",
        ))
    ]
    if blocking:
        raise ValueError(
            f"Validation FAILED with {len(blocking)} blocking errors:\n"
            + "\n".join(f"  - {e}" for e in blocking)
        )
    if validation_errors:
        logger.warning("%d non-blocking warnings", len(validation_errors))

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
        with open(args.cuts) as f:
            cuts = [CutPoint(time_sec=c["time_sec"], confidence=c.get("confidence", 1.0)) for c in json.load(f)]

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
