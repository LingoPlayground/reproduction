#!/usr/bin/env python3
"""Stage 3: Timeline plan generator — orchestrates cut fusion, line-to-node matching, prompt extraction.

v2.1: Lines are matched to canvas nodes by extracting quoted dialogue from prompts.
One shot's lines may map to multiple nodes.  Rewritten lines within the same node
are grouped together for prompt extraction.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, KeyFrame, Stage3Input,
    normalize_seedance_duration, MIN_SEEDANCE_DURATION,
)
from skills.timeline_plan.cut_fusion import determine_cut_points
from skills.timeline_plan.canvas_matcher import match_lines_to_nodes
from skills.timeline_plan.prompt_extractor import extract_and_rewrite_prompt


def _shot_needs_rewrite(shot: Any, rewrite_lines: List[Dict]) -> Tuple[bool, List[Dict]]:
    """Check which lines in a shot have rewritten dialogue."""
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


def _collect_ref_images(matched_node: Optional[CanvasNode], keyframes: List[KeyFrame], shot_number: int) -> List[str]:
    if matched_node and matched_node.reference_images:
        return list(matched_node.reference_images)
    shot_kfs = [kf.image_path for kf in keyframes if kf.shot_number == shot_number]
    return shot_kfs if shot_kfs else []


def _extend_short_group(
    group: List[Dict],
    node_line_ids: set,
    all_lines_map: Dict[str, Dict],
    line_to_node: Dict[str, str],
) -> List[Dict]:
    """Extend a short group (< 4s) by including nearby lines from the same node.
    
    Searches forward and backward in time for lines in the same node
    (including non-rewritten ones) until the total duration reaches 4s.
    Returns the extended group as rewrite-compatible dicts.
    """
    min_start = min(rl.get("start_seconds", 0.0) for rl in group)
    max_end = max(rl.get("end_seconds", 0.0) for rl in group)
    group_ids = {rl["line_id"] for rl in group}
    
    # Collect candidate lines from same node, sorted by time
    candidates = []
    for lid, info in all_lines_map.items():
        if lid not in group_ids and line_to_node.get(lid) and line_to_node[lid] in {line_to_node.get(list(group_ids)[0], '')}:
            candidates.append(info)
    candidates.sort(key=lambda l: l['start_seconds'])
    
    # Extend forward
    for c in candidates:
        if max_end - min_start >= MIN_SEEDANCE_DURATION:
            break
        if c['start_seconds'] >= max_end:
            group.append({
                'line_id': c['line_id'], 'speaker': c['speaker'],
                'original': c['dialogue'], 'rewritten': c['dialogue'],
                'start_seconds': c['start_seconds'], 'end_seconds': c['end_seconds'],
                'shot_number': 0,
            })
            max_end = c['end_seconds']
    
    # Extend backward
    for c in reversed(candidates):
        if max_end - min_start >= MIN_SEEDANCE_DURATION:
            break
        if c['end_seconds'] <= min_start:
            group.insert(0, {
                'line_id': c['line_id'], 'speaker': c['speaker'],
                'original': c['dialogue'], 'rewritten': c['dialogue'],
                'start_seconds': c['start_seconds'], 'end_seconds': c['end_seconds'],
                'shot_number': 0,
            })
            min_start = c['start_seconds']
    
    return group


def _split_contiguous(rewrite_lines: List[Dict], max_gap_sec: float = 5.0) -> List[List[Dict]]:
    """Split rewrite lines into contiguous groups, then merge-up groups < MIN_SEEDANCE_DURATION.
    
    After splitting by time gap, any group shorter than MIN_SEEDANCE_DURATION
    merges into its nearest neighbor. Isolated short groups (no neighbor) 
    are left as-is for the caller to handle as original segments.
    """
    if not rewrite_lines:
        return []
    
    # Step 1: Split by time gap
    groups = []
    current = [rewrite_lines[0]]
    for rl in rewrite_lines[1:]:
        prev_end = current[-1].get("end_seconds", 0.0)
        curr_start = rl.get("start_seconds", 0.0)
        if curr_start - prev_end > max_gap_sec:
            groups.append(current)
            current = [rl]
        else:
            current.append(rl)
    groups.append(current)
    
    # Step 2: Merge-up short groups (only if gap to neighbor ≤ max_gap_sec)
    merged = []
    i = 0
    while i < len(groups):
        group = groups[i]
        dur = max(r.get("end_seconds", 0) for r in group) - min(r.get("start_seconds", 0) for r in group)
        
        if dur >= MIN_SEEDANCE_DURATION:
            merged.append(group)
            i += 1
            continue
        
        # Check gap to neighbors before merging
        gap_forward = float('inf')
        if i + 1 < len(groups):
            my_end = max(r.get("end_seconds", 0) for r in group)
            next_start = min(r.get("start_seconds", 0) for r in groups[i + 1])
            gap_forward = next_start - my_end
        
        gap_backward = float('inf')
        if merged:
            prev_end = max(r.get("end_seconds", 0) for r in merged[-1])
            my_start = min(r.get("start_seconds", 0) for r in group)
            gap_backward = my_start - prev_end
        
        if gap_forward <= max_gap_sec:
            groups[i + 1] = group + groups[i + 1]
            i += 1
        elif gap_backward <= max_gap_sec:
            merged[-1].extend(group)
            i += 1
        else:
            merged.append(group)
            i += 1
    
    return merged


def _make_rl_objects(rewrite_lines: List[Dict]) -> List[SimpleNamespace]:
    """Convert rewrite line dicts to SimpleNamespace, adding dialogue=original
    so match_lines_to_nodes can use getattr(line, 'dialogue', '')."""
    result = []
    for rl in rewrite_lines:
        rl_obj = SimpleNamespace(**rl)
        if not getattr(rl_obj, "dialogue", None):
            setattr(rl_obj, "dialogue", rl.get("original", ""))
        result.append(rl_obj)
    return result


def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    script_output = input_data.script_output
    shots = list(script_output.script.shots) if script_output else []
    rewrite_lines_all = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    video_cuts = input_data.video_cut_points
    keyframes = input_data.keyframes
    level = input_data.level

    video_duration = max(
        [s.end_seconds for s in shots if hasattr(s, 'end_seconds')],
        default=60.0,
    )
    cut_boundaries = determine_cut_points(shots, video_cuts, video_duration)

    # ── Build helpers ──────────────────────────────────────────
    # Map line_id → shot info (for time range lookup)
    line_id_to_shot: Dict[str, Any] = {}
    all_lines_map: Dict[str, Dict] = {}
    for shot in shots:
        for line in (shot.lines or []):
            lid = str(getattr(line, 'line_id', ''))
            line_id_to_shot[lid] = shot
            all_lines_map[lid] = {
                'line_id': lid,
                'dialogue': getattr(line, 'dialogue', ''),
                'speaker': getattr(line, 'speaker', ''),
                'start_seconds': getattr(line, 'start_seconds', 0.0),
                'end_seconds': getattr(line, 'end_seconds', 0.0),
            }

    # Map node_id → CanvasNode
    node_map: Dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}

    # ── Separate rewritten vs unchanged lines ──────────────────
    rewritten_lines: List[Dict] = []
    for rl in rewrite_lines_all:
        if str(rl.get("original", "")) != str(rl.get("rewritten", "")):
            rewritten_lines.append(rl)

    # ── Match rewritten lines to nodes ─────────────────────────
    line_confidences: Dict[str, float] = {}
    if rewritten_lines and canvas_nodes:
        rl_objects = _make_rl_objects(rewritten_lines)
        node_line_groups, line_confidences = match_lines_to_nodes(rl_objects, canvas_nodes)
        # node_line_groups: {node_id: [line_id, ...]}
    else:
        node_line_groups = {}

    # ── Build reverse map: line_id → node_id ───────────────────
    line_to_node: Dict[str, str] = {}
    for node_id, lids in node_line_groups.items():
        for lid in lids:
            line_to_node[lid] = node_id

    # ── Track which shots have been handled ────────────────────
    items: List[TimelinePlanItem] = []
    handled_rewrite_line_ids: set[str] = set()

    # ── Per-node: create TimelinePlanItem for rewritten lines ──
    for node_id, line_ids in node_line_groups.items():
        node = node_map.get(node_id)
        node_rewrite_lines = [
            rl for rl in rewritten_lines
            if rl["line_id"] in line_ids
        ]
        if not node_rewrite_lines:
            continue

        # Sort by start time and split into contiguous groups
        node_rewrite_lines.sort(key=lambda rl: rl.get("start_seconds", 0.0))
        contiguous_groups = _split_contiguous(node_rewrite_lines)

        for group in contiguous_groups:
            min_start = min(rl.get("start_seconds", 0.0) for rl in group)
            max_end = max(rl.get("end_seconds", min_start + 1.0) for rl in group)
            duration = max_end - min_start

            if duration < MIN_SEEDANCE_DURATION:
                group = _extend_short_group(group, set(line_ids), all_lines_map, line_to_node)
                min_start = min(rl.get("start_seconds", 0.0) for rl in group)
                max_end = max(rl.get("end_seconds", min_start + 1.0) for rl in group)
                duration = max_end - min_start
                if duration < MIN_SEEDANCE_DURATION:
                    group_ids = {rl["line_id"] for rl in group}
                    handled_rewrite_line_ids.update(group_ids)
                    continue

            first_shot = line_id_to_shot.get(group[0]["line_id"])
            scene_desc = getattr(first_shot, "scene_description", "") if first_shot else ""
            shot_num = group[0].get("shot_number", 0)

            degradation_level = 0
            ref_images = _collect_ref_images(node, keyframes, shot_num)
            if not ref_images:
                degradation_level = 1

            prompt_str = node.prompt if node else ""
            rl_objects = _make_rl_objects(group)
            rewritten_prompt = extract_and_rewrite_prompt(
                prompt_str, rl_objects, scene_desc
            )

            seedance_dur = normalize_seedance_duration(duration)

            group_ids = {rl["line_id"] for rl in group}
            conf_values = [line_confidences.get(lid, 0.0) for lid in group_ids if lid in line_confidences]
            node_confidence = sum(conf_values) / len(conf_values) if conf_values else None

            items.append(TimelinePlanItem(
                shot_id=f"shot_{shot_num}_node_{node_id[:8]}" if node else f"shot_{shot_num}",
                shot_number=shot_num,
                source="seedance",
                start_sec=min_start,
                end_sec=max_end,
                scene_description=scene_desc,
                ref_images=ref_images,
                rewritten_prompt=rewritten_prompt,
                matched_node_id=node_id if node else None,
                match_confidence=node_confidence,
                degradation_level=degradation_level,
                seedance_duration=seedance_dur,
                original_duration=duration,
            ))
            handled_rewrite_line_ids.update(group_ids)

    # ── Remaining shots: original (unchanged dialogue) ─────────
    for idx, shot in enumerate(shots):
        start_s, end_s = cut_boundaries[idx]
        scene_desc = getattr(shot, "scene_description", "") or ""

        # Check if any of this shot's rewritten lines weren't handled
        needs_rewrite, matching = _shot_needs_rewrite(shot, rewrite_lines_all)
        shot_line_ids = {str(rl["line_id"]) for rl in matching}

        if needs_rewrite:
            if shot_line_ids - handled_rewrite_line_ids:
                # This shot has unreplaced rewritten lines → degraded fallback
                unmatched = [rl for rl in matching
                             if rl["line_id"] not in handled_rewrite_line_ids
                             and str(rl.get("original", "")) != str(rl.get("rewritten", ""))]
                if not unmatched:
                    continue
                min_start = min(rl.get("start_seconds", start_s) for rl in unmatched)
                max_end = max(rl.get("end_seconds", end_s) for rl in unmatched)
                if max_end - min_start < MIN_SEEDANCE_DURATION:
                    handled_rewrite_line_ids.update({rl["line_id"] for rl in unmatched})
                    continue
                rl_objects = _make_rl_objects(unmatched)
                rewritten_prompt = extract_and_rewrite_prompt("", rl_objects, scene_desc)
                items.append(TimelinePlanItem(
                    shot_id=f"shot_{shot.shot_number}_fallback",
                    shot_number=shot.shot_number,
                    source="seedance",
                    start_sec=min_start, end_sec=max_end,
                    scene_description=scene_desc,
                    rewritten_prompt=rewritten_prompt,
                    degradation_level=2,
                    seedance_duration=normalize_seedance_duration(max_end - min_start),
                    original_duration=max_end - min_start,
                ))
            continue

        # Shot with NO rewritten lines → original segment
        items.append(TimelinePlanItem(
            shot_id=f"shot_{shot.shot_number}",
            shot_number=shot.shot_number,
            source="original",
            start_sec=start_s, end_sec=end_s,
            scene_description=scene_desc,
            original_duration=end_s - start_s,
        ))

    # Sort by start time for correct playback order
    items.sort(key=lambda i: i.start_sec)

    # Split original items around seedance items (instead of removing entire original)
    seedance_items = [i for i in items if i.source == "seedance"]
    filtered = []
    for item in items:
        if item.source != "original":
            filtered.append(item)
            continue
        # Carve out seedance time ranges from this original segment
        segments = [(item.start_sec, item.end_sec)]
        for si in seedance_items:
            new_segments = []
            for seg_start, seg_end in segments:
                if si.start_sec >= seg_end or si.end_sec <= seg_start:
                    new_segments.append((seg_start, seg_end))
                else:
                    if seg_start < si.start_sec:
                        new_segments.append((seg_start, si.start_sec))
                    if seg_end > si.end_sec:
                        new_segments.append((si.end_sec, seg_end))
            segments = new_segments
            if not segments:
                break
        for seg_start, seg_end in segments:
            if seg_end - seg_start > 0.1:
                filtered.append(TimelinePlanItem(
                    shot_id=f"{item.shot_id}_seg",
                    shot_number=item.shot_number,
                    source="original",
                    start_sec=seg_start,
                    end_sec=seg_end,
                    scene_description=item.scene_description,
                    original_duration=seg_end - seg_start,
                ))
    items = filtered

    return TimelinePlan(
        title=getattr(script_output, "title", "Untitled") if script_output else "Untitled",
        level=level,
        total_duration_sec=video_duration,
        items=items,
        metadata={
            "num_shots": len(shots),
            "num_items": len(items),
            "num_rewritten": sum(1 for i in items if i.source == "seedance"),
            "num_original": sum(1 for i in items if i.source == "original"),
            "node_groups": len(node_line_groups),
        },
    )


def main():
    import argparse
    p = argparse.ArgumentParser(description="Stage 3: Generate timeline plan")
    p.add_argument("--script", required=True)
    p.add_argument("--rewrite", required=True)
    p.add_argument("--canvas")
    p.add_argument("--cuts")
    p.add_argument("--keyframes")
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
    kfs: List[KeyFrame] = []
    if args.keyframes and Path(args.keyframes).exists():
        with open(args.keyframes) as f:
            kfs = [KeyFrame(time_sec=k["time_sec"], image_path=k["image_path"], shot_number=k["shot_number"]) for k in json.load(f)]

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

    inp = Stage3Input(
        script_output=_SW(script_data) if script_data else None,
        video_cut_points=cuts, keyframes=kfs,
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
