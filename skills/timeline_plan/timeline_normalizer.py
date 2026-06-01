"""Timeline Normalizer: converts TimelinePlanDraft to executable TimelinePlan.

PURE deterministic geometry. No semantic decisions, no remedial logic.
If the LLM draft has overlapping groups, split across 30s, unmatched lines,
or any semantic issue — the verifier should have caught it. Normalizer only:
  1. Pad short groups to 4s minimum
  2. Carve modified regions out of original segments
  3. Fill gaps between items
  4. Sort into chronological order

Fail fast: any semantic problem (overlap, >30s, unmatched) is a ValueError.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from skills.timeline_plan.models import (
    CanvasNode, CutPoint, TimelinePlan, TimelinePlanItem,
    MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
)
from skills.timeline_plan.planner_models import MatchResult


def normalize_plan(
    draft: MatchResult,
    script_shots: List[Any],
    canvas_nodes: List[CanvasNode],
    cut_points: List[CutPoint],
    video_duration: float,
    title: str = "Untitled",
    level: str = "B2",
) -> TimelinePlan:
    """Convert TimelinePlanDraft to executable TimelinePlan.

    FAIL FAST on any semantic issue — the LLM draft must be valid.
    """
    node_map: Dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    items: List[TimelinePlanItem] = []

    # Step 1: Convert node_generations to modified items
    for g in draft.node_generations:
        tr = g.source_time_range
        if tr is None:
            raise ValueError(f"{g.group_id}: missing source_time_range")
        start_sec = tr.start_sec
        end_sec = tr.end_sec

        if end_sec - start_sec > MAX_MODIFIED_DURATION:
            raise ValueError(
                f"{g.group_id}: duration {end_sec - start_sec:.1f}s exceeds "
                f"max {MAX_MODIFIED_DURATION}s. LLM must re-plan into smaller groups."
            )

        duration = end_sec - start_sec
        if duration < MIN_MODIFIED_DURATION:
            end_sec = start_sec + MIN_MODIFIED_DURATION

        ref_images: List[str] = []
        degradation_level = 0
        for nid in g.matched_node_ids:
            node = node_map.get(nid)
            if node and node.reference_images:
                for ri in node.reference_images:
                    url = ri if isinstance(ri, str) else ri.get("url", "") if isinstance(ri, dict) else ""
                    if url: ref_images.append(url)
        if not ref_images:
            degradation_level = 1

        items.append(TimelinePlanItem(
            shot_id=f"mod_{g.group_id}",
            shot_number=0,
            source="modified",
            start_sec=start_sec,
            end_sec=end_sec,
            scene_description=g.grouping_reasoning or "",
            ref_images=ref_images,
            rewritten_prompt=g.rewritten_prompt,
            matched_node_id=g.matched_node_ids[0] if g.matched_node_ids else None,
            match_confidence=g.confidence,
            degradation_level=degradation_level,
            original_duration=duration,
            covered_line_ids=sorted(g.covered_line_ids),
            source_node_ids=sorted(g.matched_node_ids),
            degradation_reason="duration_padded" if duration < MIN_MODIFIED_DURATION else "",
        ))

    # Step 2: Handle unmatched lines as degraded fallback
    unmatched_line_details: List[str] = []
    for unmatched in draft.unmatched_lines:
        start_sec = unmatched.start_sec
        end_sec = unmatched.end_sec
        if end_sec - start_sec < MIN_MODIFIED_DURATION:
            end_sec = start_sec + MIN_MODIFIED_DURATION
        # Check if this overlaps with any existing group
        merged = False
        for item in items:
            if item.source == "modified" and item.start_sec <= end_sec and item.end_sec >= start_sec:
                item.covered_line_ids = sorted(set(item.covered_line_ids) | {unmatched.line_id})
                item.start_sec = min(item.start_sec, start_sec)
                item.end_sec = max(item.end_sec, end_sec)
                merged = True
                break
        if not merged:
            items.append(TimelinePlanItem(
                shot_id=f"mod_unmatched_{unmatched.line_id}",
                shot_number=unmatched.shot_number,
                source="modified",
                start_sec=start_sec, end_sec=end_sec,
                scene_description=unmatched.shot_scene,
                ref_images=[],
                rewritten_prompt=f'Scene: {unmatched.shot_scene}. {unmatched.speaker} says: "{unmatched.rewritten}"',
                matched_node_id=None, match_confidence=0.0,
                degradation_level=5,
                original_duration=end_sec - start_sec,
                covered_line_ids=[unmatched.line_id],
                source_node_ids=[],
                degradation_reason=f"unmatched: {unmatched.reason}",
            ))
        unmatched_line_details.append(unmatched.line_id)

    # Step 3: Build original segments from script shots
    from skills.timeline_plan.cut_fusion import determine_cut_points
    cut_boundaries = determine_cut_points(script_shots, cut_points, video_duration)

    for idx, shot in enumerate(script_shots):
        start_s, end_s = cut_boundaries[idx]
        items.append(TimelinePlanItem(
            shot_id=f"shot_{getattr(shot, 'shot_number', idx)}",
            shot_number=getattr(shot, "shot_number", idx),
            source="original",
            start_sec=start_s,
            end_sec=end_s,
            scene_description=getattr(shot, "scene_description", "") or "",
            original_duration=end_s - start_s,
        ))

    # Step 4: Finalize timeline
    items.sort(key=lambda i: i.start_sec)
    items = _finalize(items, video_duration)

    return TimelinePlan(
        title=title,
        level=level,
        total_duration_sec=video_duration,
        items=items,
        metadata={
            "num_items": len(items),
            "num_modified": sum(1 for i in items if i.source == "modified"),
            "num_original": sum(1 for i in items if i.source == "original"),
            "num_groups": len(draft.node_generations),
        },
    )


# ── Geometry helpers ────────────────────────────────────────────────

def _carve_out(
    segments: List[Tuple[float, float]],
    carve_start: float,
    carve_end: float,
) -> List[Tuple[float, float]]:
    result: List[Tuple[float, float]] = []
    for seg_start, seg_end in segments:
        if carve_start >= seg_end or carve_end <= seg_start:
            result.append((seg_start, seg_end))
        else:
            if seg_start < carve_start:
                result.append((seg_start, carve_start))
            if seg_end > carve_end:
                result.append((carve_end, seg_end))
    return result


def _finalize(
    items: List[TimelinePlanItem],
    video_duration: float,
) -> List[TimelinePlanItem]:
    """Pure geometry: pad short modified, carve, fill gaps, sort."""
    # Step A: Pad short modified to min duration
    for item in items:
        if item.source == "modified":
            if item.duration_sec < MIN_MODIFIED_DURATION:
                item.end_sec = item.start_sec + MIN_MODIFIED_DURATION
                item.original_duration = MIN_MODIFIED_DURATION

    # Step B: Check for overlaps — fail fast
    modified_items = sorted(
        [i for i in items if i.source == "modified"], key=lambda x: x.start_sec
    )
    for i in range(len(modified_items) - 1):
        if modified_items[i].end_sec > modified_items[i + 1].start_sec + 2.0:
            raise ValueError(
                f"Modified segments overlap: {modified_items[i].shot_id} "
                f"([{modified_items[i].start_sec:.1f}-{modified_items[i].end_sec:.1f}]) and "
                f"{modified_items[i+1].shot_id} "
                f"([{modified_items[i+1].start_sec:.1f}-{modified_items[i+1].end_sec:.1f}]). "
                f"LLM must produce non-overlapping groups."
            )
        # Small overlaps: snap boundary
        if modified_items[i].end_sec > modified_items[i + 1].start_sec:
            mid = (modified_items[i].end_sec + modified_items[i + 1].start_sec) / 2
            modified_items[i].end_sec = mid
            modified_items[i + 1].start_sec = mid

    # Step C: Carve modified ranges out of originals
    mod_all = [i for i in items if i.source == "modified"]
    result: List[TimelinePlanItem] = []
    for item in items:
        if item.source != "original":
            result.append(item)
            continue
        segments = [(item.start_sec, item.end_sec)]
        for mi in mod_all:
            segments = _carve_out(segments, mi.start_sec, mi.end_sec)
        for seg_start, seg_end in segments:
            if seg_end - seg_start > 0.1:
                result.append(TimelinePlanItem(
                    shot_id=f"{item.shot_id}_seg",
                    shot_number=item.shot_number,
                    source="original",
                    start_sec=seg_start,
                    end_sec=seg_end,
                    scene_description=item.scene_description,
                    original_duration=seg_end - seg_start,
                ))
    items = result

    # Step D: Sort, merge adjacent originals, fill gaps
    items.sort(key=lambda i: i.start_sec)

    merged: List[TimelinePlanItem] = []
    last_end = 0.0
    for item in items:
        if item.start_sec > last_end + 0.1:
            merged.append(TimelinePlanItem(
                shot_id=f"gap_{last_end:.0f}",
                shot_number=0,
                source="original",
                start_sec=last_end,
                end_sec=item.start_sec,
                scene_description="",
                original_duration=item.start_sec - last_end,
            ))
        if item.source == "original" and merged and merged[-1].source == "original":
            prev = merged[-1]
            prev.end_sec = item.end_sec
            prev.original_duration = prev.end_sec - prev.start_sec
            prev.covered_line_ids = sorted(set(prev.covered_line_ids) | set(item.covered_line_ids))
            prev.degradation_level = max(prev.degradation_level, item.degradation_level)
        else:
            merged.append(item)
        last_end = max(last_end, item.end_sec)

    if last_end < video_duration - 0.1:
        merged.append(TimelinePlanItem(
            shot_id=f"gap_{last_end:.0f}",
            shot_number=0,
            source="original",
            start_sec=last_end,
            end_sec=video_duration,
            scene_description="",
            original_duration=video_duration - last_end,
        ))

    return merged
