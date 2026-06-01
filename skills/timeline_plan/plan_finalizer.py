"""Plan Finalizer: GenerationWindows -> executable TimelinePlan."""
from __future__ import annotations

import logging
from typing import Any

from skills.timeline_plan.models import TimelinePlan, TimelinePlanItem, GenerationWindow

logger = logging.getLogger(__name__)


def _carve_out(segments: list[tuple[float, float]], carve_start: float, carve_end: float) -> list[tuple[float, float]]:
    """Remove [carve_start, carve_end] from segment ranges."""
    result: list[tuple[float, float]] = []
    for seg_start, seg_end in segments:
        if carve_start >= seg_end or carve_end <= seg_start:
            result.append((seg_start, seg_end))
        else:
            if seg_start < carve_start:
                result.append((seg_start, carve_start))
            if seg_end > carve_end:
                result.append((carve_end, seg_end))
    return result


def finalize_timeline_plan(
    windows: list[GenerationWindow],
    shots: list[Any],
    video_duration: float,
    title: str,
    level: str,
) -> TimelinePlan:
    items: list[TimelinePlanItem] = []

    # Step 1: Modified items from windows, excluding degraded-unmatched
    valid_windows = [w for w in windows if w.degradation_level < 5 and w.rewritten_prompt]
    fallback_windows = [w for w in windows if w not in valid_windows]

    for window in fallback_windows:
        logger.warning("Window %s degraded to original fallback: %s",
                       window.window_id, window.degradation_reason)

    for window in valid_windows:
        primary_atom = window.atoms[0] if window.atoms else None
        items.append(TimelinePlanItem(
            shot_id=window.window_id,
            shot_number=primary_atom.primary_shot_number if primary_atom else 0,
            source="modified",
            start_sec=window.start_sec, end_sec=window.end_sec,
            scene_description=primary_atom.scene_description if primary_atom else "",
            ref_images=window.ref_images,
            rewritten_prompt=window.rewritten_prompt,
            matched_node_id=window.matched_node_id,
            match_confidence=window.match_confidence,
            original_duration=window.duration_sec,
            covered_line_ids=window.covered_line_ids,
            degradation_level=window.degradation_level,
            degradation_reason=window.degradation_reason,
        ))

    # Step 2: Carve valid windows only (fallback windows keep original content)
    original_segments = [(0.0, video_duration)]
    for window in valid_windows:
        original_segments = _carve_out(original_segments, window.start_sec, window.end_sec)

    for seg_start, seg_end in original_segments:
        if seg_end - seg_start > 0.1:
            items.append(TimelinePlanItem(
                shot_id=f"orig_{seg_start:.1f}",
                shot_number=0, source="original",
                start_sec=seg_start, end_sec=seg_end,
                scene_description="",
                original_duration=seg_end - seg_start,
            ))

    # Step 3: Sort
    items.sort(key=lambda i: i.start_sec)

    num_modified = sum(1 for i in items if i.source == "modified")
    num_original = sum(1 for i in items if i.source == "original")

    plan = TimelinePlan(
        title=title, level=level,
        total_duration_sec=video_duration, items=items,
        metadata={"num_items": len(items), "num_modified": num_modified, "num_original": num_original},
    )

    # Step 4: Validation
    from skills.timeline_plan.validator import validate_timeline_item, validate_timeline_items
    errors: list[str] = []
    for item in plan.items:
        errors.extend(validate_timeline_item(item))
    errors.extend(validate_timeline_items(plan.items, video_duration))

    blocking = [e for e in errors if any(kw in e.lower() for kw in (
        "overlap", "start_sec (", "missing start_sec", ">= end_sec",
        "empty rewritten_prompt", "covered by both",
        "gap at start", "gap at end", "empty shot_id", "invalid source",
    ))]
    if blocking:
        raise ValueError(f"Validation FAILED with {len(blocking)} blocking errors:\n" + "\n".join(f"  - {e}" for e in blocking))
    if errors:
        logger.warning("%d non-blocking validation warnings", len(errors))

    return plan
