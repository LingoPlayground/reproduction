"""Cut point fusion: merge ScriptShot boundaries with PySceneDetect cut points.

Algorithm:
1. For each ScriptShot, clamp start/end to [0, video_duration].
2. Within a tolerance window, find the nearest PySceneDetect cut point.
3. If found, use the detected point; otherwise keep the LLM boundary.
4. Fill gaps between adjacent shots using intermediate cut points or extend.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from skills.timeline_plan.models import CutPoint


def find_nearest_cut(
    cuts: List[CutPoint],
    target: float,
    tolerance: float = 0.5,
) -> Optional[CutPoint]:
    """Find the cut point closest to target within tolerance.

    Args:
        cuts: List of detected cut points.
        target: Target time in seconds.
        tolerance: Maximum allowed distance from target.

    Returns:
        The nearest CutPoint within tolerance, or None.
    """
    candidates = [
        cut for cut in cuts
        if abs(cut.time_sec - target) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.time_sec - target))


def fuse_cut_boundary(
    shot: Any,
    video_cut_points: List[CutPoint],
    tolerance: float = 0.5,
) -> Tuple[float, float]:
    """Fuse a single ScriptShot boundary with nearby PySceneDetect cut points.

    Args:
        shot: ScriptShot object with .start_seconds and .end_seconds.
        video_cut_points: All detected cut points from the original video.
        tolerance: Max distance window for refinement.

    Returns:
        (refined_start, refined_end) in seconds.
    """
    start_cut = find_nearest_cut(video_cut_points, shot.start_seconds, tolerance)
    end_cut = find_nearest_cut(video_cut_points, shot.end_seconds, tolerance)
    return (
        start_cut.time_sec if start_cut else shot.start_seconds,
        end_cut.time_sec if end_cut else shot.end_seconds,
    )


def determine_cut_points(
    script_shots: List[Any],
    scene_cuts: List[CutPoint],
    video_duration: float,
    tolerance: float = 0.5,
) -> List[Tuple[float, float]]:
    """Determine precise cut positions for all shots in a video.

    For each ScriptShot, clamps boundaries to valid range, then refines
    using detected scene cuts.  Fills gaps between adjacent shots.

    Args:
        script_shots: List of ScriptShot objects from Stage 1 output.
        scene_cuts: List of CutPoint objects from PySceneDetect.
        video_duration: Total video duration in seconds.
        tolerance: Max distance for cut refinement.

    Returns:
        List of (start_seconds, end_seconds) tuples, one per shot.
    """
    results: List[Tuple[float, float]] = []

    for shot in script_shots:
        raw_start = max(0.0, min(float(shot.start_seconds), video_duration))
        raw_end = max(0.0, min(float(shot.end_seconds), video_duration))

        if raw_end - raw_start < 1.0:
            raw_end = min(raw_start + 1.0, video_duration)

        start_cut = find_nearest_cut(scene_cuts, raw_start, tolerance)
        end_cut = find_nearest_cut(scene_cuts, raw_end, tolerance)

        final_start = start_cut.time_sec if start_cut else raw_start
        final_end = end_cut.time_sec if end_cut else raw_end

        results.append((final_start, final_end))

    results = _snap_to_neighbor(results)
    return results


def _snap_to_neighbor(
    boundaries: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Ensure no gaps between adjacent shot boundaries."""
    if len(boundaries) <= 1:
        return boundaries

    filled = [list(boundaries[0])]
    for i in range(1, len(boundaries)):
        prev_end = filled[i - 1][1]
        curr_start, curr_end = boundaries[i]
        if curr_start > prev_end:
            filled[i - 1][1] = curr_start
        filled.append([curr_start, curr_end])

    return [(s, e) for s, e in filled]
