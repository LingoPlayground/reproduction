"""Cut point fusion: merge ScriptShot boundaries with PySceneDetect cut points.

Algorithm:
1. For each ScriptShot, clamp start/end to [0, video_duration].
2. Within a tolerance window, find the nearest PySceneDetect cut point.
3. If found, use the detected point; otherwise keep the LLM boundary.
4. Fill gaps between adjacent shots using intermediate cut points or extend.
"""
from __future__ import annotations

import logging


from skills.common.models import CutPoint

logger = logging.getLogger(__name__)


def find_nearest_cut(
    cuts: list[CutPoint],
    target: float,
    tolerance: float = 0.5,
) -> CutPoint | None:
    candidates = [
        cut for cut in cuts
        if abs(cut.time_sec - target) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.time_sec - target))


def determine_cut_points(
    script_shots: list,
    scene_cuts: list[CutPoint],
    video_duration: float,
    tolerance: float = 0.5,
) -> list[tuple[float, float]]:
    """Determine precise cut positions for all shots in a video.

    For each ScriptShot, clamps boundaries to valid range, then refines
    using detected scene cuts. Fills gaps between adjacent shots.
    """
    results: list[tuple[float, float]] = []

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

    return _snap_to_neighbor(results)


def _snap_to_neighbor(
    boundaries: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Ensure no gaps between adjacent shot boundaries."""
    if len(boundaries) <= 1:
        return boundaries

    filled = [list(boundaries[0])]
    for i in range(1, len(boundaries)):
        prev_end = filled[i - 1][1]
        curr_start, curr_end = boundaries[i]
        if curr_start > prev_end:
            gap = curr_start - prev_end
            if gap > 1.0:
                logger.warning(
                    "Gap %.1fs between shots absorbed by extending previous shot", gap
                )
            filled[i - 1][1] = curr_start
        filled.append([curr_start, curr_end])

    return [(s, e) for s, e in filled]
