"""Duration Resolver: ensures every rewritten group meets min seedance duration.

Replaces the silent-drop behavior in generate_plan.py. Applies pad strategies
in priority order, never silently discarding rewritten lines.

Strategy priority:
  1. pad_after:  if deficit <= 1.0s, extend end to reach min
  2. pad_before: if deficit <= 1.0s, extend start to reach min
  3. forced_min_duration: last resort, force-extend to min
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


DURATION_STRATEGIES = [
    "pad_after",
    "pad_before",
    "forced_min_duration",
]


def _try_pad_strategy(
    group: List[Dict],
    strategy: str,
    min_duration: float = 4.0,
) -> Optional[List[Dict]]:
    if strategy == "direct":
        return list(group)

    min_start = min(r.get("start_seconds", 0.0) for r in group)
    max_end = max(r.get("end_seconds", min_start + 1.0) for r in group)
    duration = max_end - min_start

    if duration >= min_duration:
        return list(group)

    deficit = min_duration - duration

    if strategy == "pad_after" and deficit <= 1.0:
        group = list(group)
        group[-1]["end_seconds"] = group[-1]["end_seconds"] + deficit
        return group

    if strategy == "pad_before" and deficit <= 1.0:
        group = list(group)
        group[0]["start_seconds"] = max(0.0, group[0]["start_seconds"] - deficit)
        return group

    if strategy == "forced_min_duration":
        group = list(group)
        anchor_start = min(r.get("start_seconds", 0.0) for r in group)
        group[-1]["end_seconds"] = anchor_start + min_duration
        return group

    return None


def resolve_duration(
    group: List[Dict],
    all_lines_map: Dict[str, Dict],
    line_to_node: Dict[str, str],
    min_duration: float = 4.0,
) -> Tuple[List[Dict], str, float]:
    if not group:
        return [], "direct", 0.0

    min_start = min(r.get("start_seconds", 0.0) for r in group)
    max_end = max(r.get("end_seconds", min_start + 1.0) for r in group)
    duration = max_end - min_start

    if duration >= min_duration:
        return list(group), "direct", duration

    for strategy in DURATION_STRATEGIES:
        result = _try_pad_strategy(group, strategy, min_duration)
        if result is not None:
            new_duration = max(
                r.get("end_seconds", 0.0) for r in result
            ) - min(
                r.get("start_seconds", 0.0) for r in result
            )
            return result, strategy, new_duration

    return list(group), "forced_min_duration", min_duration
