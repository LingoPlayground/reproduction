"""Generation Window Resolver: EditAtoms -> executable >=4s GenerationWindows."""
from __future__ import annotations

import logging
from typing import Optional

from skills.timeline_plan.models import (
    EditAtom, AtomLine, GenerationWindow, CanvasNode,
    MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
)

logger = logging.getLogger(__name__)


def resolve_generation_windows(
    atoms: list[EditAtom],
    all_lines: list[AtomLine],
    canvas_nodes: list[CanvasNode],
    video_duration: float,
    min_duration_sec: float = MIN_MODIFIED_DURATION,
    max_duration_sec: float = MAX_MODIFIED_DURATION,
) -> list[GenerationWindow]:
    if not atoms:
        return []

    node_map: dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    windows: list[GenerationWindow] = []
    window_counter = 0

    sorted_atoms = sorted(atoms, key=lambda a: a.start_sec)

    # Step 1: Group atoms with same node + small gap
    groups: list[list[EditAtom]] = []
    current: list[EditAtom] = [sorted_atoms[0]]

    for atom in sorted_atoms[1:]:
        prev = current[-1]
        gap = atom.start_sec - prev.end_sec
        same_node = (
            prev.matched_node_id
            and atom.matched_node_id
            and prev.matched_node_id == atom.matched_node_id
        )
        if same_node and gap <= 2.0:
            current.append(atom)
        else:
            groups.append(current)
            current = [atom]
    groups.append(current)

    # Step 2: Build GenerationWindow per group
    for group in groups:
        window_counter += 1
        group_start = min(a.start_sec for a in group)
        group_end = max(a.end_sec for a in group)
        duration = group_end - group_start

        # Expand short windows to at least min_duration_sec
        if duration < min_duration_sec:
            deficit = min_duration_sec - duration
            right_room = video_duration - group_end
            right_expand = min(deficit, right_room)
            left_room = group_start
            left_expand = min(deficit - right_expand, left_room)
            group_end += right_expand
            group_start -= left_expand

        group_start = max(0.0, group_start)
        group_end = min(video_duration, group_end)

        primary = group[0]
        matched_nid = primary.matched_node_id

        ref_images: list[str] = []
        degradation_level = 0
        degradation_reason = ""

        if matched_nid and matched_nid in node_map:
            node = node_map[matched_nid]
            for ri in node.reference_images:
                url = ri if isinstance(ri, str) else ri.get("url", "") if isinstance(ri, dict) else ""
                if url:
                    ref_images.append(url)
            if not ref_images:
                degradation_level = max(degradation_level, 1)
                degradation_reason = "no_ref_images_in_node"
        else:
            degradation_level = 5
            degradation_reason = "unmatched_atom"
            matched_nid = None

        windows.append(GenerationWindow(
            window_id=f"window_{window_counter:03d}",
            start_sec=group_start,
            end_sec=group_end,
            atoms=group,
            matched_node_id=matched_nid,
            match_confidence=primary.match_confidence,
            ref_images=ref_images,
            degradation_level=degradation_level,
            degradation_reason=degradation_reason,
        ))

    logger.info("Window resolver: %d atoms -> %d windows", len(atoms), len(windows))
    return windows
