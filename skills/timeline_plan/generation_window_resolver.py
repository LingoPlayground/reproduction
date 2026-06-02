"""Generation Window Resolver: EditAtoms -> executable >=4s GenerationWindows."""
from __future__ import annotations

import logging

from skills.timeline_plan.models import (
    EditAtom, AtomLine, GenerationWindow, CanvasNode, WindowPlanDraft,
    MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
)

logger = logging.getLogger(__name__)


def _snap_to_safe_boundaries(
    start_sec: float,
    end_sec: float,
    all_lines: list[AtomLine],
    video_duration: float,
) -> tuple[float, float]:
    """Expand boundaries outward so they never cut through an ASR line."""
    if not all_lines:
        return start_sec, end_sec

    lines_sorted = sorted(all_lines, key=lambda l: l.start_sec)

    for line in lines_sorted:
        if line.start_sec < start_sec < line.end_sec:
            start_sec = line.start_sec
            break

    for line in reversed(lines_sorted):
        if line.start_sec < end_sec < line.end_sec:
            end_sec = line.end_sec
            break

    return max(0.0, start_sec), min(video_duration, end_sec)


def _expand_to_min_duration(
    start_sec: float,
    end_sec: float,
    all_lines: list[AtomLine],
    video_duration: float,
    min_duration_sec: float,
) -> tuple[float, float]:
    """Expand a range to min duration, preferring right side then left side."""
    start_sec = max(0.0, start_sec)
    end_sec = min(video_duration, end_sec)

    for _ in range(3):
        duration = end_sec - start_sec
        if duration >= min_duration_sec:
            break
        deficit = min_duration_sec - duration
        right_expand = min(deficit, video_duration - end_sec)
        end_sec += right_expand
        deficit -= right_expand
        if deficit > 0:
            start_sec -= min(deficit, start_sec)
        start_sec, end_sec = _snap_to_safe_boundaries(
            start_sec, end_sec, all_lines, video_duration,
        )

    return start_sec, end_sec


def _mark_fallback(window: GenerationWindow, reason: str) -> None:
    window.degradation_level = max(window.degradation_level, 5)
    window.degradation_reason = (
        f"{window.degradation_reason}; {reason}" if window.degradation_reason else reason
    )


def _group_atoms_from_drafts(
    atoms: list[EditAtom],
    window_drafts: list[WindowPlanDraft],
) -> list[tuple[list[EditAtom], WindowPlanDraft | None]]:
    atom_map = {a.atom_id: a for a in atoms}
    used: set[str] = set()
    groups: list[tuple[list[EditAtom], WindowPlanDraft | None]] = []
    for draft in window_drafts:
        group: list[EditAtom] = []
        for aid in draft.atom_ids:
            if aid not in atom_map:
                logger.warning(
                    "Draft %s references unknown atom %s — skipping", draft.draft_id, aid
                )
                continue
            if aid in used:
                logger.debug(
                    "Draft %s: atom %s already used — skipping", draft.draft_id, aid
                )
                continue
            group.append(atom_map[aid])
        if not group:
            continue
        group.sort(key=lambda a: a.start_sec)
        groups.append((group, draft))
        used.update(a.atom_id for a in group)

    for atom in atoms:
        if atom.atom_id not in used:
            groups.append(([atom], None))
    groups.sort(key=lambda pair: pair[0][0].start_sec)
    return groups


def _fallback_group_atoms(
    sorted_atoms: list[EditAtom],
    max_duration_sec: float,
) -> list[tuple[list[EditAtom], WindowPlanDraft | None]]:
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
            cur_duration = atom.end_sec - current[0].start_sec
            if cur_duration <= max_duration_sec:
                current.append(atom)
            else:
                groups.append(current)
                current = [atom]
        else:
            groups.append(current)
            current = [atom]
    groups.append(current)
    return [(group, None) for group in groups]


def resolve_generation_windows(
    atoms: list[EditAtom],
    all_lines: list[AtomLine],
    canvas_nodes: list[CanvasNode],
    video_duration: float,
    min_duration_sec: float = MIN_MODIFIED_DURATION,
    max_duration_sec: float = MAX_MODIFIED_DURATION,
    window_drafts: list[WindowPlanDraft] | None = None,
) -> list[GenerationWindow]:
    if not atoms:
        return []

    node_map: dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    windows: list[GenerationWindow] = []
    window_counter = 0

    sorted_atoms = sorted(atoms, key=lambda a: a.start_sec)

    if window_drafts:
        groups = _group_atoms_from_drafts(sorted_atoms, window_drafts)
    else:
        groups = _fallback_group_atoms(sorted_atoms, max_duration_sec)

    # Step 2: Build GenerationWindow per group
    for group, draft in groups:
        window_counter += 1
        group_start = min(a.start_sec for a in group)
        group_end = max(a.end_sec for a in group)
        group_start, group_end = _expand_to_min_duration(
            group_start, group_end, all_lines, video_duration, min_duration_sec,
        )

        primary = group[0]
        matched_nid = draft.node_id if draft else primary.matched_node_id

        ref_images: list[str] = []
        degradation_level = 0
        degradation_reason = draft.fallback_reason if draft else ""

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
            match_confidence=draft.confidence if draft else primary.match_confidence,
            ref_images=ref_images,
            degradation_level=degradation_level,
            degradation_reason=degradation_reason,
        ))

    # Resolve overlapping executable windows. If two different-node windows are
    # too close to both remain >= min duration, keep the earlier one and mark
    # the later one as original fallback instead of emitting an invalid clip.
    windows.sort(key=lambda w: w.start_sec)
    resolved: list[GenerationWindow] = []
    for w in windows:
        if not resolved:
            resolved.append(w)
            continue
        prev = resolved[-1]
        if prev.end_sec > w.start_sec + 0.1:
            if prev.matched_node_id and w.matched_node_id and prev.matched_node_id == w.matched_node_id:
                merged_end = max(prev.end_sec, w.end_sec)
                if merged_end - prev.start_sec <= max_duration_sec:
                    prev.end_sec = merged_end
                    prev.atoms.extend(w.atoms)
                    prev.degradation_level = max(prev.degradation_level, w.degradation_level)
                    prev.degradation_reason = (prev.degradation_reason + "; merged_" + w.window_id).strip("; ")
                else:
                    _mark_fallback(w, f"overlap_would_exceed_max_duration_with_{prev.window_id}")
                    resolved.append(w)
            else:
                mid = (prev.end_sec + w.start_sec) / 2
                prev_can_shrink = mid - prev.start_sec >= min_duration_sec
                curr_can_shrink = w.end_sec - mid >= min_duration_sec
                if prev_can_shrink and curr_can_shrink:
                    prev.end_sec = mid
                    w.start_sec = mid
                    resolved.append(w)
                else:
                    _mark_fallback(w, f"overlap_too_tight_with_{prev.window_id}")
                    resolved.append(w)
        else:
            resolved.append(w)
    windows = resolved

    for w in windows:
        if w.degradation_level >= 5:
            continue
        if w.duration_sec < min_duration_sec:
            _mark_fallback(w, "duration_below_min_after_resolution")
        elif w.duration_sec > max_duration_sec:
            _mark_fallback(w, "duration_above_max_after_resolution")

    logger.info("Window resolver: %d atoms -> %d windows", len(atoms), len(windows))
    return windows
