"""Edit Atom Builder: Stage 1 shot + rewrite lines -> EditAtom list."""
from __future__ import annotations

import logging
import re
from typing import Any

from skills.timeline_plan.models import CutPoint, EditAtom, AtomLine

logger = logging.getLogger(__name__)

CLUSTER_GAP_SEC = 1.5
SNAP_TOLERANCE_SEC = 0.5


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text


def _is_rewritten(original: str, rewritten: str) -> bool:
    return _normalize_text(original) != _normalize_text(rewritten)


def _snap_boundary(target: float, cut_times: list[float]) -> float:
    best = target
    best_dist = float('inf')
    for ct in cut_times:
        dist = abs(ct - target)
        if dist <= SNAP_TOLERANCE_SEC and dist < best_dist:
            best = ct
            best_dist = dist
    return best


def _cuts_any_line(boundary: float, cluster_lines: list[dict]) -> bool:
    for rl in cluster_lines:
        ls = float(rl.get("start_seconds", 0.0))
        le = float(rl.get("end_seconds", 0.0))
        if ls < boundary < le:
            return True
    return False


def build_edit_atoms(
    script_shots: list[Any],
    rewrite_lines: list[dict],
    scene_cuts: list[CutPoint],
    video_duration: float,
) -> list[EditAtom]:
    if not rewrite_lines:
        return []

    lines_by_shot: dict[int, list[dict]] = {}
    for rl in rewrite_lines:
        sn = int(rl.get("shot_number", 0))
        lines_by_shot.setdefault(sn, []).append(rl)

    atoms: list[EditAtom] = []
    atom_counter = 0
    cut_times = sorted(c.time_sec for c in scene_cuts)

    for shot in script_shots:
        sn = getattr(shot, "shot_number", 0)
        scene_desc = getattr(shot, "scene_description", "") or ""
        shot_lines = sorted(
            lines_by_shot.get(sn, []),
            key=lambda rl: float(rl.get("start_seconds", 0.0)),
        )

        clusters: list[list[dict]] = []
        current: list[dict] = []

        for rl in shot_lines:
            if _is_rewritten(str(rl.get("original", "")), str(rl.get("rewritten", ""))):
                if not current:
                    current = [rl]
                else:
                    prev_end = float(current[-1].get("end_seconds", 0.0))
                    curr_start = float(rl.get("start_seconds", 0.0))
                    if curr_start - prev_end > CLUSTER_GAP_SEC:
                        clusters.append(current)
                        current = [rl]
                    else:
                        current.append(rl)
            else:
                # Unchanged lines do NOT auto-merge clusters.
                if current:
                    clusters.append(current)
                    current = []

        if current:
            clusters.append(current)

        for cluster in clusters:
            atom_counter += 1
            start_sec = min(float(rl.get("start_seconds", 0.0)) for rl in cluster)
            end_sec = max(float(rl.get("end_seconds", 0.0)) for rl in cluster)

            snapped_start = _snap_boundary(start_sec, cut_times)
            snapped_end = _snap_boundary(end_sec, cut_times)

            if not _cuts_any_line(snapped_start, cluster):
                start_sec = snapped_start
            if not _cuts_any_line(snapped_end, cluster):
                end_sec = snapped_end

            atom_lines = [
                AtomLine(
                    line_id=str(rl.get("line_id", "")),
                    speaker=str(rl.get("speaker", "")),
                    original=str(rl.get("original", "")),
                    rewritten=str(rl.get("rewritten", "")),
                    start_sec=float(rl.get("start_seconds", 0.0)),
                    end_sec=float(rl.get("end_seconds", 0.0)),
                    shot_scene=str(rl.get("shot_scene", "")),
                )
                for rl in cluster
            ]

            atoms.append(EditAtom(
                atom_id=f"atom_{atom_counter:03d}",
                shot_numbers=[sn],
                primary_shot_number=sn,
                start_sec=start_sec,
                end_sec=end_sec,
                scene_description=scene_desc,
                lines=atom_lines,
                boundary_reason="asr_line_range",
            ))

    atoms.sort(key=lambda a: a.start_sec)
    return atoms
