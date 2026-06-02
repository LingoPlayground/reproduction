"""Edit Atom Builder: Stage 1 shot + rewrite lines -> EditAtom list."""
from __future__ import annotations

import logging
import re


from skills.common.models import CutPoint
from skills.timeline_plan.models import EditAtom, AtomLine, _normalize_text

logger = logging.getLogger(__name__)

CLUSTER_GAP_SEC = 1.5
SNAP_TOLERANCE_SEC = 0.5


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


def _scene_similarity(desc_a: str, desc_b: str) -> bool:
    if not desc_a or not desc_b:
        return False

    def tokens(text: str) -> set[str]:
        return set(re.findall(r'\w+', text.lower()))

    a_tokens = tokens(desc_a)
    b_tokens = tokens(desc_b)
    if not a_tokens or not b_tokens:
        return False

    overlap = a_tokens & b_tokens
    ratio = len(overlap) / min(len(a_tokens), len(b_tokens))
    return ratio >= 0.45


def _is_effectively_unchanged(original: str, rewritten: str) -> bool:
    """Check if a rewrite is effectively unchanged.

    More aggressive than AtomLine.is_rewritten — catches empty rewrites
    and short interjections that differ textually but carry no semantic change.
    When this returns True, the line's rewritten field is reset to original
    so no EditAtom is created for it.
    """
    if not rewritten or not rewritten.strip():
        return True
    norm_orig = re.sub(r'[^\w\s]', '', original.strip().lower())
    norm_rew = re.sub(r'[^\w\s]', '', rewritten.strip().lower())
    if norm_orig == norm_rew:
        return True
    short_words = {'what', 'huh', 'okay', 'ok', 'yeah', 'yes', 'no', 'hi', 'hey',
                   'oh', 'uh', 'right', 'thanks', 'bye', 'stop', 'wait',
                   'come in', 'good afternoon', 'good morning', 'good evening',
                   'mmhmm', 'hmm'}
    if norm_orig in short_words and norm_rew in short_words:
        return True
    return False


def build_edit_atoms(
    script_shots: list,
    rewrite_lines: list[dict],
    scene_cuts: list[CutPoint],
    video_duration: float,
) -> list[EditAtom]:
    if not rewrite_lines:
        return []

    # Build working copies so we never mutate caller data
    working_lines: list[dict] = []
    for rl in rewrite_lines:
        orig = str(rl.get("original", ""))
        rew_raw = rl.get("rewritten")
        rew = str(rew_raw) if rew_raw is not None else ""
        
        if _is_effectively_unchanged(orig, rew):
            working_lines.append({**rl, "rewritten": orig})
        else:
            working_lines.append(dict(rl))
    
    # Use working_lines instead of rewrite_lines from here on
    rewrite_lines = working_lines

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

    # Cross-shot merge: adjacent shots with similar scene descriptions
    merged_atoms: list[EditAtom] = []
    for a in atoms:
        if not merged_atoms:
            merged_atoms.append(a)
            continue
        prev = merged_atoms[-1]
        gap = a.start_sec - prev.end_sec
        same_scene = _scene_similarity(prev.scene_description, a.scene_description)
        
        # Check speaker continuity
        prev_speakers = {l.speaker for l in prev.lines}
        curr_speakers = {l.speaker for l in a.lines}
        shared_speakers = bool(prev_speakers & curr_speakers)
        
        if gap <= 1.0 and prev.primary_shot_number != a.primary_shot_number:
            # Merge conditions: scene similarity OR speaker continuity
            should_merge = same_scene or (shared_speakers and gap <= 0.7)
            if should_merge:
                prev.end_sec = max(prev.end_sec, a.end_sec)
                prev.lines.extend(a.lines)
                prev.shot_numbers = sorted(set(prev.shot_numbers + a.shot_numbers))
                prev.boundary_reason = "cross_shot_merge_speaker" if shared_speakers and not same_scene else "cross_shot_merge"
                prev.source_cut_times = sorted(set(prev.source_cut_times + a.source_cut_times))
            else:
                merged_atoms.append(a)
        else:
            merged_atoms.append(a)
    atoms = merged_atoms

    return atoms
