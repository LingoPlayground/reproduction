"""Evidence Pack builder: constructs structured evidence packages for the EditPlanner.

Takes the data already assembled in generate_plan.py (rewrite lines, canvas node,
keyframes, scene cuts) and packages it into a typed EvidencePack for LLM consumption.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from skills.timeline_plan.models import (
    CanvasNode,
    CanvasNodeEvidence,
    Constraints,
    EvidencePack,
    KeyFrame,
    CutPoint,
    LineEvidence,
    NodeSection,
    VideoEvidence,
)


def build_evidence_pack(
    group_id: str,
    rewrite_lines: List[Dict],
    all_lines_map: Dict[str, Dict],
    node: Optional[CanvasNode],
    node_sections: Optional[List[NodeSection]] = None,
    matched_section_id: Optional[str] = None,
    keyframes: Optional[List[KeyFrame]] = None,
    scene_cuts: Optional[List[CutPoint]] = None,
) -> EvidencePack:
    group_line_ids = {rl["line_id"] for rl in rewrite_lines}

    target_lines = []
    for rl in rewrite_lines:
        target_lines.append(LineEvidence(
            line_id=rl["line_id"],
            speaker=rl.get("speaker", ""),
            original=rl.get("original", ""),
            rewritten=rl.get("rewritten", ""),
            start_seconds=rl.get("start_seconds", 0.0),
            end_seconds=rl.get("end_seconds", 0.0),
            shot_number=rl.get("shot_number", 0),
            shot_scene=rl.get("shot_scene", ""),
            rewrite_status="rewritten",
        ))

    neighbor_lines = []
    for lid, info in all_lines_map.items():
        if lid not in group_line_ids:
            neighbor_lines.append(LineEvidence(
                line_id=lid,
                speaker=info.get("speaker", ""),
                original=info.get("dialogue", ""),
                rewritten=info.get("dialogue", ""),
                start_seconds=info.get("start_seconds", 0.0),
                end_seconds=info.get("end_seconds", 0.0),
                shot_number=0,
                shot_scene="",
                rewrite_status="unchanged",
            ))

    canvas_evidence = None
    if node:
        canvas_evidence = CanvasNodeEvidence(
            node_id=node.node_id,
            name=getattr(node, "name", ""),
            full_prompt=node.prompt,
            sections=node_sections or [],
            reference_images=node.reference_images,
        )

    video_evidence = None
    if keyframes or scene_cuts:
        video_evidence = VideoEvidence(
            keyframe_paths=[kf.image_path for kf in (keyframes or [])],
            scene_cuts=[c.time_sec for c in (scene_cuts or [])],
        )

    return EvidencePack(
        group_id=group_id,
        target_lines=target_lines,
        neighbor_lines=neighbor_lines,
        canvas_node=canvas_evidence,
        matched_section_id=matched_section_id,
        video=video_evidence,
        constraints=Constraints(),
    )
