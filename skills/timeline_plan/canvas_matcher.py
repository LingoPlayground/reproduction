"""Canvas node matching by dialogue text + scene description similarity.

Uses text overlap scoring (inspired by pipeline.py's fuzzy_match approach) to
find the best canvas node for a given ScriptShot.  Matching is NOT
line-level precise — it only needs to be good enough for reference
image extraction and prompt fragment sourcing.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from skills.timeline_plan.models import CanvasNode

TEXT_OVERLAP_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.3


def _normalize(text: str) -> str:
    """Normalize text: lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def text_overlap_score(dialogue_text: str, node_prompt: str) -> float:
    """Compute how much dialogue text appears in the node prompt.

    Uses word-level overlap with sliding window for robustness against
    ASR noise, punctuation differences, and mixed Chinese/English text.

    Args:
        dialogue_text: Combined dialogue from a ScriptShot's lines.
        node_prompt: Full prompt text from a canvas node.

    Returns:
        Score between 0.0 (no overlap) and 1.0 (full overlap).
    """
    d_norm = _normalize(dialogue_text)
    p_norm = _normalize(node_prompt)

    if not d_norm or not p_norm:
        return 0.0

    d_words = [w for w in d_norm.split() if len(w) >= 2]
    if not d_words:
        return 0.0

    hits = 0
    for w in d_words:
        if w in p_norm:
            hits += 1

    for window_size in range(min(4, len(d_words)), 0, -1):
        for i in range(len(d_words) - window_size + 1):
            phrase = " ".join(d_words[i : i + window_size])
            if phrase in p_norm:
                hits += window_size * 0.5
                break

    raw = hits / max(len(d_words), 1)
    return min(1.0, raw)


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Simple word-overlap based semantic similarity."""
    a_words = set(_normalize(text_a).split())
    b_words = set(_normalize(text_b).split())
    if not a_words or not b_words:
        return 0.0
    intersection = a_words & b_words
    return len(intersection) / max(len(a_words | b_words), 1)


def match_canvas_node_for_shot(
    shot: Any,
    nodes: List[CanvasNode],
    rewrite_lines: Optional[List[Any]] = None,
) -> Tuple[Optional[CanvasNode], float]:
    """Match a ScriptShot to the best canvas node.

    Priority signals:
    1. Dialogue text overlap with node prompt (primary)
    2. Scene description semantic similarity (tiebreaker)

    Args:
        shot: ScriptShot with .lines[] and .scene_description.
        nodes: All available canvas nodes.
        rewrite_lines: Optional rewrite lines.

    Returns:
        (matched_node, confidence) tuple, or (None, 0.0).
    """
    if not nodes:
        return None, 0.0

    dialogue_text = " ".join(
        line.dialogue for line in (shot.lines or [])
        if getattr(line, "dialogue", "")
    )

    candidates: List[Tuple[CanvasNode, float]] = []
    for node in nodes:
        score = text_overlap_score(dialogue_text, node.prompt)
        if score >= TEXT_OVERLAP_THRESHOLD:
            candidates.append((node, score))

    if not candidates:
        return None, 0.0

    if len(candidates) > 1 and shot.scene_description:
        candidates.sort(
            key=lambda c: (
                c[1],
                _semantic_similarity(shot.scene_description, c[0].prompt),
            ),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda c: c[1], reverse=True)

    best_node, confidence = candidates[0]
    if confidence >= CONFIDENCE_THRESHOLD:
        return best_node, confidence
    return None, 0.0
