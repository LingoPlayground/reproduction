"""Multilayer prompt validator for canvas node prompt rewriting.

L3: Dialogue inclusion   — rewritten text must appear verbatim in output (in prompt_composer.py)
L4: Style preservation   — original prompt's visual style keywords must be retained at ≥60%
"""
from __future__ import annotations

from typing import List, Tuple


_STYLE_ANCHORS_CN = [
    "美式情景喜剧", "真实短剧", "柔光雾化", "画面通透",
    "电影级布光", "超高清", "电影质感", "浅景深",
    "8k", "4k", "hdr", "固定机位",
]

_STYLE_ANCHORS_EN = [
    "cinematic", "8k", "4k", "hdr", "shallow depth of field",
    "film grain", "soft lighting", "ultra hd",
]


def extract_style_anchors(prompt: str) -> List[str]:
    """Extract style-related keywords present in a canvas node prompt."""
    anchors = []
    prompt_lower = prompt.lower()
    for kw in _STYLE_ANCHORS_CN + _STYLE_ANCHORS_EN:
        if kw.lower() in prompt_lower and kw not in anchors:
            anchors.append(kw)
    return anchors


def validate_style_preservation(
    original_prompt: str,
    rewritten_prompt: str,
    threshold: float = 0.6,
) -> tuple:
    """Check that style anchors from the original prompt are preserved.

    Returns (passes: bool, missing: List[str], preserved_ratio: float).
    """
    anchors = extract_style_anchors(original_prompt)
    if not anchors:
        return True, [], 1.0

    preserved = [kw for kw in anchors if kw.lower() in rewritten_prompt.lower()]
    missing = [kw for kw in anchors if kw not in preserved]
    ratio = len(preserved) / len(anchors)
    passes = ratio >= threshold
    return passes, missing, ratio
