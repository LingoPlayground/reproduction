"""Multilayer validator for timeline plan items and prompt quality.

L1: Structure     — field presence, types, value ranges on TimelinePlanItem
L2: Cross-item    — no overlaps, full coverage, no duplicate lines
L3: Dialogue      — rewritten text must appear verbatim (in prompt_composer.py)
L4: Style         — original prompt's visual style keywords retained at ≥60%
L5: LLM-Judge     — semantic consistency of rewritten prompt vs original
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Module-level env loading ────────────────────────────────────────

def _load_env():
    for env_path in [
        str(Path("~/workspace/lingolens/backend/.env").expanduser()),
        str(Path("~/workspace/shakespeare/.env").expanduser()),
    ]:
        if Path(env_path).exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

_load_env()


# ── Style anchors ────────────────────────────────────────────────────

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
    anchors = extract_style_anchors(original_prompt)
    if not anchors:
        return True, [], 1.0
    preserved = [kw for kw in anchors if kw.lower() in rewritten_prompt.lower()]
    missing = [kw for kw in anchors if kw not in preserved]
    ratio = len(preserved) / len(anchors)
    return ratio >= threshold, missing, ratio


# ── L1: Structural validation ────────────────────────────────────────

def validate_timeline_item(item: Any) -> List[str]:
    errors = []
    if not getattr(item, "shot_id", ""):
        errors.append(f"Item has empty shot_id")
    if getattr(item, "source", "") not in ("seedance", "original"):
        errors.append(f"Item {getattr(item, 'shot_id', '?')}: invalid source '{getattr(item, 'source', '')}'")
    start = getattr(item, "start_sec", -1)
    end = getattr(item, "end_sec", -1)
    if start < 0 or end < 0:
        errors.append(f"Item {getattr(item, 'shot_id', '?')}: missing start_sec or end_sec")
    elif start >= end:
        errors.append(f"Item {getattr(item, 'shot_id', '?')}: start_sec ({start}) >= end_sec ({end})")
    if getattr(item, "source", "") == "seedance":
        prompt = getattr(item, "rewritten_prompt", None)
        if not prompt or not prompt.strip():
            errors.append(f"Item {getattr(item, 'shot_id', '?')}: seedance item has empty rewritten_prompt")
    return errors


# ── L2: Cross-item validation ────────────────────────────────────────

def validate_timeline_items(items: List[Any], video_duration: float = 0.0) -> List[str]:
    errors = []
    sorted_items = sorted(items, key=lambda i: getattr(i, "start_sec", 0.0))

    # Check for overlaps
    for i in range(len(sorted_items) - 1):
        curr_end = getattr(sorted_items[i], "end_sec", 0.0)
        next_start = getattr(sorted_items[i + 1], "start_sec", 0.0)
        if curr_end > next_start + 0.05:
            errors.append(
                f"Overlap: item {getattr(sorted_items[i], 'shot_id', '?')} ends at {curr_end:.2f}s "
                f"but item {getattr(sorted_items[i + 1], 'shot_id', '?')} starts at {next_start:.2f}s"
            )

    # Check for duplicate line coverage (skip split seedance segments: same node = same source)
    seen_lines: Dict[str, str] = {}
    seen_nodes: Dict[str, str] = {}
    for item in items:
        node = getattr(item, "matched_node_id", None)
        for lid in getattr(item, "covered_line_ids", []) or []:
            if lid in seen_lines:
                prev_node = seen_nodes.get(lid, "")
                if node and prev_node and node != prev_node:
                    errors.append(
                        f"Line {lid} covered by both {seen_lines[lid]} and {getattr(item, 'shot_id', '?')}"
                    )
            seen_lines[lid] = getattr(item, "shot_id", "?")
            if node:
                seen_nodes[lid] = node

    # Check full duration coverage if provided
    if video_duration > 0 and sorted_items:
        first_start = getattr(sorted_items[0], "start_sec", 0.0)
        last_end = getattr(sorted_items[-1], "end_sec", 0.0)
        if first_start > 0.1:
            errors.append(f"Timeline gap at start: first item begins at {first_start:.2f}s")
        if last_end < video_duration - 0.1:
            errors.append(f"Timeline gap at end: last item ends at {last_end:.2f}s, video is {video_duration:.2f}s")

    return errors


# ── L5: LLM-Judge semantic quality assessment ────────────────────────

def llm_judge_prompt_quality(
    original_prompt: str,
    rewritten_prompt: str,
    dialogue_lines: List[Dict],
    scene_description: str = "",
) -> Optional[Dict]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key or not original_prompt or not rewritten_prompt:
        return None

    import json as _j
    system_msg = """You are a quality judge for rewritten video generation prompts.
Rate the rewrite on three dimensions (1-5), then list any issues.

## Criteria
1. dialogue_accuracy: Are all rewritten dialogue lines correctly placed?
2. style_preservation: Is the original visual style/camera/quality preserved?
3. scene_coherence: Is the scene context consistent with the original?

## Output
JSON only: {"dialogue_accuracy": N, "style_preservation": N, "scene_coherence": N, "issues": ["..."]}"""

    user_msg = f"""## Original Prompt
{original_prompt[:2000]}

## Rewritten Prompt
{rewritten_prompt[:2000]}

## Target Dialogue
{_j.dumps(dialogue_lines, ensure_ascii=False, indent=2)}

## Scene Context
{scene_description or '(none)'}"""

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        text = resp.choices[0].message.content or ""
        return _j.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
    except Exception:
        return None

