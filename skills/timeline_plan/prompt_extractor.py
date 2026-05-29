"""Prompt fragment extraction and dialogue replacement.

4-level degradation:
  L1: Structured extraction by section headers ("镜头 N", "Shot N")
  L3: Dialogue keyword proximity search
  L4: Full fallback from scene_description

(L2 LLM-based segmentation reserved for future.)
"""
from __future__ import annotations

import re
from typing import Any, List, Optional


def _normalize(text: str) -> str:
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


SECTION_PATTERNS = [
    re.compile(r"镜头\s*{n}\b", re.IGNORECASE),
    re.compile(r"shot\s*{n}\b", re.IGNORECASE),
    re.compile(r"(?:场景|片段)\s*{n}\b", re.IGNORECASE),
]


def extract_by_section_headers(full_prompt: str, shot_number: int) -> Optional[str]:
    """L1: Extract prompt section by structured section headers."""
    lines = full_prompt.split("\n")
    sections: List[tuple[int, int]] = []  # (shot_num, line_index)
    for i, line in enumerate(lines):
        for pat in SECTION_PATTERNS:
            pattern_str = pat.pattern.format(n=r"(\d+)")
            m = re.search(pattern_str, line, re.IGNORECASE)
            if m:
                sections.append((int(m.group(1)) if m.lastindex else i, i))
                break
    if not sections:
        return None
    target_idx = None
    for idx, (s_num, _) in enumerate(sections):
        if s_num == shot_number:
            target_idx = idx
            break
    if target_idx is None:
        return None
    start_line = sections[target_idx][1]
    end_line = sections[target_idx + 1][1] if target_idx + 1 < len(sections) else len(lines)
    return "\n".join(lines[start_line:end_line]).strip()


def extract_by_dialogue_keywords(full_prompt: str, shot: Any, context_window: int = 3) -> Optional[str]:
    """L3: Extract around dialogue keyword matches."""
    keywords: List[str] = []
    for line in (shot.lines or []):
        dialogue = getattr(line, "dialogue", "")
        words = _normalize(dialogue).split()
        keywords.extend([w for w in words if len(w) >= 3])
    if not keywords:
        return None
    lines = full_prompt.split("\n")
    p_norm_lines = [_normalize(l) for l in lines]
    hit_indices: set[int] = set()
    for i, norm_line in enumerate(p_norm_lines):
        for kw in keywords:
            if kw in norm_line:
                hit_indices.add(i)
                break
    if not hit_indices:
        return None
    min_idx = max(0, min(hit_indices) - context_window)
    max_idx = min(len(lines), max(hit_indices) + context_window + 1)
    return "\n".join(lines[min_idx:max_idx]).strip()


def replace_dialogue_in_fragment(fragment: str, rewrite_lines: List[Any]) -> str:
    """Replace original dialogue with rewritten text in a prompt fragment.

    Strategy: quoted text → colon context → exact substring.
    """
    result = fragment
    for line in rewrite_lines:
        original = getattr(line, "original", "") or getattr(line, "dialogue", "")
        rewritten = getattr(line, "rewritten", "")
        if not original or not rewritten or original.strip() == rewritten.strip():
            continue
        oc = original.strip()
        rc = rewritten.strip()
        on = _normalize(oc.lower())

        # L1: Inside quotes
        for pat in [r'"([^"]{3,})"', r'\u201c([^\u201d]{3,})\u201d', r'\u300c([^\u300d]{3,})\u300d']:
            for m in re.finditer(pat, result):
                if on in _normalize(m.group(1).lower()):
                    result = result[:m.start(1)] + rc + result[m.end(1):]
                    break
            else:
                continue
            break
        else:
            # L2: After colon
            for m in re.finditer(r'[：:]\s*(.{3,}?)(?:[.！。\n]|$)', result):
                ac = m.group(1).strip()
                if on in _normalize(ac.lower()):
                    s = result.lower().find(ac.lower(), m.start(1))
                    s = m.start(1) if s < 0 else s
                    result = result[:s] + rc + result[s + len(ac):]
                    break
            else:
                # L3: Exact substring
                idx = result.lower().find(oc.lower())
                if idx >= 0:
                    result = result[:idx] + rc + result[idx + len(oc):]
    return result


def _generate_prompt_from_scene(shot: Any, rewrite_lines: List[Any]) -> str:
    """L4: Generate prompt from scene_description + rewritten dialogue."""
    desc = getattr(shot, "scene_description", "") or "A cinematic scene"
    dialogues = []
    for line in rewrite_lines:
        speaker = getattr(line, "speaker", "Character")
        rewritten = getattr(line, "rewritten", "") or getattr(line, "dialogue", "")
        if rewritten:
            dialogues.append(f'{speaker} says: "{rewritten}"')
    dialogue_block = "\n".join(dialogues) if dialogues else ""
    return f"{desc}\n{dialogue_block}".strip()


def extract_and_rewrite_prompt(
    full_prompt: str,
    target_shot: Any,
    rewrite_lines: List[Any],
    node_cut_points: Optional[list] = None,
) -> str:
    """Main orchestrator: L1 → L3 → L4 fallback."""
    shot_number = getattr(target_shot, "shot_number", 0)

    # Level 1
    fragment = extract_by_section_headers(full_prompt, shot_number)
    if fragment:
        return replace_dialogue_in_fragment(fragment, rewrite_lines)

    # Level 3
    fragment = extract_by_dialogue_keywords(full_prompt, target_shot)
    if fragment:
        return replace_dialogue_in_fragment(fragment, rewrite_lines)

    # Level 4
    return _generate_prompt_from_scene(target_shot, rewrite_lines)
