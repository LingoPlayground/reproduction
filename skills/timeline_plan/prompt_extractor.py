"""Prompt fragment extraction and dialogue replacement.

Extracts only the prompt portion relevant to the specific lines being rewritten.
Handles multi-shot canvas nodes — a node's prompt may cover 5+ scenes, but we
only want the visual description for the lines we're actually rewriting.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional


def _normalize(text: str) -> str:
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ── Section boundary detection ──────────────────────────────────────

SECTION_HEADER_PATTERNS = [
    re.compile(r"镜头\s*\d+", re.IGNORECASE),
    re.compile(r"shot\s*\d+", re.IGNORECASE),
    re.compile(r"(?:场景|片段)\s*\d+", re.IGNORECASE),
    re.compile(r"^#{1,3}\s+", re.MULTILINE),  # Markdown headings
]


def _find_section_boundaries(prompt: str) -> List[int]:
    """Find line indices where new sections start in a prompt.

    Returns list of line numbers (0-indexed) where a new section header appears.
    """
    lines = prompt.split("\n")
    boundaries = [0]  # First section always starts at line 0
    for i, line in enumerate(lines):
        if i == 0:
            continue
        for pat in SECTION_HEADER_PATTERNS:
            if pat.search(line):
                boundaries.append(i)
                break
    return boundaries


def _find_lines_containing_dialogue(prompt: str, dialogue_fragments: List[str]) -> List[int]:
    """Find line indices in the prompt that contain any of the dialogue fragments.

    Uses normalized text comparison to handle ASR noise and punctuation differences.
    """
    lines = prompt.split("\n")
    norm_lines = [_normalize(l) for l in lines]

    hit_indices: List[int] = []
    for i, nl in enumerate(norm_lines):
        for frag in dialogue_fragments:
            frag_norm = _normalize(frag)
            if frag_norm and frag_norm in nl:
                hit_indices.append(i)
                break
    return hit_indices


# ── Content-driven extraction ───────────────────────────────────────

def extract_prompt_fragment_for_lines(
    full_prompt: str,
    target_lines: List[Any],
    context_lines: int = 5,
) -> Optional[str]:
    """Extract the minimal prompt section covering the target lines.

    Strategy:
    1. Find which prompt lines contain the target dialogue fragments
    2. Find the section boundaries that contain those lines
    3. Return the union of those sections (the minimum spanning section set)
    4. If section boundaries don't exist, use context window around dialogue hits

    This ensures we extract the visual description for ONLY the relevant
    lines, not the entire multi-shot node prompt.

    Args:
        full_prompt: Complete canvas node prompt.
        target_lines: Lines being rewritten (need .original or .dialogue attrs).
        context_lines: Extra context lines to include before/after dialogue hits
                      when section boundaries aren't available.

    Returns:
        Extracted prompt fragment, or None if no dialogue found in prompt.
    """
    # Collect dialogue fragments from target lines
    fragments: List[str] = []
    for line in target_lines:
        original = getattr(line, "original", "") or getattr(line, "dialogue", "")
        if original.strip():
            fragments.append(original.strip())

    if not fragments:
        return None

    # Find which lines contain the dialogue
    hit_lines = _find_lines_containing_dialogue(full_prompt, fragments)
    if not hit_lines:
        return None

    # Find section boundaries
    boundaries = _find_section_boundaries(full_prompt)
    prompt_lines = full_prompt.split("\n")

    if len(boundaries) > 1:
        # Determine which sections contain our dialogue hits
        hit_sections: set[int] = set()
        for hl in hit_lines:
            for s_idx in range(len(boundaries)):
                section_start = boundaries[s_idx]
                section_end = boundaries[s_idx + 1] if s_idx + 1 < len(boundaries) else len(prompt_lines)
                if section_start <= hl < section_end:
                    hit_sections.add(s_idx)
                    break

        if hit_sections:
            # Extract the minimum spanning section range
            min_section = min(hit_sections)
            max_section = max(hit_sections)
            start_line = boundaries[min_section]
            end_line = boundaries[max_section + 1] if max_section + 1 < len(boundaries) else len(prompt_lines)
            return "\n".join(prompt_lines[start_line:end_line]).strip()

    # Fallback: context window around dialogue hits
    min_line = max(0, min(hit_lines) - context_lines)
    max_line = min(len(prompt_lines), max(hit_lines) + context_lines + 1)
    return "\n".join(prompt_lines[min_line:max_line]).strip()


# ── Dialogue replacement ────────────────────────────────────────────

def replace_dialogue_in_fragment(fragment: str, rewrite_lines: List[Any]) -> str:
    """Replace original dialogue with rewritten text in a prompt fragment.

    Strategy: quoted text → colon context → exact substring.
    Returns the modified fragment.
    """
    result = fragment
    replaced_count = 0

    for line in rewrite_lines:
        original = getattr(line, "original", "") or getattr(line, "dialogue", "")
        rewritten = getattr(line, "rewritten", "")
        if not original or not rewritten or original.strip() == rewritten.strip():
            continue
        oc = original.strip()
        rc = rewritten.strip()
        on = _normalize(oc.lower())

        replaced = False
        # L1: Inside English or Chinese quotes
        for pat in [r'"([^"]{3,})"', r'\u201c([^\u201d]{3,})\u201d', r'\u300c([^\u300d]{3,})\u300d']:
            for m in re.finditer(pat, result):
                if on in _normalize(m.group(1).lower()):
                    result = result[:m.start(1)] + rc + result[m.end(1):]
                    replaced = True
                    break
            if replaced:
                break

        if not replaced:
            # L2: After colon (： or :)
            for m in re.finditer(r'[：:]\s*(.{3,}?)(?:[.！。\n]|$)', result):
                ac = m.group(1).strip()
                if on in _normalize(ac.lower()):
                    s = result.lower().find(ac.lower(), m.start(1))
                    s = m.start(1) if s < 0 else s
                    result = result[:s] + rc + result[s + len(ac):]
                    replaced = True
                    break

        if not replaced:
            # L3: Exact case-insensitive substring
            idx = result.lower().find(oc.lower())
            if idx >= 0:
                result = result[:idx] + rc + result[idx + len(oc):]
                replaced = True

        if replaced:
            replaced_count += 1

    return result


# ── Full fallback ───────────────────────────────────────────────────

def _generate_prompt_from_scene(rewrite_lines: List[Any], scene_description: str = "") -> str:
    """L4: Generate prompt from rewritten dialogue + optional scene context.

    Used when no prompt fragment can be extracted from the canvas node.
    """
    desc = scene_description or "A cinematic scene"
    dialogues = []
    for line in rewrite_lines:
        speaker = getattr(line, "speaker", "Character")
        rewritten = getattr(line, "rewritten", "") or getattr(line, "dialogue", "")
        if rewritten:
            dialogues.append(f'{speaker} says: "{rewritten}"')
    dialogue_block = "\n".join(dialogues) if dialogues else ""
    return f"{desc}\n{dialogue_block}".strip()


# ── Main orchestrator ───────────────────────────────────────────────

def extract_and_rewrite_prompt(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
) -> str:
    """Extract prompt fragment for rewritten lines and replace dialogue.

    Tries: content-driven extraction → full fallback.
    Always returns a prompt string (never None).

    Args:
        full_prompt: Complete canvas node prompt (empty string if no match).
        rewrite_lines: Lines being rewritten (.original, .rewritten, .speaker).
        scene_description: Scene context for fallback prompt generation.

    Returns:
        Rewritten prompt fragment ready for seedance.
    """
    if full_prompt:
        fragment = extract_prompt_fragment_for_lines(full_prompt, rewrite_lines)
        if fragment:
            return replace_dialogue_in_fragment(fragment, rewrite_lines)

    return _generate_prompt_from_scene(rewrite_lines, scene_description)
