"""PromptPatchComposer: layered prompt editing with operation_type support.

Replaces prompt_extractor.py. Supports multiple operation types:
  - literal_replace: original dialogue exists → replace
  - semantic_insert: no original text, but visual scene matches → insert
  - style_preserving_fallback: keep style layer, generate scene + dialogue
  - full_fallback: no usable node evidence

Key improvement over v2: fallback preserves style layer from original prompt,
and semantic_insert does not require finding original dialogue text.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List


# ── Module-level env loading (once, not per-call) ──────────────────

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


# ── Style layer keywords for extraction ────────────────────────────

_STYLE_KEYWORDS_CN = [
    "美式情景喜剧", "真实短剧", "柔光雾化", "画面通透",
    "电影级布光", "超高清", "电影质感", "浅景深",
    "8k", "4k", "hdr",
]

_STYLE_KEYWORDS_EN = [
    "cinematic", "8k", "4k", "hdr", "shallow depth of field",
    "film grain", "soft lighting", "ultra hd",
]


def _extract_style_prefix(prompt: str, max_chars: int = 300) -> str:
    """Extract global visual style prefix from a canvas node prompt.

    Scans the prompt for style keywords and returns the prefix containing them,
    stopping at the first scene/dialogue indicator.
    """
    if not prompt or not prompt.strip():
        return ""

    cutoff = len(prompt)
    for marker in ["镜头", "场景", "Scene", "scene", '"', "'"]:
        idx = prompt.find(marker)
        if idx != -1 and idx < cutoff:
            cutoff = idx

    prefix = prompt[:min(cutoff, max_chars)].strip()

    has_style = False
    for kw in _STYLE_KEYWORDS_CN + _STYLE_KEYWORDS_EN:
        if kw.lower() in prefix.lower():
            has_style = True
            break

    return prefix if has_style else ""


# ── LLM-based prompt rewriting ─────────────────────────────────────

def _build_system_prompt(operation_type: str) -> str:
    if operation_type == "semantic_insert":
        return """## Role
You rewrite video generation prompts for seedance. The original prompt does NOT contain
the original dialogue as quoted text. Instead, visual scene descriptions match the
dialogue context.

## Instructions
1. PRESERVE the global visual style settings (resolution, lighting, camera style) verbatim.
2. KEEP the visual scene description that matches the rewritten dialogue context.
3. INSERT each rewritten dialogue line at the appropriate position in the scene.
4. REMOVE scene sections with no rewritten dialogue.
5. Do NOT attempt to find and replace non-existent original text — INSERT instead.

## Output
Rewritten prompt text only. No explanations, no JSON."""

    return """## Role
You rewrite video generation prompts for seedance, keeping only visual content tied to rewritten dialogue.

The original prompt mixes style settings (resolution, lighting, camera style), scene descriptions with camera angles and character actions, and dialogue in quotes. Style settings apply to the entire video and must be preserved. Scene descriptions should be kept only if they contain dialogue being rewritten — within them, keep only the visuals directly around the dialogue moment and cut background filler. Remove entire scenes with no rewritten dialogue.

Replace each original dialogue line with its rewritten version, preserving the speaker attribution format. If no scene matches the rewrite lines, output the original prompt unchanged.

## Output
Rewritten prompt text only. No explanations, no JSON."""


def _build_user_prompt(
    full_prompt: str,
    mappings: List[Dict],
    scene_description: str,
    operation_type: str,
) -> str:
    import json as _j

    base = f"""## Original Prompt
{full_prompt}

## Dialogue to Rewrite
{_j.dumps(mappings, ensure_ascii=False, indent=2)}

## Scene Context
{scene_description or '(none)'}"""

    if operation_type == "semantic_insert":
        base += """

## Operation: semantic_insert
The original dialogue is NOT present as quoted text in the original prompt.
Instead, a visual scene description matches the dialogue context.
INSERT each rewritten dialogue line at the appropriate visual position."""
    else:
        base += """

Output the rewritten prompt."""

    return base


def _llm_rewrite_prompt(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
    operation_type: str = "literal_replace",
) -> str:
    mappings = []
    for line in rewrite_lines:
        original = getattr(line, "original", "") or getattr(line, "dialogue", "")
        rewritten = getattr(line, "rewritten", "")
        speaker = getattr(line, "speaker", "")
        if rewritten.strip() and original.strip() != rewritten.strip():
            mappings.append({
                "speaker": speaker,
                "original": original.strip(),
                "rewritten": rewritten.strip(),
            })

    if not mappings:
        return ""

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": _build_system_prompt(operation_type)},
                {"role": "user", "content": _build_user_prompt(full_prompt, mappings, scene_description, operation_type)},
            ],
            temperature=0.0,
            max_tokens=8192,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception:
        return ""


# ── Fallback: generate prompt from scene description ──────────────

def _generate_prompt_from_scene(
    rewrite_lines: List[Any],
    scene_description: str = "",
    style_layer: str = "",
) -> str:
    dialogue_lines = []
    for line in rewrite_lines:
        speaker = getattr(line, "speaker", "Character")
        rewritten = getattr(line, "rewritten", "") or getattr(line, "dialogue", "")
        if rewritten:
            dialogue_lines.append(f'{speaker} says: "{rewritten}"')

    parts = []
    if style_layer:
        parts.append(style_layer)

    desc = scene_description or "A cinematic scene"
    parts.append(desc)

    if dialogue_lines:
        parts.append("\n".join(dialogue_lines))

    return "\n".join(parts).strip()


# ── Validation ─────────────────────────────────────────────────────

def _validate_rewrite(
    prompt: str,
    rewrite_lines: List[Any],
    operation_type: str = "literal_replace",
) -> bool:
    for line in rewrite_lines:
        rewritten = getattr(line, "rewritten", "") or ""
        if rewritten.strip() and rewritten.strip() not in prompt:
            return False
    return True


# ── Main orchestrator ──────────────────────────────────────────────

def compose_prompt_patch(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
    operation_type: str = "literal_replace",
) -> str:
    all_same = all(
        getattr(line, "original", "") == getattr(line, "rewritten", "")
        for line in rewrite_lines
    )
    if all_same and full_prompt:
        return full_prompt

    if not full_prompt:
        return _generate_prompt_from_scene(rewrite_lines, scene_description)

    result = _llm_rewrite_prompt(full_prompt, rewrite_lines, scene_description, operation_type)
    if result and _validate_rewrite(result, rewrite_lines, operation_type):
        return result

    style_layer = _extract_style_prefix(full_prompt)
    if style_layer:
        return _generate_prompt_from_scene(rewrite_lines, scene_description, style_layer)

    return _generate_prompt_from_scene(rewrite_lines, scene_description)
