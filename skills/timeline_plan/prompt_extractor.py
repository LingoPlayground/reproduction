"""Prompt rewriting via LLM: preserves style/quality settings, extracts relevant scenes, replaces dialogue.

Canvas node prompts are complex — they mix visual quality settings, scene descriptions,
camera directions, and dialogue across multiple languages. LLM is the only reliable way
to parse this structure and produce clean rewritten prompts for seedance.
"""
from __future__ import annotations

from typing import Any, List


# ── LLM-based prompt rewriting ─────────────────────────────────────

def _llm_rewrite_prompt(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
) -> str:
    """Use LLM to extract style settings, keep relevant scenes, replace dialogue.
    
    The LLM understands:
    - Which text is visual style/quality (preserve ALL)
    - Which sections describe scenes containing the target dialogue
    - Where to replace original dialogue with rewritten text
    - Which sections to remove (scenes for non-rewritten lines)
    
    Returns rewritten prompt, or empty string on failure.
    """
    mappings = []
    for line in rewrite_lines:
        original = getattr(line, "original", "") or getattr(line, "dialogue", "")
        rewritten = getattr(line, "rewritten", "")
        speaker = getattr(line, "speaker", "")
        if original.strip() and rewritten.strip() and original.strip() != rewritten.strip():
            mappings.append({
                "speaker": speaker,
                "original": original.strip(),
                "rewritten": rewritten.strip(),
            })
    
    if not mappings:
        return ""
    
    system_msg = """## Role
You rewrite video generation prompts for seedance (AI video generator).

## The Original Prompt
A canvas node prompt is a video generation instruction containing:
- **Style/Quality settings**: visual quality descriptors, resolution, lighting, color grading, camera style. These apply to the ENTIRE video and MUST be preserved in full.
- **Scene descriptions**: numbered sections (镜头 1, Shot 2, etc.) describing specific shots with camera angles, character actions, expressions, and dialogue.
- **Dialogue**: English text in quotation marks that characters speak on screen, often preceded by speaker names like "Donny: " or "台词配合：".

Style settings can appear ANYWHERE — at the top, bottom, or interspersed between scene descriptions.

## Your Task

1. **Preserve ALL style/quality settings** from the original prompt — every quality keyword, resolution spec, lighting description, camera style, and visual directive. These are non-negotiable.

2. **Keep entire scene sections that contain ANY rewritten dialogue.** Within those sections, replace ONLY the specific dialogue lines listed in the mappings below. Other dialogue in the same section should be left unchanged. Remove only scene sections that have NO rewritten dialogue at all.

3. **Replace original dialogue with rewritten dialogue** in the kept scene descriptions. The rewritten text may be longer or shorter — adjust naturally. Preserve the speaker attribution format (e.g., "Donny: " before the dialogue).

4. **Maintain the original structure and formatting** as much as possible.

5. If none of the scene descriptions match the rewrite lines, output the original prompt unchanged.

Output ONLY the rewritten prompt text — no explanations, no JSON wrappers."""

    import json as _j
    user_msg = f"""## Original Prompt
{full_prompt}

## Dialogue to Rewrite
{_j.dumps(mappings, ensure_ascii=False, indent=2)}

## Scene Context
{scene_description or '(none)'}

Output the rewritten prompt."""

    # Load API key
    import os as _os
    from pathlib import Path as _Path
    for env_path in [
        str(_Path("~/workspace/lingolens/backend/.env").expanduser()),
        str(_Path("~/workspace/shakespeare/.env").expanduser()),
    ]:
        if _Path(env_path).exists():
            for line in open(env_path):
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    _os.environ.setdefault(k.strip(), v.strip())

    api_key = _os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=_os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        resp = client.chat.completions.create(
            model=_os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
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
) -> str:
    """Generate a prompt from scene_description + rewritten dialogue.
    
    Used when no canvas node matched or LLM is unavailable.
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


# ── Main orchestrator ──────────────────────────────────────────────

def extract_and_rewrite_prompt(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
) -> str:
    """Rewrite a canvas node prompt for seedance generation.
    
    Uses LLM to preserve style settings, keep only relevant scenes,
    and replace dialogue. Falls back to scene_description-based generation.
    
    Args:
        full_prompt: Complete canvas node prompt (empty if no match).
        rewrite_lines: Lines with .original, .rewritten, .speaker attributes.
        scene_description: Fallback scene context.
    
    Returns:
        Rewritten prompt ready for seedance.
    """
    if not full_prompt:
        return _generate_prompt_from_scene(rewrite_lines, scene_description)
    
    # Try LLM-based rewriting
    result = _llm_rewrite_prompt(full_prompt, rewrite_lines, scene_description)
    if result:
        return result
    
    # LLM unavailable or failed — fallback
    return _generate_prompt_from_scene(rewrite_lines, scene_description)
