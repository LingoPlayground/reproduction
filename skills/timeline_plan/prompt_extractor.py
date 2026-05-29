"""Prompt rewriting via LLM: preserves style/quality settings, extracts relevant scenes, replaces dialogue.

Canvas node prompts are complex — they mix visual quality settings, scene descriptions,
camera directions, and dialogue across multiple languages. LLM is the only reliable way
to parse this structure and produce clean rewritten prompts for seedance.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List


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
You rewrite video generation prompts for seedance, keeping only visual content tied to rewritten dialogue.

The original prompt mixes style settings (resolution, lighting, camera style), scene descriptions with camera angles and character actions, and dialogue in quotes. Style settings apply to the entire video and must be preserved. Scene descriptions should be kept only if they contain dialogue being rewritten — within them, keep only the visuals directly around the dialogue moment and cut background filler. Remove entire scenes with no rewritten dialogue.

Replace each original dialogue line with its rewritten version, preserving the speaker attribution format. If no scene matches the rewrite lines, output the original prompt unchanged.

## Output
Rewritten prompt text only. No explanations, no JSON."""

    import json as _j
    user_msg = f"""## Original Prompt
{full_prompt}

## Dialogue to Rewrite
{_j.dumps(mappings, ensure_ascii=False, indent=2)}

## Scene Context
{scene_description or '(none)'}

Output the rewritten prompt."""

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
