"""EditPlanner: LLM-driven structured edit planning for canvas node prompt rewriting.

Replaces the Python-based _classify_operation_type with LLM reasoning.
Takes an EvidencePack → LLM outputs structured EditPlan JSON containing:
  - operation_type    — literal_replace, semantic_insert, fuzzy_replace, etc.
  - match_evidence    — signal-level match reasoning
  - prompt_patch      — layered prompt editing guidance
  - final_prompt      — optional: if LLM generates final prompt text
  - risks             — known issues

Falls back to Python-based classification when LLM is unavailable.
"""
from __future__ import annotations

import json as _json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _fuzzy_word_match(original: str, prompt: str, threshold: float = 0.6) -> bool:
    words = [w.lower() for w in original.split() if len(w) >= 2]
    if not words:
        return False
    prompt_lower = prompt.lower()
    matched = sum(1 for w in words if re.search(r'\b' + re.escape(w) + r'\b', prompt_lower))
    return matched / len(words) >= threshold


_SYSTEM_PROMPT = """## Role
You are a structured edit planner for video generation prompts. Given an evidence
package (script lines, canvas node, scene context), output a structured edit plan.

## Task
1. Determine the operation_type:
   - "literal_replace": original dialogue text appears verbatim in the node prompt
   - "fuzzy_replace": similar but not exact dialogue appears (ASR drift, punctuation)
   - "semantic_insert": no dialogue text, but visual scene context matches
   - "section_reconstruct": the relevant prompt section is broken/incomplete
   - "style_preserving_fallback": node is usable but local section is unreliable
   - "full_fallback": no usable node evidence

2. Provide match_evidence: for each match signal found, list {signal, detail, confidence}

3. Provide prompt_patch_guidance:
   - global_style: extracted visual style prefix from original prompt
   - local_visual_context: the matching scene section description
   - discarded_sections: sections to remove (no rewritten dialogue)
   - dialogue_placement: where to insert/replace each rewritten line

4. List any risks.

## Output
JSON only, no markdown fences:
{"operation_type": "...", "match_evidence": [...], "prompt_patch_guidance": {...}, "risks": [...], "confidence": 0.0}"""


def plan_edit(evidence_pack: Any) -> Dict[str, Any]:
    """Send EvidencePack to LLM, return structured EditPlan dict.

    Returns dict with keys: operation_type, match_evidence, prompt_patch_guidance, risks, confidence.
    Falls back to Python-based classification on failure.
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return _fallback_plan(evidence_pack)

    node_evidence = getattr(evidence_pack, "canvas_node", None)
    if not node_evidence or not getattr(node_evidence, "full_prompt", ""):
        return {"operation_type": "full_fallback", "match_evidence": [], "prompt_patch_guidance": {}, "risks": [], "confidence": 0.0}

    # Build evidence summary for LLM
    target_summary = []
    for line in getattr(evidence_pack, "target_lines", []):
        target_summary.append({
            "line_id": getattr(line, "line_id", ""),
            "speaker": getattr(line, "speaker", ""),
            "original": getattr(line, "original", ""),
            "rewritten": getattr(line, "rewritten", ""),
            "time": f"{getattr(line, 'start_seconds', 0):.1f}-{getattr(line, 'end_seconds', 0):.1f}s",
            "shot_scene": getattr(line, "shot_scene", ""),
        })

    sections_summary = []
    for sec in getattr(node_evidence, "sections", []) or []:
        sections_summary.append({
            "section_id": getattr(sec, "section_id", ""),
            "description": getattr(sec, "description", ""),
            "has_quoted_dialogue": getattr(sec, "contains_quoted_dialogue", False),
            "quoted_text": getattr(sec, "quoted_dialogue", []),
        })

    matched_section = getattr(evidence_pack, "matched_section_id", "")

    user_msg = f"""## Node Prompt
{node_evidence.full_prompt[:3000]}

## Node Sections
{_json.dumps(sections_summary, ensure_ascii=False, indent=2) if sections_summary else '(no section analysis)'}

## Matched Section ID
{matched_section or '(none)'}

## Target Lines to Rewrite
{_json.dumps(target_summary, ensure_ascii=False, indent=2)}

## Node Reference Images
{len(getattr(node_evidence, 'reference_images', []))} available"""

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=4096,
        )
        text = resp.choices[0].message.content or ""
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
        return _json.loads(text)
    except Exception:
        return _fallback_plan(evidence_pack)


def _fallback_plan(evidence_pack: Any) -> Dict[str, Any]:
    """Python-based operation_type classification when LLM is unavailable."""
    node_evidence = getattr(evidence_pack, "canvas_node", None)
    prompt = getattr(node_evidence, "full_prompt", "") if node_evidence else ""

    if not prompt:
        return {"operation_type": "full_fallback", "match_evidence": [], "prompt_patch_guidance": {}, "risks": [], "confidence": 0.0}

    any_literal = False
    any_fuzzy = False

    for line in getattr(evidence_pack, "target_lines", []):
        original = getattr(line, "original", "").strip()
        rewritten = getattr(line, "rewritten", "").strip()
        if not original or not rewritten or original == rewritten:
            continue
        if original in prompt:
            any_literal = True
        elif _fuzzy_word_match(original, prompt):
            any_fuzzy = True

    if any_literal:
        op = "literal_replace"
    elif any_fuzzy:
        op = "fuzzy_replace"
    else:
        op = "semantic_insert"

    return {
        "operation_type": op,
        "match_evidence": [],
        "prompt_patch_guidance": {},
        "risks": ["python_fallback_classification"],
        "confidence": 0.5,
    }
