"""Segment Matcher: match EditAtoms to Canvas Node prompts via LLM."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from skills.timeline_plan.models import EditAtom, CanvasNode

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("LLM_PLANNER_MODEL", "deepseek-v4-pro")
_LOG_DIR = Path("runs/v4_plans/matcher_logs")
_log_counter = 0


def _get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def _log_llm(prompt: str, resp: str, dur: float):
    global _log_counter
    _log_counter += 1
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _LOG_DIR / f"{_log_counter:03d}_matcher_{time.strftime('%H%M%S')}.json"
    with open(fpath, "w") as f:
        json.dump({
            "prompt_chars": len(prompt), "response_chars": len(resp),
            "duration_sec": round(dur, 1),
            "prompt": prompt, "response": resp,
        }, f, ensure_ascii=False, indent=2)


def _build_matching_prompt(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
) -> str:
    atoms_json = json.dumps([
        {
            "atom_id": a.atom_id,
            "shot_numbers": a.shot_numbers,
            "primary_shot_number": a.primary_shot_number,
            "scene": a.scene_description[:400],
            "dialogue": [
                {"line_id": l.line_id, "speaker": l.speaker,
                 "original": l.original, "rewritten": l.rewritten}
                for l in a.rewritten_lines
            ],
        }
        for a in atoms
    ], ensure_ascii=False, indent=2)

    nodes_json = json.dumps([
        {"node_id": n.node_id, "prompt": n.prompt[:1200]}
        for n in canvas_nodes
    ], ensure_ascii=False, indent=2)

    return f"""## Role
Match each Edit Atom to the canvas node prompt that best fits its
dialogue + scene + character context.

Canvas node prompts describe original video generation intent.
Focus on matching spoken dialogue (exact or semantic) AND the
scene/environment/action described.

## Edit Atoms ({len(atoms)})
```json
{atoms_json}
```

## Canvas Nodes ({len(canvas_nodes)})
```json
{nodes_json}
```

## Rules
- Prefer dialogue match over scene keyword match.
- If no node reasonably matches an atom, put it in unmatched.
- Every atom must appear in either matches or unmatched.
- Semantic similarity is acceptable when exact text differs.

## Output
Return ONLY JSON:
```json
{{
  "matches": [
    {{"atom_id": "A1", "node_id": "n1", "confidence": 0.9, "reasoning": "..."}}
  ],
  "unmatched": [
    {{"atom_id": "A2", "reason": "no canvas prompt matches this scene"}}
  ]
}}
```"""


def _parse_match_response(text: str) -> tuple[list[dict], list[dict]]:
    if not text or not text.strip():
        return [], []
    t = text.strip()
    if t.startswith("```"):
        ls = t.split("\n")
        if ls[0].startswith("```"):
            ls = ls[1:]
        if ls and ls[-1].strip() in ("```", "```json"):
            ls = ls[:-1]
        t = "\n".join(ls).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        b = t.find("{")
        if b < 0:
            return [], []
        d = 0
        for i in range(b, len(t)):
            if t[i] in "[{":
                d += 1
            elif t[i] in "]}":
                d -= 1
            if d == 0:
                try:
                    data = json.loads(t[b:i + 1])
                    break
                except json.JSONDecodeError:
                    pass
        else:
            return [], []
    return data.get("matches", []), data.get("unmatched", [])


def match_atoms_to_nodes(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
) -> None:
    """Match each EditAtom to a CanvasNode via LLM. Updates atoms in-place."""
    if not atoms:
        return

    client = _get_client()
    if not client:
        logger.warning("No LLM client available")
        return

    prompt = _build_matching_prompt(atoms, canvas_nodes)
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32768,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "enabled"}},
        )
        text = resp.choices[0].message.content or ""
        _log_llm(prompt, text, time.time() - t0)
    except Exception as e:
        logger.error("Segment matcher LLM call failed: %s", e)
        return

    matches, unmatched = _parse_match_response(text)
    match_map = {m["atom_id"]: m for m in matches}
    unmatched_ids = {u["atom_id"] for u in unmatched}

    for atom in atoms:
        if atom.atom_id in match_map:
            m = match_map[atom.atom_id]
            atom.matched_node_id = m.get("node_id")
            atom.match_confidence = float(m.get("confidence", 0.0))
            atom.match_reasoning = m.get("reasoning", "")
        elif atom.atom_id not in unmatched_ids:
            logger.warning("Atom %s missing from LLM response", atom.atom_id)

    matched_count = sum(1 for a in atoms if a.matched_node_id)
    logger.info("Matcher: %d/%d atoms matched", matched_count, len(atoms))
