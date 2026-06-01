"""Segment Matcher: match EditAtoms to Canvas Node prompts via LLM."""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from skills.timeline_plan.models import EditAtom, CanvasNode, WindowPlanDraft

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
- Also group matched atoms into generation window drafts. A window draft
  should contain atoms that belong to one coherent prompt intent and one
  canvas node. Do not group different node intents together.
- Each window_draft must reference only atom_ids that are matched to the same
  node_id in matches. Never include unmatched atoms in window_drafts.

## Output
Return ONLY JSON:
```json
{{
  "matches": [
    {{"atom_id": "A1", "node_id": "n1", "confidence": 0.9, "reasoning": "..."}}
  ],
  "window_drafts": [
    {{"draft_id": "draft_001", "atom_ids": ["A1", "A2"], "node_id": "n1", "confidence": 0.9, "reasoning": "..."}}
  ],
  "unmatched": [
    {{"atom_id": "A2", "reason": "no canvas prompt matches this scene"}}
  ]
}}
```"""


def _parse_match_response(text: str) -> tuple[list[dict], list[dict], list[dict]]:
    if not text or not text.strip():
        return [], [], []
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
            return [], [], []
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
            return [], [], []
    return data.get("matches", []), data.get("unmatched", []), data.get("window_drafts", [])


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fallback_window_drafts(atoms: list[EditAtom]) -> list[WindowPlanDraft]:
    # Unmatched atoms are intentionally excluded; resolver emits degraded windows for them.
    drafts: list[WindowPlanDraft] = []
    for idx, atom in enumerate([a for a in atoms if a.matched_node_id], start=1):
        drafts.append(WindowPlanDraft(
            draft_id=f"draft_{idx:03d}",
            atom_ids=[atom.atom_id],
            node_id=atom.matched_node_id,
            confidence=atom.match_confidence,
            reasoning="fallback_single_atom_draft",
        ))
    return drafts


def _coerce_window_drafts(raw_drafts: list[dict], atoms: list[EditAtom]) -> list[WindowPlanDraft]:
    atom_map = {a.atom_id: a for a in atoms}
    drafts: list[WindowPlanDraft] = []
    used: set[str] = set()
    for idx, raw in enumerate(raw_drafts, start=1):
        node_id = str(raw["node_id"]) if raw.get("node_id") else None
        atom_ids = []
        for aid in raw.get("atom_ids", []):
            atom_id = str(aid)
            atom = atom_map.get(atom_id)
            if not atom or atom_id in used:
                continue
            if node_id and atom.matched_node_id != node_id:
                continue
            atom_ids.append(atom_id)
        if not atom_ids:
            continue
        drafts.append(WindowPlanDraft(
            draft_id=str(raw.get("draft_id") or f"draft_{idx:03d}"),
            atom_ids=atom_ids,
            node_id=node_id,
            confidence=_safe_float(raw.get("confidence")),
            reasoning=str(raw.get("reasoning", "")),
            fallback_reason=str(raw.get("fallback_reason", "")),
        ))
        used.update(atom_ids)

    for atom in atoms:
        if atom.atom_id in used or not atom.matched_node_id:
            continue
        drafts.append(WindowPlanDraft(
            draft_id=f"draft_{len(drafts) + 1:03d}",
            atom_ids=[atom.atom_id],
            node_id=atom.matched_node_id,
            confidence=atom.match_confidence,
            reasoning="fallback_for_unplanned_matched_atom",
        ))
    return drafts


def match_atoms_to_nodes(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
) -> list[WindowPlanDraft]:
    """Match EditAtoms to CanvasNodes and return holistic window drafts."""
    if not atoms:
        return []

    client = _get_client()
    if not client:
        logger.warning("No LLM client available")
        return []

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
        return []

    matches, unmatched, raw_drafts = _parse_match_response(text)
    match_map = {str(m["atom_id"]): m for m in matches if m.get("atom_id")}
    unmatched_ids = {str(u["atom_id"]) for u in unmatched if u.get("atom_id")}

    for atom in atoms:
        if atom.atom_id in match_map:
            m = match_map[atom.atom_id]
            confidence = _safe_float(m.get("confidence"))
            atom.matched_node_id = m.get("node_id")
            atom.match_confidence = 0.0 if confidence is None else confidence
            atom.match_reasoning = m.get("reasoning", "")
        elif atom.atom_id not in unmatched_ids:
            logger.warning("Atom %s missing from LLM response", atom.atom_id)

    matched_count = sum(1 for a in atoms if a.matched_node_id)
    drafts = _coerce_window_drafts(raw_drafts, atoms)
    if not drafts:
        drafts = _fallback_window_drafts(atoms)
    logger.info("Matcher: %d/%d atoms matched, %d window drafts", matched_count, len(atoms), len(drafts))
    return drafts
