"""Segment Matcher: match EditAtoms to Canvas Node prompts via LLM."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from skills.timeline_plan.models import EditAtom, CanvasNode, WindowPlanDraft
from skills.timeline_plan._llm_utils import get_llm_client, _DEFAULT_MODEL

logger = logging.getLogger(__name__)

_LOG_DIR = Path("runs/v4_plans/matcher_logs")
_log_counter = 0
_log_lock = threading.Lock()


def _log_llm(prompt: str, resp: str, dur: float):
    global _log_counter
    with _log_lock:
        _log_counter += 1
        counter = _log_counter
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _LOG_DIR / f"{counter:03d}_matcher_{time.strftime('%H%M%S')}.json"
    with open(fpath, "w") as f:
        json.dump({
            "prompt_chars": len(prompt), "response_chars": len(resp),
            "duration_sec": round(dur, 1),
            "prompt": prompt, "response": resp,
        }, f, ensure_ascii=False, indent=2)


def _coarse_recall(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
    min_candidates: int = 3,
) -> dict[str, list[str]]:
    """Deterministic coarse recall: per-atom candidate nodes by keyword overlap.
    
    Uses dialogue text and speaker name to find candidate nodes.
    Always returns at least min_candidates nodes per atom (if available).
    Falls back to all nodes when no candidates found.
    """
    candidates: dict[str, list[str]] = {}
    
    for atom in atoms:
        atom_words = set()
        for line in atom.rewritten_lines:
            for w in line.original.lower().split():
                w = w.strip(".!?,;:\"'")
                if len(w) > 1:
                    atom_words.add(w)
            atom_words.add(line.speaker.lower())
        
        if not atom_words:
            candidates[atom.atom_id] = [n.node_id for n in canvas_nodes[:10]]
            continue
        
        scored = []
        for node in canvas_nodes:
            prompt_lower = node.prompt.lower()
            score = sum(1 for w in atom_words if w in prompt_lower)
            if score > 0:
                scored.append((node.node_id, score))
        
        scored.sort(key=lambda x: -x[1])
        top = [nid for nid, _ in scored[:max(min_candidates, 3)]]
        
        if len(top) < min_candidates:
            # Pad with all nodes to ensure minimum recall
            extra = [n.node_id for n in canvas_nodes if n.node_id not in top]
            top.extend(extra[:min_candidates - len(top)])
        
        candidates[atom.atom_id] = top if top else [n.node_id for n in canvas_nodes[:10]]
    
    return candidates


def _filter_canvas_nodes(
    candidates: dict[str, list[str]],
    canvas_nodes: list[CanvasNode],
) -> list[CanvasNode]:
    """Filter canvas nodes to only those that appear in candidate lists."""
    node_map = {n.node_id: n for n in canvas_nodes}
    used_ids: set[str] = set()
    for nids in candidates.values():
        used_ids.update(nids)
    return [node_map[nid] for nid in used_ids if nid in node_map]


def _build_matching_prompt(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
    atom_candidates: dict[str, list[str]] | None = None,
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
            "candidates": atom_candidates.get(a.atom_id, [])[:8] if atom_candidates else [],
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
- Each atom has a "candidates" list — hints from keyword overlap, not constraints.
- Prefer dialogue match over scene keyword match.
- If no node reasonably matches an atom, put it in unmatched.
- Every atom must appear in either matches or unmatched.
- Semantic similarity is acceptable when exact text differs.
- Atoms from the same shot or consecutive shots with shared characters
  often belong to the same canvas node. If dialogue + scene context
  supports it, match them to the same node.
- Group matched atoms into window drafts. A window draft contains atoms
  that share one canvas node and one coherent prompt intent.
- Each window_draft must reference only atom_ids matched to the same node_id.
  Never include unmatched atoms in window_drafts.

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
                logger.debug("Draft %s: atom %s matched to %s, but draft claims %s — skipping",
                            raw.get("draft_id", "?"), atom_id, atom.matched_node_id, node_id)
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

    # Coarse recall: deterministic keyword matching → candidate nodes
    atom_candidates = _coarse_recall(atoms, canvas_nodes)
    candidate_nodes = _filter_canvas_nodes(atom_candidates, canvas_nodes)

    client = get_llm_client()
    if not client:
        logger.warning("No LLM client available")
        return []

    prompt = _build_matching_prompt(atoms, candidate_nodes, atom_candidates)
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

    # Post-processing: propagate matches to adjacent unmatched atoms
    matched_count = sum(1 for a in atoms if a.matched_node_id)
    drafts = _coerce_window_drafts(raw_drafts, atoms)
    if not drafts:
        drafts = _fallback_window_drafts(atoms)
    logger.info("Matcher: %d/%d atoms matched, %d window drafts", matched_count, len(atoms), len(drafts))
    return drafts
