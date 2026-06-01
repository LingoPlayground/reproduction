"""LLM Planner: two-stage pipeline.

Stage A1 (Coarse): LLM lists ALL candidate nodes per line (high recall).
Stage A2 (Fine):   LLM picks best match from candidates only (small context).
Code: deterministic grouping by node_id + time proximity.
Stage B (Rewrite): Per-group LLM rewrites the prompt.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from skills.timeline_plan.planner_models import (
    MatchResult, NodeGeneration, LineNodeMatch, SourceTimeRange,
    UnmatchedLine, RewriteInput,
)

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("LLM_PLANNER_MODEL", "deepseek-v4-pro")

_LOG_DIR = Path("runs/v3_plans/llm_logs")
_log_counter = 0


def _log_llm(stage: str, gid: str, prompt: str, resp: str, dur: float):
    global _log_counter; _log_counter += 1
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    f = _LOG_DIR / f"{_log_counter:03d}_{stage}_{gid}_{time.strftime('%H%M%S')}.json"
    with open(f, "w") as fh:
        json.dump({"stage": stage, "group_id": gid, "duration_sec": round(dur, 1),
                    "prompt_chars": len(prompt), "response_chars": len(resp),
                    "prompt": prompt, "response": resp}, fh, ensure_ascii=False, indent=2)


def _get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key: return None
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))


def _parse_json(text: str) -> Optional[Dict]:
    if not text or not text.strip(): return None
    t = text.strip()
    if t.startswith("```"):
        ls = t.split("\n")
        if ls[0].startswith("```"): ls = ls[1:]
        if ls[-1].strip() in ("```", "```json"): ls = ls[:-1]
        t = "\n".join(ls).strip()
    try: return json.loads(t)
    except json.JSONDecodeError:
        b = t.find("{"); d = 0
        if b < 0: return None
        for i in range(b, len(t)):
            if t[i] in "[{": d += 1
            elif t[i] in "]}": d -= 1
            if d == 0:
                try: return json.loads(t[b:i+1])
                except: pass
        return None


# ═══════════════════════════════════════════════════════════════════
# Stage A1: Coarse filter — list ALL candidate nodes per line
# ═══════════════════════════════════════════════════════════════════

def _make_coarse_prompt(evidence: Dict) -> str:
    lines = evidence.get("rewrite_lines", [])
    nodes = evidence.get("canvas_nodes", [])

    lines_json = json.dumps([
        {"line_id": l["line_id"], "speaker": l["speaker"], "dialogue": l["original"]}
        for l in lines
    ], ensure_ascii=False, indent=2)

    nodes_json = json.dumps([
        {"node_id": n["node_id"], "prompt": n["prompt"]}
        for n in nodes
    ], ensure_ascii=False, indent=2)

    return f"""## Role
For each line, list ALL canvas nodes whose prompt contains the dialogue
(exact or semantic match). Dialogue may be split across multiple utterances.
Be inclusive — false positives are filtered later.

## Lines ({len(lines)})
```json
{lines_json}
```

## Canvas Nodes ({len(nodes)})
```json
{nodes_json}
```

## Output
Return ONLY JSON. Every line MUST appear.
```json
{{{{
  "candidates": [
    {{{{"line_id": "L1", "candidate_node_ids": ["n1", "n3"], "candidate_count": 2}}}},
    {{{{"line_id": "L2", "candidate_node_ids": [], "candidate_count": 0}}}}
  ]
}}}}
```"""


def _run_coarse(evidence: Dict) -> Optional[Dict]:
    client = _get_client()
    if not client: return None
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)
    prompt = _make_coarse_prompt(evidence)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=32768,
            reasoning_effort="low", extra_body={"thinking": {"type": "enabled"}},
        )
        text = resp.choices[0].message.content or ""
        _log_llm("coarse", "all", prompt, text, time.time() - t0)
        return _parse_json(text)
    except Exception as e:
        logger.warning("Coarse failed: %s", e)
    return None


# ═══════════════════════════════════════════════════════════════════
# Stage A2: Fine match (all lines × deduped candidate nodes, one call)
# ═══════════════════════════════════════════════════════════════════

def _make_fine_prompt(evidence: Dict, line_candidates: Dict[str, List[str]]) -> str:
    """Build prompt for global fine matching: all lines + all candidate nodes."""
    all_lines = evidence.get("rewrite_lines", [])
    all_nodes = {n["node_id"]: n["prompt"] for n in evidence.get("canvas_nodes", [])}

    # Dedupe candidate nodes across all lines
    deduped_nids = set()
    for nids in line_candidates.values():
        deduped_nids.update(nids)

    lines_json = json.dumps([
        {"line_id": l["line_id"], "speaker": l["speaker"],
         "dialogue": l["original"],
         "scene": l.get("shot_scene", "")[:300],
         "start_sec": l["start_sec"], "end_sec": l["end_sec"],
         "candidates": line_candidates.get(l["line_id"], [])}
        for l in all_lines
    ], ensure_ascii=False, indent=2)

    nodes_json = json.dumps([
        {"node_id": nid, "prompt": all_nodes.get(nid, "")}
        for nid in sorted(deduped_nids)
    ], ensure_ascii=False, indent=2)

    return f"""## Role
For each line, find the canvas node whose prompt contains this dialogue
(exact or semantic equivalent). The dialogue may appear as a single utterance
or split across multiple quotes in the prompt (e.g. \"It's no use\" and
\"Forget me\" separately matching \"it's no use forget me\").

## Lines ({len(all_lines)})
```json
{lines_json}
```

## Candidate Nodes ({len(deduped_nids)})
```json
{nodes_json}
```

## Output
Return ONLY JSON. Every line MUST appear (either with a node_id or in unmatched).
```json
{{{{
  "line_matches": [
    {{{{"line_id": "L1", "node_id": "n1", "match_reasoning": "...", "original_dialogue_in_prompt": "..." or null, "confidence": 0.95}}}}
  ],
  "unmatched": [
    {{{{"line_id": "L2", "reason": "no candidate matches"}}}}
  ]
}}}}
```"""


def _run_fine(evidence: Dict, line_candidates: Dict[str, List[str]]) -> Optional[Dict]:
    client = _get_client()
    if not client: return None
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)
    prompt = _make_fine_prompt(evidence, line_candidates)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=32768,
            reasoning_effort="low", extra_body={"thinking": {"type": "enabled"}},
        )
        text = resp.choices[0].message.content or ""
        _log_llm("fine", "all", prompt, text, time.time() - t0)
        return _parse_json(text)
    except Exception as e:
        logger.warning("Fine failed: %s", e)
    return None


# ═══════════════════════════════════════════════════════════════════
# Stage B: Prompt Rewrite (per group)
# ═══════════════════════════════════════════════════════════════════

def _make_rewrite_prompt(ri: RewriteInput) -> str:
    changed = [l for l in ri.covered_lines if l.get("original") != l.get("rewritten")]
    unchanged = [l for l in ri.covered_lines if l.get("original") == l.get("rewritten")]
    changed_json = json.dumps(changed, ensure_ascii=False, indent=2)
    return f"""## Role
Rewrite a video generation prompt. ONLY change the spoken dialogue listed
under "Lines to Change". All other text — environment, actions, camera,
lighting, style keywords, resolution — must be preserved verbatim.

## Original Prompt
```
{ri.original_prompt}
```

## Lines to Change ({len(changed)})
Replace each ORIGINAL dialogue with the REWRITTEN version in the prompt.
```json
{changed_json}
```

## Lines to Keep Unchanged ({len(unchanged)})
These lines appear in the prompt but must NOT be changed. Keep their
dialogue exactly as in the original prompt.
{json.dumps([l.get('line_id') for l in unchanged]) if unchanged else '(none)'}

Return ONLY the rewritten prompt text."""


def run_rewrite(ri: RewriteInput) -> Tuple[str, str]:
    client = _get_client()
    if not client: return "", ""
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)
    prompt = _make_rewrite_prompt(ri)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=32768,
            reasoning_effort="low", extra_body={"thinking": {"type": "enabled"}},
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            text = "\n".join(lines).strip()
        _log_llm("rewrite", ri.group_id, prompt, text, time.time() - t0)
        return text, ""
    except Exception as e:
        logger.warning("Rewrite %s failed: %s", ri.group_id, e)
    return "", ""


# ═══════════════════════════════════════════════════════════════════
# Deterministic Grouping
# ═══════════════════════════════════════════════════════════════════

def _group_by_node(matches: List[LineNodeMatch], evidence: Dict) -> List[NodeGeneration]:
    rewrite_lines = evidence.get("rewrite_lines", [])
    neighbor_lines = evidence.get("neighbor_lines", [])
    line_times: Dict[str, Tuple[float, float]] = {}
    for rl in rewrite_lines + neighbor_lines:
        line_times[rl["line_id"]] = (float(rl.get("start_sec", 0)), float(rl.get("end_sec", 0)))

    by_node: Dict[str, List[LineNodeMatch]] = {}
    for m in matches:
        if m.node_id: by_node.setdefault(m.node_id, []).append(m)

    gens = []; gid = 0
    for node_id, ms in by_node.items():
        ms.sort(key=lambda m: line_times.get(m.line_id, (0, 0))[0])
        subgroups = []; cur = [ms[0]]
        for m in ms[1:]:
            pe = line_times.get(cur[-1].line_id, (0, 0))[1]
            cs = line_times.get(m.line_id, (0, 0))[0]
            cur_dur = line_times.get(cur[-1].line_id, (0, 0))[1] - line_times.get(cur[0].line_id, (0, 0))[0]
            if cs - pe > 3.0 or cur_dur + (line_times.get(m.line_id, (0, 0))[1] - cs) > 20.0 or len(cur) >= 4:
                subgroups.append(cur); cur = [m]
            else: cur.append(m)
        subgroups.append(cur)

        for sg in subgroups:
            gid += 1
            lids = [m.line_id for m in sg]
            times = [line_times.get(lid) for lid in lids if lid in line_times]
            if not times: continue
            start = min(t[0] for t in times); end = max(t[1] for t in times)
            if end - start < 4.0:
                # Only absorb neighbors from the same shots as matched lines
                allowed_shots = {int(nl.get("shot_number", -1)) for nl in neighbor_lines
                                 if nl.get("line_id") in lids}
                nearby = []
                for nl in neighbor_lines:
                    if int(nl.get("shot_number", -1)) not in allowed_shots: continue
                    nlid = nl.get("line_id", ""); ns = float(nl.get("start_sec", 0))
                    ne = float(nl.get("end_sec", 0))
                    if abs(ns - end) < 3.0 or abs(ne - start) < 3.0:
                        nearby.append((nlid, ns, ne))
                nearby.sort(key=lambda x: min(abs(x[1]-end), abs(x[2]-start)))
                for nlid, ns, ne in nearby[:2]:
                    lids.append(nlid); start = min(start, ns); end = max(end, ne)
                lids = sorted(set(lids))
            gens.append(NodeGeneration(
                group_id=f"G{gid}", covered_line_ids=sorted(set(lids)),
                matched_node_ids=[node_id],
                source_time_range=SourceTimeRange(start, end),
                line_matches=[m for m in sg],
                grouping_reasoning=f"node {node_id[:12]}, {len(sg)} lines",
                confidence=sum(m.confidence for m in sg) / max(1, len(sg)),
            ))

    # Merge overlapping
    gens.sort(key=lambda g: g.source_time_range.start_sec if g.source_time_range else 0)
    merged = []
    for g in gens:
        if not merged: merged.append(g); continue
        p = merged[-1]
        if (p.source_time_range and g.source_time_range and
                p.source_time_range.end_sec > g.source_time_range.start_sec + 0.05 and
                p.matched_node_ids != g.matched_node_ids):
            p.covered_line_ids = sorted(set(p.covered_line_ids) | set(g.covered_line_ids))
            p.matched_node_ids = list(set(p.matched_node_ids) | set(g.matched_node_ids))
            p.source_time_range.end_sec = max(p.source_time_range.end_sec, g.source_time_range.end_sec)
            p.line_matches.extend(g.line_matches)
            p.degradation_level += 1
            p.grouping_reasoning += f"; merged {g.group_id}"
            dur = p.source_time_range.end_sec - p.source_time_range.start_sec
            if dur > 30.0 or len(p.covered_line_ids) > 8:
                raise ValueError(
                    f"Merge exceeded limits: {p.group_id} has {len(p.covered_line_ids)} lines, "
                    f"{dur:.1f}s. Grouping logic cannot resolve — check A2 fine match results."
                )
        else: merged.append(g)
    return merged


# ═══════════════════════════════════════════════════════════════════
# Verifiers
# ═══════════════════════════════════════════════════════════════════

def _check_coverage(matches: List[LineNodeMatch], unmatched: List[UnmatchedLine], evidence: Dict) -> List[str]:
    errors = []
    all_ids = {rl["line_id"] for rl in evidence.get("rewrite_lines", [])}
    if not all_ids: return errors
    matched_ids = {m.line_id for m in matches}
    unmatched_ids = {u.line_id for u in unmatched}
    valid = set(all_ids)
    for nl in evidence.get("neighbor_lines", []): valid.add(nl["line_id"])
    for m in matches:
        if m.line_id not in valid: errors.append(f"unknown: {m.line_id}")
    for lid in all_ids:
        if lid not in matched_ids and lid not in unmatched_ids: errors.append(f"missing: {lid}")
    return errors


def _check_prompt(g: NodeGeneration, prompt: str) -> List[str]:
    errors = []
    if not prompt or not prompt.strip(): return ["empty prompt"]
    for m in g.line_matches:
        if not m.rewritten_line or m.original_line == m.rewritten_line: continue
        rw = m.rewritten_line
        if rw in prompt or rw.lower() in prompt.lower() or rw.rstrip('.!?,;:') in prompt: continue
        errors.append(f"'{rw[:60]}' not in prompt")
    return errors


# ═══════════════════════════════════════════════════════════════════
# Orchestrator
# ═══════════════════════════════════════════════════════════════════

def generate_plan_draft(evidence: Dict, canvas_nodes: Optional[List] = None) -> MatchResult:
    rewrite_lines = evidence.get("rewrite_lines", [])
    lid_map = {l["line_id"]: l for l in rewrite_lines}

    # ── Stage A1: Coarse ──
    logger.info("A1: coarse filter (%d lines × %d nodes)", len(rewrite_lines), len(evidence.get("canvas_nodes", [])))
    coarse = _run_coarse(evidence)
    if not coarse: raise ValueError("Coarse filter failed")
    line_cands = {c["line_id"]: c.get("candidate_node_ids", []) for c in coarse.get("candidates", [])}
    zero_cands = [lid for lid, cands in line_cands.items() if not cands]
    if zero_cands:
        logger.warning("A1: %d lines have 0 candidates, retrying with broader search: %s", len(zero_cands), zero_cands)
        coarse = _run_coarse(evidence)  # retry once
        if coarse:
            for c in coarse.get("candidates", []):
                lid = c["line_id"]
                if lid in zero_cands and c.get("candidate_node_ids"):
                    line_cands[lid] = c["candidate_node_ids"]
                    zero_cands.remove(lid)
        if zero_cands:
            logger.warning("A1: %d lines STILL 0 candidates after retry → unmatched: %s", len(zero_cands), zero_cands)
    logger.info("A1 done: %d lines, avg %.1f candidates, %d with 0",
                len(line_cands),
                sum(len(v) for v in line_cands.values()) / max(1, len(line_cands)),
                sum(1 for v in line_cands.values() if not v))

    # ── Stage A2: Fine match ──
    matches: List[LineNodeMatch] = []
    unmatched: List[UnmatchedLine] = []

    # Pre-mark lines with 0 candidates as unmatched
    for lid in zero_cands:
        ld = lid_map.get(lid, {})
        unmatched.append(UnmatchedLine(
            line_id=lid, reason="no candidate nodes from coarse filter",
            original=ld.get("original", ""), rewritten=ld.get("rewritten", ""),
            start_sec=float(ld.get("start_sec", 0)), end_sec=float(ld.get("end_sec", 0)),
            speaker=ld.get("speaker", ""), shot_scene=ld.get("shot_scene", ""),
            shot_number=int(ld.get("shot_number", 0))))

    # Only run A2 for lines WITH candidates
    lines_with_cands = {lid: cands for lid, cands in line_cands.items() if cands}
    if lines_with_cands:
        deduped = set(nid for nids in lines_with_cands.values() for nid in nids)
        logger.info("A2: fine match with %d candidates (%d lines)", len(deduped), len(lines_with_cands))
        fine = _run_fine(evidence, lines_with_cands)
        if not fine: raise ValueError("Fine match failed")

        for m in fine.get("line_matches", []):
            lid = m.get("line_id", ""); nid = m.get("node_id")
            ld = lid_map.get(lid, {})
            if nid:
                matches.append(LineNodeMatch(
                    line_id=lid, node_id=nid,
                    original_line=ld.get("original", ""), rewritten_line=ld.get("rewritten", ""),
                    match_reasoning=m.get("match_reasoning", ""),
                    original_dialogue_in_prompt=m.get("original_dialogue_in_prompt"),
                    confidence=float(m.get("confidence", 0)),
                ))
            else:
                unmatched.append(UnmatchedLine(
                    line_id=lid, reason=m.get("match_reasoning", "no match"),
                    original=ld.get("original", ""), rewritten=ld.get("rewritten", ""),
                    start_sec=float(ld.get("start_sec", 0)), end_sec=float(ld.get("end_sec", 0)),
                    speaker=ld.get("speaker", ""), shot_scene=ld.get("shot_scene", ""),
                    shot_number=int(ld.get("shot_number", 0))))

        for u in fine.get("unmatched", []):
            lid = u.get("line_id", ""); ld = lid_map.get(lid, {})
            unmatched.append(UnmatchedLine(
                line_id=lid, reason=u.get("reason", "unmatched"),
                original=ld.get("original", ""), rewritten=ld.get("rewritten", ""),
                start_sec=float(ld.get("start_sec", 0)), end_sec=float(ld.get("end_sec", 0)),
                speaker=ld.get("speaker", ""), shot_scene=ld.get("shot_scene", ""),
                shot_number=int(ld.get("shot_number", 0))))

    logger.info("A2 done: %d matched, %d unmatched", len(matches), len(unmatched))
    for u in unmatched:
        logger.warning("  unmatched: %s — \"%s\" → \"%s\"", u.line_id, u.original[:60], u.rewritten[:60])
    errs = _check_coverage(matches, unmatched, evidence)
    if errs: logger.warning("Coverage: %d errors: %s", len(errs), errs[:3])

    # ── Grouping ──
    gens = _group_by_node(matches, evidence)
    logger.info("Groups: %d", len(gens))
    for g in gens:
        logger.info("  %s: node=%s lines=%d [%.1f-%.1fs]",
                    g.group_id, (g.matched_node_ids[0] or "")[:12],
                    len(g.covered_line_ids),
                    g.source_time_range.start_sec if g.source_time_range else 0,
                    g.source_time_range.end_sec if g.source_time_range else 0)
    result = MatchResult(line_matches=matches, unmatched_lines=unmatched, node_generations=gens)

    # ── Stage B: Rewrite ──
    if canvas_nodes:
        node_map = {getattr(n, "node_id", ""): getattr(n, "prompt", "") for n in canvas_nodes}

        def rewrite_one(g: NodeGeneration) -> bool:
            primary = g.matched_node_ids[0] if g.matched_node_ids else ""
            orig = node_map.get(primary, "")
            # Only pass lines that actually changed (not neighbor lines)
            covered = [{"line_id": lid, "original": lid_map.get(lid, {}).get("original", ""),
                       "rewritten": lid_map.get(lid, {}).get("rewritten", ""),
                       "speaker": lid_map.get(lid, {}).get("speaker", "")}
                      for lid in g.covered_line_ids if lid in lid_map
                      and lid_map[lid].get("original", "") != lid_map[lid].get("rewritten", "")]
            if not covered: return True
            ri = RewriteInput(group_id=g.group_id, original_prompt=orig, covered_lines=covered)
            best = ""; best_e = float('inf')
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                for rf in concurrent.futures.as_completed([pool.submit(run_rewrite, ri) for _ in range(3)]):
                    rp, _ = rf.result()
                    if not rp: continue
                    e = len(_check_prompt(g, rp))
                    if e < best_e: best = rp; best_e = e
                    if best_e == 0: break
            if best:
                g.rewritten_prompt = best
                if best_e > 0: logger.warning("Rewrite %s: %d errors", g.group_id, best_e)
                return True
            return False

        logger.info("Rewriting %d groups...", len(gens))
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gens)) as pool:
            pool.map(rewrite_one, gens)
        ok = sum(1 for g in gens if g.has_prompt)
        logger.info("Rewrite: %d/%d groups have prompt", ok, len(gens))
        for g in gens:
            if not g.has_prompt:
                logger.error("  %s: EMPTY prompt", g.group_id)

    return result
