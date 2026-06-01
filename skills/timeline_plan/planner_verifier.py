"""Planner Verifier: split for two-stage pipeline.

verify_match: checks coverage, duplicates, time, known IDs (no prompt check)
verify_prompt: checks dialogue in rewritten prompt for a single group
"""
from __future__ import annotations

from typing import Any, Dict, List, Set

from skills.timeline_plan.planner_models import MatchResult, NodeGeneration


def _time_tolerance(duration_sec: float) -> float:
    return max(0.5, min(2.0, duration_sec * 0.15))


# ═══════════════════════════════════════════════════════════════════
# Stage A: Match Verification
# ═══════════════════════════════════════════════════════════════════

def verify_match(result: MatchResult, evidence: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    valid_lids = _get_line_ids(evidence)
    valid_nids = _get_node_ids(evidence)

    # Schema
    if not result.node_generations and not result.unmatched_lines:
        errors.append("schema: both node_generations and unmatched_lines empty")
    for g in result.node_generations:
        if not g.group_id:
            errors.append("schema: empty group_id")
        if not g.covered_line_ids:
            errors.append(f"schema: {g.group_id} empty covered_line_ids")
        if not g.line_matches:
            errors.append(f"schema: {g.group_id} empty line_matches")

    # Known IDs
    for g in result.node_generations:
        for lid in g.covered_line_ids:
            if lid not in valid_lids:
                errors.append(f"unknown_id: {g.group_id} unknown line '{lid}'")
        for nid in g.matched_node_ids:
            if nid not in valid_nids:
                errors.append(f"unknown_id: {g.group_id} unknown node '{nid}'")
        for m in g.line_matches:
            if m.line_id not in valid_lids:
                errors.append(f"unknown_id: {g.group_id} line_match unknown line '{m.line_id}'")
    for u in result.unmatched_lines:
        if u.line_id not in valid_lids:
            errors.append(f"unknown_id: unmatched unknown line '{u.line_id}'")

    # Coverage
    all_ids = valid_lids
    covered: Dict[str, List[str]] = {}
    for g in result.node_generations:
        for lid in g.covered_line_ids:
            covered.setdefault(lid, []).append(g.group_id)
    unmatched = {u.line_id for u in result.unmatched_lines}
    for lid in all_ids:
        if lid not in covered and lid not in unmatched:
            errors.append(f"coverage: line {lid} missing")
        if lid in covered and lid in unmatched:
            errors.append(f"coverage: line {lid} both covered and unmatched")
    for lid, groups in covered.items():
        if len(groups) > 1:
            errors.append(f"duplicate: line {lid} in {groups}")

    # line_matches must match covered_line_ids (only rewritten lines)
    rewrite_ids = {rl["line_id"] for rl in evidence.get("rewrite_lines", [])}
    for g in result.node_generations:
        cv = set(g.covered_line_ids) & rewrite_ids
        mv = {m.line_id for m in g.line_matches}
        if mv - set(g.covered_line_ids):
            errors.append(f"match: {g.group_id} line_matches for non-covered lines: {mv - set(g.covered_line_ids)}")
        if cv - mv:
            errors.append(f"match: {g.group_id} missing line_matches for: {cv - mv}")

    # Time
    line_times: Dict[str, tuple] = {}
    for rl in evidence.get("rewrite_lines", []):
        line_times[rl["line_id"]] = (float(rl.get("start_sec", 0)), float(rl.get("end_sec", 0)))
    for g in result.node_generations:
        tr = g.source_time_range
        if tr is None or not g.covered_line_ids:
            continue
        if tr.start_sec < 0:
            errors.append(f"time: {g.group_id} negative start")
        if tr.end_sec <= tr.start_sec:
            errors.append(f"time: {g.group_id} end <= start")
        times = [line_times[lid] for lid in g.covered_line_ids if lid in line_times]
        if not times:
            continue
        actual_start = min(t[0] for t in times)
        actual_end = max(t[1] for t in times)
        tol = _time_tolerance(tr.end_sec - tr.start_sec)
        if tr.start_sec > actual_start + tol:
            errors.append(f"time: {g.group_id} start {tr.start_sec:.1f}s too late (actual {actual_start:.1f}s, tol {tol:.1f}s)")
        if tr.end_sec < actual_end - tol:
            errors.append(f"time: {g.group_id} end {tr.end_sec:.1f}s too early (actual {actual_end:.1f}s, tol {tol:.1f}s)")

    return errors


# ═══════════════════════════════════════════════════════════════════
# Stage B: Prompt Verification
# ═══════════════════════════════════════════════════════════════════

def verify_prompt(group: NodeGeneration, rewritten_prompt: str) -> List[str]:
    errors: List[str] = []
    if not rewritten_prompt or not rewritten_prompt.strip():
        errors.append(f"prompt: {group.group_id} empty rewritten_prompt")
        return errors
    for m in group.line_matches:
        if m.original_line == m.rewritten_line:
            continue
        rewritten = m.rewritten_line
        # Check that the core content of the rewritten dialogue appears
        # (loose substring match, not exact — LLM may add quotes/formatting)
        if rewritten and _loose_match(rewritten, rewritten_prompt):
            continue
        if rewritten:
            errors.append(f"prompt: {group.group_id} line {m.line_id} dialogue '{rewritten[:80]}' not in prompt")
    return errors


def _loose_match(dialogue: str, prompt: str) -> bool:
    """Check if dialogue appears in prompt with loose matching."""
    if dialogue in prompt:
        return True
    # Try without trailing punctuation
    stripped = dialogue.rstrip('.!?,;:')
    if stripped and stripped in prompt:
        return True
    # Try case-insensitive
    if dialogue.lower() in prompt.lower():
        return True
    return False


# ── Helpers ────────────────────────────────────────────────────────

def _get_line_ids(evidence: Dict[str, Any]) -> Set[str]:
    ids: Set[str] = {rl["line_id"] for rl in evidence.get("rewrite_lines", [])}
    for nl in evidence.get("neighbor_lines", []):
        ids.add(nl["line_id"])
    return ids


def _get_node_ids(evidence: Dict[str, Any]) -> Set[str]:
    return {n.get("node_id", "") for n in evidence.get("canvas_nodes", []) if n.get("node_id")}
