"""Canvas node matching: LLM end-to-end line-to-node matching with CoT + voting.

Key insight: Canvas node prompts contain the actual dialogue in quotes (e.g., "This ceremony is boring.").
These are GROUND TRUTH for matching — more reliable than ASR transcription.

API:
  match_lines_to_nodes(lines, nodes, num_runs=3) → (node_groups, line_confidences)
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from skills.timeline_plan.models import CanvasNode


# ── Quality scoring helpers ───────────────────────────────────────────

def _score_mapping(mapping: Dict[str, str], lines: List[Any]) -> float:
    """Score a mapping run: matches minus contiguity penalties.
    
    Penalizes when consecutive lines from the same shot map to different nodes
    that aren't adjacent in the result.
    """
    if not mapping:
        return 0.0
    
    score = float(len(mapping))
    
    # Group lines by shot_number
    shot_lines: Dict[int, List[Any]] = {}
    for l in lines:
        sn = getattr(l, 'shot_number', 0)
        lid = getattr(l, 'line_id', '')
        if lid in mapping:
            shot_lines.setdefault(sn, []).append(l)
    
    # Penalize when consecutive lines in same shot → different nodes
    for sn, slines in shot_lines.items():
        slines.sort(key=lambda l: getattr(l, 'line_id', ''))
        for i in range(len(slines) - 1):
            n1 = mapping.get(getattr(slines[i], 'line_id', ''))
            n2 = mapping.get(getattr(slines[i+1], 'line_id', ''))
            if n1 and n2 and n1 != n2:
                score -= 0.5  # Penalty for split
    
    return max(0.0, score)


def _compute_consistency(results: List[Dict[str, str]]) -> Dict[str, float]:
    """Compute per-line confidence from cross-run agreement.
    
    confidence = (runs agreeing on node) / (total runs where line was matched)
    """
    if not results:
        return {}
    
    line_runs: Dict[str, List[str]] = {}
    for mapping in results:
        for lid, nid in mapping.items():
            line_runs.setdefault(lid, []).append(nid)
    
    confidences: Dict[str, float] = {}
    for lid, nodes in line_runs.items():
        most_common_count = Counter(nodes).most_common(1)[0][1]
        confidences[lid] = most_common_count / len(nodes)
    
    return confidences


# ── Main matching API ─────────────────────────────────────────────────

def match_lines_to_nodes(
    lines: List[Any],
    nodes: List[CanvasNode],
    num_runs: int = 3,
) -> Tuple[Dict[str, List[str]], Dict[str, float]]:
    """Match script lines to canvas nodes using LLM CoT + quality-scored voting.
    
    LLM first identifies actual spoken dialogue in each node's prompt (CoT),
    then matches each ASR line to the node containing that dialogue.
    Multiple runs with shuffled node order reduce positional bias.
    Quality scoring penalizes contiguity violations.
    
    Args:
        lines: Script lines with .line_id, .dialogue, .speaker, .shot_number, .shot_scene.
        nodes: All available canvas nodes.
        num_runs: Number of LLM runs for voting (default 3).
    
    Returns:
        Tuple of (node_groups, line_confidences):
        - node_groups: {node_id: [line_id, ...]}
        - line_confidences: {line_id: confidence (0.0-1.0)}
    """
    if not nodes or not lines:
        return {}, {}
    
    # Run LLM matching multiple times with shuffled node order
    import random
    results: List[Dict[str, str]] = []  # Each result: {line_id → node_id}
    scores: List[float] = []
    
    for run in range(num_runs):
        shuffled = list(enumerate(nodes))
        if run > 0:
            random.shuffle(shuffled)
        
        mapping = _llm_match_run(lines, shuffled)
        if mapping:
            results.append(mapping)
            score = _score_mapping(mapping, lines)
            scores.append(score)
    
    if not results:
        return {}, {}
    
    # Pick the best run by quality score
    best_idx = scores.index(max(scores))
    best_mapping = results[best_idx]
    
    # Compute per-line confidence from cross-run consistency
    line_confidences = _compute_consistency(results)
    
    # Group line_ids by node_id
    node_groups: Dict[str, List[str]] = {}
    for line_id, node_id in best_mapping.items():
        node_groups.setdefault(node_id, []).append(line_id)
    
    return node_groups, line_confidences


# ── LLM matching run ──────────────────────────────────────────────────

def _llm_match_run(
    lines: List[Any],
    indexed_nodes: List[tuple],
) -> Optional[Dict[str, str]]:
    """Single LLM run: CoT dialogue extraction + line-to-node matching.
    
    Returns {line_id → node_id} or None on failure.
    """
    # Build compact node catalog — use index as node reference, full prompt for LLM context
    node_entries = []
    for idx, node in indexed_nodes:
        prompt = node.prompt
        if len(prompt) > 3000:
            prompt = prompt[:3000] + "..."
        node_entries.append({
            "id": idx,
            "node_id": node.node_id,
            "prompt": prompt,
        })
    
    # Build line catalog with rich context
    line_entries = []
    for l in lines:
        line_entries.append({
            "line_id": getattr(l, 'line_id', ''),
            "dialogue": getattr(l, 'dialogue', '') or '',
            "speaker": getattr(l, 'speaker', '') or '',
            "shot_number": getattr(l, 'shot_number', 0),
            "shot_scene": getattr(l, 'shot_scene', '') or '',
            "start_seconds": getattr(l, 'start_seconds', None),
            "end_seconds": getattr(l, 'end_seconds', None),
        })
    
    system_msg = """## Role
You match script dialogue lines to the canvas nodes that generated them.

## Step 1: Extract Dialogue from Node Prompts (CoT)
Each node's prompt is a video generation instruction. It contains:
- Scene descriptions, camera directions, visual style (in Chinese)
- Actual spoken dialogue (in English, often inside quotation marks "")
- Non-dialogue quoted text: banner signs ("CLASS OF 2026"), sound effects ("黑胶唱片划痕声"), inner thoughts, character descriptions

For each node, identify ONLY the actual spoken English dialogue lines. Ignore everything else.

## Step 2: Match Lines to Nodes
For each script line, find the node whose prompt contains that dialogue.

Matching signals (priority order):
1. Dialogue text in the node's prompt (primary — node prompt is ground truth)
2. Speaker attribution — same character should map to nodes where that character appears
3. Scene description — the visual context should match the node's scene

The script dialogue comes from ASR — expect minor errors (punctuation, capitalization, word substitutions). Match semantically.

## Contiguity Constraint
Lines from the same shot (same shot_number) should map to the same or adjacent nodes.
Lines from the same speaker in consecutive lines should typically stay in the same node.
Avoid mapping lines from widely separated shots to the same node unless the dialogue content strongly supports it.

## Output
JSON only:
```json
{"mappings": [
  {"line_id": "p001_l001", "node_index": 0},
  {"line_id": "p001_l003", "node_index": 2}
]}
```
Use `node_index` (the "id" field from the node catalog), NOT `node_id`.

If no node matches a line, omit it from the output."""

    import json as _j
    user_msg = f"""## Nodes
{_j.dumps(node_entries, ensure_ascii=False)}

## Lines to Match
{_j.dumps(line_entries, ensure_ascii=False)}

Output the mapping. First identify dialogue in each node's prompt, then match."""

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
        return None

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
            max_tokens=4096,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return None

    # Parse JSON response
    try:
        obj_match = re.search(r'\{[^{]*"mappings"', text)
        if obj_match:
            start = obj_match.start()
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] in '{[': depth += 1
                elif text[i] in '}]':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            data = _j.loads(text[start:end])
        else:
            data = _j.loads(text)
    except (_j.JSONDecodeError, ValueError):
        try:
            data = _j.loads(text.replace('\n', ' ').replace('  ', ' '))
        except (_j.JSONDecodeError, ValueError):
            return None

    mappings = data.get("mappings", data if isinstance(data, list) else [])
    
    # Convert node_index → node_id
    idx_to_node_id = {idx: node.node_id for idx, node in indexed_nodes}
    
    result: Dict[str, str] = {}
    for m in mappings:
        if not isinstance(m, dict):
            continue
        lid = m.get("line_id", "")
        nidx = m.get("node_index")
        if lid and nidx is not None and nidx in idx_to_node_id:
            result[lid] = idx_to_node_id[nidx]
    
    return result
