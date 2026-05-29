"""Canvas node matching: extract quoted dialogue from prompts, line-level fuzzy match, multi-node grouping.

Key insight: Canvas node prompts contain the actual dialogue in quotes (e.g., "This ceremony is boring.").
These are GROUND TRUTH for matching — more reliable than ASR transcription.

New multi-node API (preferred):
  match_lines_to_nodes(lines, nodes) → {node_id: [line_ids]}

Backward-compat single-node API (deprecated, kept for existing callers):
  match_canvas_node_for_shot(shot, nodes) → (Optional[CanvasNode], float)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from skills.timeline_plan.models import CanvasNode

TEXT_OVERLAP_THRESHOLD = 0.2
CONFIDENCE_THRESHOLD = 0.3


# ── Normalization helpers ───────────────────────────────────────────

def _normalize(text: str) -> str:
    """Normalize text: lowercase, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# ── Step 0: Legacy single-node matching (backward compat) ──────────

def text_overlap_score(dialogue_text: str, node_prompt: str) -> float:
    """Compute how much dialogue text appears in the node prompt.

    Uses word-level overlap with sliding window for robustness against
    ASR noise, punctuation differences, and mixed Chinese/English text.

    Args:
        dialogue_text: Combined dialogue from a ScriptShot's lines.
        node_prompt: Full prompt text from a canvas node.

    Returns:
        Score between 0.0 (no overlap) and 1.0 (full overlap).
    """
    d_norm = _normalize(dialogue_text)
    p_norm = _normalize(node_prompt)

    if not d_norm or not p_norm:
        return 0.0

    d_words = [w for w in d_norm.split() if len(w) >= 2]
    if not d_words:
        return 0.0

    hits = 0
    for w in d_words:
        if w in p_norm:
            hits += 1

    for window_size in range(min(4, len(d_words)), 0, -1):
        for i in range(len(d_words) - window_size + 1):
            phrase = " ".join(d_words[i : i + window_size])
            if phrase in p_norm:
                hits += window_size * 0.5
                break

    raw = hits / max(len(d_words), 1)
    return min(1.0, raw)


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Simple word-overlap based semantic similarity."""
    a_words = set(_normalize(text_a).split())
    b_words = set(_normalize(text_b).split())
    if not a_words or not b_words:
        return 0.0
    intersection = a_words & b_words
    return len(intersection) / max(len(a_words | b_words), 1)


def match_canvas_node_for_shot(
    shot: Any,
    nodes: List[CanvasNode],
    rewrite_lines: Optional[List[Any]] = None,
) -> Tuple[Optional[CanvasNode], float]:
    """Match a ScriptShot to the best canvas node (legacy single-node API).

    Priority signals:
    1. Dialogue text overlap with node prompt (primary)
    2. Scene description semantic similarity (tiebreaker)

    DEPRECATED: Prefer match_lines_to_nodes() for multi-node support.
    This returns only the BEST-matching node for the shot, which is wrong
    when a shot's lines are split across multiple nodes.

    Args:
        shot: ScriptShot with .lines[] and .scene_description.
        nodes: All available canvas nodes.
        rewrite_lines: Optional rewrite lines.

    Returns:
        (matched_node, confidence) tuple, or (None, 0.0).
    """
    if not nodes:
        return None, 0.0

    dialogue_text = " ".join(
        line.dialogue for line in (shot.lines or [])
        if getattr(line, "dialogue", "")
    )

    candidates: List[Tuple[CanvasNode, float]] = []
    for node in nodes:
        score = text_overlap_score(dialogue_text, node.prompt)
        if score >= TEXT_OVERLAP_THRESHOLD:
            candidates.append((node, score))

    if not candidates:
        return None, 0.0

    if len(candidates) > 1 and shot.scene_description:
        candidates.sort(
            key=lambda c: (
                c[1],
                _semantic_similarity(shot.scene_description, c[0].prompt),
            ),
            reverse=True,
        )
    else:
        candidates.sort(key=lambda c: c[1], reverse=True)

    best_node, confidence = candidates[0]
    if confidence >= CONFIDENCE_THRESHOLD:
        return best_node, confidence
    return None, 0.0


# ── Step 1: Extract quoted dialogue from prompts ────────────────────

# Patterns for quoted English text in Chinese/English mixed prompts
QUOTE_PATTERNS = [
    re.compile(r'"([^"]{3,})"'),                    # English double quotes
    re.compile(r'\u201c([^\u201d]{3,})\u201d'),     # Chinese left/right double quotes ""
    re.compile(r'\u300c([^\u300d]{3,})\u300d'),     # Corner brackets 「」
    re.compile(r'["\u201c]([^"\u201d]{3,})["\u201d]'),  # Mixed quote chars
]


def extract_quoted_dialogues(prompt: str) -> List[str]:
    """Extract all quoted English dialogue fragments from a canvas node prompt.

    Filters out Chinese-only text and short fragments (< 3 chars).
    Returns deduplicated list of dialogue strings.
    """
    found: List[str] = []
    seen = set()
    for pat in QUOTE_PATTERNS:
        for m in pat.finditer(prompt):
            text = m.group(1).strip()
            # Only keep fragments that contain English alphabet characters
            if re.search(r'[a-zA-Z]', text) and len(text) >= 3:
                normalized = ' '.join(text.split())  # collapse whitespace
                if normalized.lower() not in seen:
                    found.append(normalized)
                    seen.add(normalized.lower())
    return found


def build_node_quote_index(nodes: List[CanvasNode]) -> Dict[str, List[str]]:
    """Build a lookup: {node_id: [quoted_dialogue_1, quoted_dialogue_2, ...]}."""
    index: Dict[str, List[str]] = {}
    for node in nodes:
        quotes = extract_quoted_dialogues(node.prompt)
        if quotes:
            index[node.node_id] = quotes
    return index


# ── Step 2: Line-level matching ─────────────────────────────────────

def _fuzzy_match_text(asr_text: str, quotes: List[str]) -> float:
    """Fuzzy-match an ASR text against a list of ground-truth quotes.

    Returns best confidence score (0.0-1.0).
    Uses normalized word overlap with substring windowing.
    """
    def _norm(t: str) -> str:
        t = re.sub(r'[^\w\s]', ' ', t.lower())
        return re.sub(r'\s+', ' ', t).strip()

    a_norm = _norm(asr_text)
    a_words = [w for w in a_norm.split() if len(w) >= 2]
    if not a_words:
        return 0.0

    best = 0.0
    for quote in quotes:
        q_norm = _norm(quote)
        if not q_norm:
            continue

        # Word overlap
        hits = sum(1 for w in a_words if w in q_norm)
        word_score = hits / max(len(a_words), 1)

        # Phrase bonus: sliding window n-gram match
        phrase_bonus = 0.0
        for win in range(min(4, len(a_words)), 1, -1):
            for i in range(len(a_words) - win + 1):
                phrase = ' '.join(a_words[i:i+win])
                if phrase in q_norm:
                    phrase_bonus = win * 0.3
                    break
            if phrase_bonus > 0:
                break

        score = min(1.0, word_score + phrase_bonus)
        best = max(best, score)

    return best


def match_lines_to_nodes(
    lines: List[Any],
    nodes: List[CanvasNode],
    use_llm: bool = True,
) -> Dict[str, List[str]]:
    """Match script lines to canvas nodes, returning line groupings per node.

    Strategy:
    1. Extract quoted dialogues from all node prompts (ground truth)
    2. For each line, find the best-matching node by fuzzy matching
       the ASR dialogue against each node's extracted quotes
    3. Optionally use LLM for ambiguous cases (lines with low confidence)

    Args:
        lines: Script lines, each with .line_id, .dialogue attributes.
        nodes: All available canvas nodes.
        use_llm: If True, use LLM for low-confidence cases.

    Returns:
        Dict mapping node_id → list of line_ids assigned to that node.
    """
    if not nodes:
        return {}

    # Step 1: Build quote index
    quote_index = build_node_quote_index(nodes)

    # Also build node_id → CanvasNode map for reference
    node_map = {n.node_id: n for n in nodes}

    # Step 2: Match each line
    line_matches: Dict[str, List[str]] = {}  # node_id → [line_ids]
    unmatched: List[Any] = []

    for line in lines:
        dialogue = getattr(line, 'dialogue', '') or ''
        if not dialogue.strip():
            continue

        # Score against each node's quotes
        best_node = None
        best_score = 0.0
        for node_id, quotes in quote_index.items():
            score = _fuzzy_match_text(dialogue, quotes)
            if score > best_score:
                best_score = score
                best_node = node_id

        if best_node and best_score >= 0.2:
            line_matches.setdefault(best_node, []).append(line.line_id)
        else:
            unmatched.append(line)

    # Step 3: LLM fallback for unmatched or low-confidence matches
    if use_llm and unmatched:
        llm_matches = _llm_match_unmatched(unmatched, nodes, quote_index)
        for node_id, lids in llm_matches.items():
            line_matches.setdefault(node_id, []).extend(lids)

    return line_matches


def _llm_match_unmatched(
    lines: List[Any],
    nodes: List[CanvasNode],
    quote_index: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Use LLM to match unmatched lines to nodes.

    Much simpler than the old e2e match: per-line, not global alignment.
    """
    if not lines:
        return {}

    # Build a compact context: each node's quotes
    node_context = []
    for node in nodes:
        quotes = quote_index.get(node.node_id, [])
        if quotes:
            node_context.append({
                "node_id": node.node_id,
                "quotes": quotes[:10],  # Cap to avoid token waste
            })

    if not node_context:
        return {}

    lines_text = "\n".join(
        f'{l.line_id}: "{l.dialogue}"'
        for l in lines
    )

    quotes_text = "\n".join(
        f'{nc["node_id"]}: {nc["quotes"]}'
        for nc in node_context
    )

    system_msg = """Match script lines to canvas nodes based on dialogue similarity.
The node quotes are GROUND TRUTH. The line dialogue is ASR (may have errors).
For each line, pick the node whose quotes best match the dialogue.
Ignore capitalization, punctuation, minor typos, and trailing character names.
Output ONLY valid JSON with format: {"mappings": [{"line_id": "...", "node_id": "..."}, ...]}"""

    user_msg = f"""## Node Quotes (ground truth)
{quotes_text}

## Lines to Match
{lines_text}

Output the mapping for all {len(lines)} lines."""

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
        return {}

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=_os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"))
        resp = client.chat.completions.create(
            model=_os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0, max_tokens=4096,
        )
        text = resp.choices[0].message.content or ""
    except Exception:
        return {}

    import json
    try:
        obj_match = re.search(r'\{[^{]*"mappings"', text)
        if obj_match:
            start = obj_match.start()
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] in '{[':
                    depth += 1
                elif text[i] in '}]':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            data = json.loads(text[start:end])
        else:
            data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        try:
            data = json.loads(text.replace('\n', ' ').replace('  ', ' '))
        except (json.JSONDecodeError, ValueError):
            return {}

    mappings = data.get("mappings", data if isinstance(data, list) else [])
    result: Dict[str, List[str]] = {}
    for m in mappings:
        if not isinstance(m, dict):
            continue
        lid = m.get("line_id", "")
        nid = m.get("node_id", "")
        if lid and nid:
            result.setdefault(nid, []).append(lid)
    return result
