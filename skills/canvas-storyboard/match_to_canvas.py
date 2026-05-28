#!/usr/bin/env python3
"""
Stage 3: Canvas Storyboard — LLM e2e matching with Rule A/B scoring

Step A (no --rewrite): Match original script dialogue to canvas nodes → original storyboard
Step B (with --rewrite): Use original mapping, replace dialogue in prompts → rewrite storyboard

Matching uses LLM end-to-end with contextual continuity (Rule A) and dead-clip
discrimination (Rule B). Multiple runs are voted on by score_mapping().

Usage:
  # Step A: Generate original storyboard (single LLM run)
  python3 skills/canvas-storyboard/match_to_canvas.py \
    --script episode1_script.json \
    --canvas m2VuuIZfI \
    --output storyboard_ep1_original.md

  # Step A with voting (5 runs, pick best score):
  python3 skills/canvas-storyboard/match_to_canvas.py \
    --script episode1_script.json \
    --canvas m2VuuIZfI \
    --output storyboard_ep1_original.md \
    --llm-runs 5

  # Step B: Generate rewrite storyboard
  python3 skills/canvas-storyboard/match_to_canvas.py \
    --script episode1_script.json \
    --rewrite rewrites/ep1_A2.json \
    --canvas m2VuuIZfI \
    --output storyboard_ep1_A2.md
"""

import argparse
import json
import random
import re
import ssl
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CANVAS_API = "https://api.liblib.tv/api/canvas/project/share/detail"


def fetch_canvas(share_id: str) -> dict:
    url = f"{CANVAS_API}?shareId={share_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "CanvasStoryboard/3.0"})
    for verify in [True, False]:
        try:
            ctx = ssl.create_default_context()
            if not verify:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(req, timeout=60, context=ctx) as resp:
                data = json.loads(resp.read())
            if data.get("code") != 0:
                sys.exit(f"API error: {data}")
            return data["data"]
        except (ssl.SSLError, Exception):
            if verify:
                continue
            raise
    sys.exit("Cannot connect to canvas API")


def load_canvas_data(source: str) -> dict:
    if source.endswith(".json"):
        p = Path(source).resolve()
        if not p.exists():
            sys.exit(f"Canvas JSON not found: {p}")
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    print(f"Fetching canvas shareId={source} ...")
    return fetch_canvas(source)


def parse_nodes(data: dict) -> list[dict]:
    parsed = []
    for n in data.get("nodeList", []):
        nd = dict(n)
        try:
            nd["data_obj"] = json.loads(n.get("data", "{}"))
        except (json.JSONDecodeError, TypeError):
            nd["data_obj"] = None
        parsed.append(nd)
    return parsed


def video_nodes(parsed: list[dict]) -> list[dict]:
    return [n for n in parsed if n.get("type") == 3 and n.get("data_obj")]


def extract_lines_from_script(script_path: str) -> List[dict]:
    """Extract all dialogue lines with shot context from a ScriptInput JSON."""
    with open(script_path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    shots = data.get("script", {}).get("shots", [])
    for shot in shots:
        for line in shot.get("lines", []):
            lines.append({
                "line_id": line.get("line_id", ""),
                "speaker": line.get("speaker", ""),
                "original": line.get("dialogue", ""),
                "start_seconds": line.get("start_seconds"),
                "end_seconds": line.get("end_seconds"),
                "shot_number": shot.get("shot_number"),
                "shot_scene": shot.get("scene_description", ""),
            })
    return lines


def _normalize(text: str) -> str:
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()





def _find_dialogue_span(prompt: str, orig_norm: str) -> Optional[Tuple[int, int]]:
    pl = prompt.lower()
    words = [w for w in orig_norm.split() if len(w) >= 3 or w.isdigit()]
    if not words:
        return None
    for win in range(len(words), max(1, len(words) // 2), -1):
        for i in range(len(words) - win + 1):
            chunk = " ".join(words[i : i + win])
            idx = pl.find(chunk)
            if idx >= 0:
                end = idx + len(chunk)
                while end < len(prompt) and prompt[end] not in '.!?\n"\u201d\u300d':
                    end += 1
                end += int(end < len(prompt) and prompt[end] in '.!?\n"\u201d\u300d')
                while idx > 0 and pl[idx - 1] not in '\n."\u201c:：':
                    idx -= 1
                return (idx, end)
    return None


def replace_dialogue_in_prompt(prompt: str, original: str, rewritten: str) -> str:
    if not original or not rewritten or original == rewritten:
        return prompt
    oc = original.strip()
    rc = rewritten.strip()
    on = _normalize(oc.lower())
    pl = prompt.lower()

    for pat in [r'"([^"]{3,})"', r'\u201c([^\u201d]{3,})\u201d', r'\u300c([^\u300d]{3,})\u300d']:
        for m in re.finditer(pat, prompt):
            if on in _normalize(m.group(1).lower()) or (len(on) >= 10 and on[:10] in _normalize(m.group(1).lower())):
                return prompt[: m.start(1)] + rc + prompt[m.end(1):]

    for m in re.finditer(r'[：:]\s*(.{3,}?)(?:[.！。\n]|$)', prompt):
        ac = m.group(1).strip()
        if on in _normalize(ac.lower()):
            s = pl.find(ac.lower(), m.start(1))
            s = m.start(1) if s < 0 else s
            return prompt[:s] + rc + prompt[s + len(ac):]

    idx = pl.find(oc.lower())
    if idx >= 0:
        return prompt[:idx] + rc + prompt[idx + len(oc):]

    span = _find_dialogue_span(prompt, on)
    if span:
        return prompt[: span[0]] + rc + prompt[span[1]:]

    return prompt


def node_info(v: dict, repl_prompt: Optional[str] = None) -> dict:
    d = v["data_obj"]
    p = d.get("params", {})
    return {
        "name": v.get("name", "?"),
        "nodeKey": v.get("nodeKey", "")[:12],
        "video_urls": d.get("url", []),
        "ref_images": [img.get("url", "") for img in (p.get("imageList", []) or p.get("mixedList", []))[:12]],
        "prompt": repl_prompt if repl_prompt is not None else p.get("prompt", ""),
        "original_prompt": p.get("prompt", ""),
    }


def generate_original_storyboard(
    lines: List[dict], mapping: Dict[str, Tuple], title: str
) -> str:
    shots: Dict[int, List[dict]] = defaultdict(list)
    for l in lines:
        shots[l.get("shot_number", 0)].append(l)

    out = [f"# {title} — 原版分镜故事板\n"]
    out.append("> 原台词 ↔ 画布节点 映射表\n")
    shown = set()

    for sn in sorted(shots):
        sl = shots[sn]
        sc = sl[0].get("shot_scene", "")
        sp = sorted({l.get("speaker", "?") for l in sl})
        ss = min(l.get("start_seconds") or 0 for l in sl)
        se = max(l.get("end_seconds") or 0 for l in sl)

        out.append(f"## 镜头 {sn} ({se - ss:.1f}s) — {', '.join(sp)}")
        out.append(f"*{sc}*\n")
        out.append("| line_id | 台词 | 对话 | 对应画布节点 |")
        out.append("|---------|------|------|-------------|")

        last_nk = None
        nodes_meta = {}

        for l in sl:
            lid = l["line_id"]
            spk = l["speaker"]
            orig = l["original"]
            entry = mapping.get(lid)
            if entry and entry[0]:
                node, score = entry
                nk = node.get("nodeKey", "")
                if nk == last_nk:
                    nd = "*(同上)*"
                else:
                    nd = f"✅ {node.get('name', '?')} ({node.get('nodeKey', '')[:8]})"
                    if nk not in shown:
                        nodes_meta[nk] = node
                last_nk = nk
            else:
                nd = "—"
                last_nk = None
            out.append(f"| {lid} | 💬 {spk} | {orig} | {nd} |")

        out.append("")

        for nk, node in nodes_meta.items():
            if nk in shown:
                continue
            shown.add(nk)
            info = node_info(node)
            if info["video_urls"]:
                for vu in info["video_urls"]:
                    out.append(f"🎥 {vu}")
                out.append("")
            if info["ref_images"]:
                out.append(f"📸 **{info['name']} 参考图** ({len(info['ref_images'])}张):")
                for u in info["ref_images"]:
                    out.append(f"  - {u}")
                out.append("")
            if info["prompt"]:
                out.append(f"📝 **{info['name']} ({info['nodeKey'][:8]}) Prompt**:")
                out.append("```")
                out.append(info["prompt"][:2000])
                out.append("```")
            out.append("")
        out.append("---\n")

    mapped = sum(1 for l in lines if mapping.get(l["line_id"]) and mapping[l["line_id"]][0])
    out.append(f"*{mapped}/{len(lines)} lines mapped to canvas nodes*")
    return "\n".join(out)


def generate_rewrite_storyboard(
    lines: List[dict], mapping: Dict[str, Tuple], title: str, level: str
) -> str:
    shots: Dict[int, List[dict]] = defaultdict(list)
    for l in lines:
        shots[l.get("shot_number", 0)].append(l)

    out = [f"# {title} — {level} 等级改写分镜故事板\n"]
    out.append(f"> 在原版映射基础上，将 prompt 中的原台词替换为 {level} 改写台词\n")
    shown = set()
    replaced = 0
    not_found = 0
    total = 0

    for sn in sorted(shots):
        sl = shots[sn]
        sc = sl[0].get("shot_scene", "")
        sp = sorted({l.get("speaker", "?") for l in sl})
        ss = min(l.get("start_seconds") or 0 for l in sl)
        se = max(l.get("end_seconds") or 0 for l in sl)

        out.append(f"## 镜头 {sn} ({se - ss:.1f}s) — {', '.join(sp)}")
        out.append(f"*{sc}*\n")
        out.append(f"| line_id | 台词 | 原台词 | {level} 改写 | 对应画布节点 | Prompt 替换 |")
        out.append("|---------|------|--------|------------|-------------|-------------|")

        last_nk = None
        nodes_meta = {}

        for l in sl:
            total += 1
            lid = l["line_id"]
            spk = l["speaker"]
            orig = l["original"]
            rew = l.get("rewritten", orig)
            entry = mapping.get(lid)

            if entry and entry[0]:
                node, _ = entry
                nk = node.get("nodeKey", "")
                if nk == last_nk:
                    nd = "*(同上)*"
                    ps = "*(同上)*"
                else:
                    nd = f"{node.get('name', '?')} ({nk[:8]})"
                    op = node["data_obj"].get("params", {}).get("prompt", "")
                    np = replace_dialogue_in_prompt(op, orig, rew)
                    if np != op:
                        replaced += 1
                        ps = "✅ 已替换"
                    else:
                        not_found += 1
                        ps = "⚠️ 未找到替换位置"
                    if nk not in shown:
                        nodes_meta[nk] = (node, orig, rew, np if np != op else op)
                last_nk = nk
            else:
                nd = "—"
                ps = "（无对应节点）"
                last_nk = None

            out.append(f"| {lid} | 💬 {spk} | {orig} | {rew} | {nd} | {ps} |")
        out.append("")

        for nk, (node, orig_t, rew_t, prompt) in nodes_meta.items():
            if nk in shown:
                continue
            shown.add(nk)
            info = node_info(node, prompt)
            if info["video_urls"]:
                for vu in info["video_urls"]:
                    out.append(f"🎥 {vu}")
                out.append("")
            if info["ref_images"]:
                out.append(f"📸 **{info['name']} 参考图** ({len(info['ref_images'])}张):")
                for u in info["ref_images"]:
                    out.append(f"  - {u}")
                out.append("")
            if prompt != info["original_prompt"]:
                out.append(f"🔄 **{info['name']} ({nk[:8]}) Prompt 替换**:")
                out.append("```")
                out.append(prompt[:2000])
                out.append("```")
            else:
                out.append(f"📝 **{info['name']} ({nk[:8]}) Prompt**:")
                out.append("```")
                out.append(prompt[:2000])
                out.append("```")
            out.append("")
        out.append("---\n")

    mapped = sum(1 for l in lines if mapping.get(l["line_id"]) and mapping[l["line_id"]][0])
    out.append(f"*{mapped}/{total} lines have canvas nodes, {replaced} replaced, {not_found} not found*")
    return "\n".join(out)


def llm_end_to_end_match(
    all_lines: List[dict],
    all_nodes: list[dict],
) -> Dict[str, Tuple[Optional[dict], Optional[int]]]:
    """Single-shot LLM: provide all nodes + all lines, let LLM decide globally."""

    node_catalog = []
    for i, n in enumerate(all_nodes):
        p = n["data_obj"].get("params", {}).get("prompt", "")
        has_video = bool(n["data_obj"].get("url"))
        node_catalog.append({
            "id": i,
            "name": n["name"],
            "has_video": has_video,
            "prompt": p,
        })

    lines_text = []
    current_shot = None
    for l in all_lines:
        sn = l["shot_number"]
        if sn != current_shot:
            current_shot = sn
            lines_text.append(f"\n[Shot {sn}]")
            lines_text.append(f'scene: "{l.get("shot_scene", "")}"')
        lines_text.append(f'{l["line_id"]} | {l["speaker"]}: "{l["original"]}"')

    system_msg = """## Role
You are an expert Video Production Data Aligner. Your task is to map every messy ASR script line to its EXACT corresponding final production canvas node from Liblib TV, filtering out "dead/discarded clips" (废片) and iteration copies.

## Input Data Format
1. Script Lines: Format is `line_id | Speaker: "ASR Dialogue"`, preceded by `[Shot X]` and a `scene: "Visual description from Multimodal LLM"`.
   *ASR dialogue is noisy. Scene description is highly accurate visually.*
2. Node Catalog: A list of JSON objects `{"id": X, "name": "...", "has_video": true/false, "prompt": "..."}`.
   *Prompt contains ground-truth dialogue in quotes and scene descriptions.*

## The "Dead Clip" & Iteration Challenge (CRITICAL)
The Node Catalog contains many "dead clips" (废片) and duplicates (e.g., "视频节点 22 - 副本"). The final video ONLY uses the polished final versions.
- **Rule A**: Storytelling flows linearly. If Line N maps to Node X, Line N+1 should naturally map to Node X or Node X+1.
  Discard nodes that break the narrative timeline unless forced.
- **Rule B**: If multiple nodes have identical prompts, prioritize `has_video: true` and chronological fit.
  Never split consecutive lines across identical twin nodes.

## Matching Strategy (Priority Order)

**About ASR quality**: The script's dialogue comes from automatic speech recognition (ASR) and is often unreliable. Expect: character names appended to lines ("are they going crazy Donnie" vs "are they going crazy"), punctuation noise ("You are not a man." vs "You are not a man!"), word substitutions, and phantom words. The node prompts contain the actual dialogue spoken in the video — treat them as ground truth. When in doubt, trust the scene description over the ASR text.

1. Tier 1: Extract dialogue from `""` in node prompt. Compare with ASR.
   Ignore punctuation, capitalization, minor typos, and trailing character names.
2. Tier 2: Match script `scene:` description with node prompt scene (cross-lingual).
   Core meaning and subclause matching — focus on semantic similarity rather than token-level overlap.
3. Tier 3: If truly no matching node, assign null.

## Grouping rules — minimize nodes, avoid duplicates

- If one node's prompt contains dialogue for multiple consecutive lines, assign ALL those lines to that node.
- When multiple nodes could match a given line, prefer the one that also covers adjacent lines — fewer overall nodes is better.
- NEVER assign a line to a node whose prompt does not contain it (or a close approximation), even if that node covers adjacent lines.
- If two nodes are different copies containing the SAME English dialogue, pick only ONE of them. Do not split a continuous conversation across two copies of the same scene.

## Output Format
Output ONLY valid JSON.

```json
{"mappings": [
  {"line_id": "p001_l001", "node_id": 9, "reason": "Tier1 exact dialogue match; selected over Node 10 (copy) to maintain chronological order"},
  {"line_id": "p007_l001", "node_id": 63, "reason": "Tier2 scene match: vending machine"},
  {"line_id": "p099_l001", "node_id": null, "reason": "no relevant node in catalog"}
]}
```

The `reason` field MUST include: (1) which Tier was used, and (2) if Rule A or Rule B influenced the choice, state how (e.g., "Selected over Node 10 (copy) to maintain chronological order", "Preferred has_video=true node").

## Verification Checklist
- Every `line_id` from the script appears exactly once in the output?
- No dead copy node that breaks timeline?
- All `node_id` integers or null?
- Consecutive lines within the same shot map to the same or adjacent nodes (Rule A)?
- When twin nodes exist, the one with `has_video=true` was preferred (Rule B)?"""

    import json as _j
    user_msg = f"""## Canvas Nodes

{_j.dumps(node_catalog, ensure_ascii=False)}

## Script Lines

{chr(10).join(lines_text)}

Output the mapping for all {len(all_lines)} lines."""

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
    base_url = _os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    model = _os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash")

    if not api_key:
        print("  ⚠️  No DEEPSEEK_API_KEY, skipping e2e LLM match")
        return {}

    prompt_chars = len(system_msg) + len(user_msg)
    print(f"  🤖 LLM e2e match: {len(all_lines)} lines, {len(all_nodes)} nodes, ~{prompt_chars // 4} tokens")
    print(f"     model={model} base_url={base_url}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=128000,
        )
        msg = resp.choices[0].message
        text = msg.content or ""
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}")
        return {}

    if not text:
        print("  ⚠️  Empty LLM response")
        return {}

    import re as _re
    obj_match = _re.search(r'\{[^{]*"mappings"', text)
    if not obj_match:
        obj_match = _re.search(r'\[\s*\{', text)
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
        json_str = text[start:end]
    else:
        print(f"  ⚠️  Could not parse LLM response: {text[:300]}")
        return {}

    try:
        data = _j.loads(json_str)
    except _j.JSONDecodeError:
        cleaned = json_str.replace('\n', ' ').replace('  ', ' ')
        try:
            data = _j.loads(cleaned)
        except _j.JSONDecodeError as e:
            print(f"  ⚠️  Invalid JSON: {e}")
            print(f"     ...{json_str[-200:]}")
            return {}

    mappings = data.get("mappings", data if isinstance(data, list) else [])

    result = {}
    matched = unmatched = 0
    for m in mappings:
        if not isinstance(m, dict):
            continue
        lid = m.get("line_id")
        if not lid:
            continue
        nid = m.get("node_id")
        reason = m.get("reason", "")
        if nid is not None and isinstance(nid, int) and 0 <= nid < len(all_nodes):
            node = all_nodes[nid]
            result[lid] = (node, 95)
            print(f"     ✅ {lid} → #{nid} {node['name']} [{reason}]")
            matched += 1
        else:
            result[lid] = (None, None)
            print(f"     —  {lid} → NONE [{reason}]")
            unmatched += 1

    print(f"     LLM e2e result: {matched} matched, {unmatched} unmatched")
    return result


def score_mapping(
    mapping: Dict[str, Tuple[Optional[dict], Optional[int]]],
    all_lines: List[dict],
    all_nodes: Optional[list] = None,
) -> int:
    """Score a mapping. Higher is better.

    Penalties:
    - Uncovered lines: -100 per line
    - Rule A (continuity): -30 when consecutive lines in same shot jump to non-adjacent nodes
    - Rule B (dead clips): -20 when has_video=false node chosen over a has_video=true twin
    Bonus:
    - +5 per matched line
    """
    score = 0

    # Penalty 1: Uncovered lines
    uncovered = 0
    for l in all_lines:
        entry = mapping.get(l["line_id"])
        if not entry or not entry[0]:
            uncovered += 1
    score -= uncovered * 100

    # Bonus: matched lines
    score += (len(all_lines) - uncovered) * 5

    if all_nodes is None:
        return score

    # Build node index lookup (same object identity)
    node_to_idx: Dict[int, int] = {id(n): i for i, n in enumerate(all_nodes)}

    # Rule B: Detect twin nodes (identical normalized prompt, different has_video)
    prompt_to_nodes: Dict[str, List[Tuple[int, bool]]] = defaultdict(list)
    for i, n in enumerate(all_nodes):
        p = n["data_obj"].get("params", {}).get("prompt", "")
        key = _normalize(p.lower()) if p else ""
        if len(key) >= 20:
            prompt_to_nodes[key].append((i, bool(n["data_obj"].get("url"))))

    # Nodes that are dead clips (has_video=false) with a has_video=true twin
    dead_clip_ids: set = set()
    for key, members in prompt_to_nodes.items():
        if len(members) <= 1:
            continue
        has_live = any(hv for _, hv in members)
        if has_live:
            for idx, hv in members:
                if not hv:  # has_video=false twin exists alongside has_video=true
                    dead_clip_ids.add(idx)

    # Rule B penalty: lines mapped to dead clips
    rule_b_violations = 0
    for l in all_lines:
        entry = mapping.get(l["line_id"])
        if entry and entry[0]:
            idx = node_to_idx.get(id(entry[0]))
            if idx is not None and idx in dead_clip_ids:
                rule_b_violations += 1
    score -= rule_b_violations * 20

    # Rule A: Continuity — consecutive lines in same shot should not jump far
    shot_lines: Dict[Optional[int], List[dict]] = defaultdict(list)
    for l in all_lines:
        sn = l.get("shot_number")
        if sn is not None:
            shot_lines[sn].append(l)

    rule_a_violations = 0
    for sn, lines in shot_lines.items():
        lines_sorted = sorted(lines, key=lambda x: x["line_id"])
        for i in range(len(lines_sorted) - 1):
            curr = mapping.get(lines_sorted[i]["line_id"])
            nxt = mapping.get(lines_sorted[i + 1]["line_id"])
            if not curr or not curr[0] or not nxt or not nxt[0]:
                continue
            curr_idx = node_to_idx.get(id(curr[0]))
            nxt_idx = node_to_idx.get(id(nxt[0]))
            if curr_idx is not None and nxt_idx is not None:
                # Large node-id gap within same shot = potential timeline break
                if abs(curr_idx - nxt_idx) > 5:
                    rule_a_violations += 1
    score -= rule_a_violations * 30

    return score


def llm_e2e_vote(
    all_lines: List[dict],
    all_nodes: list[dict],
    num_runs: int = 3,
) -> Dict[str, Tuple[Optional[dict], Optional[int]]]:
    """Run LLM e2e matching multiple times with shuffled node order, pick the best-scoring result.

    Shuffling the node catalog each run mitigates LLM 'lost in the middle' bias,
    where nodes at the edges of the catalog receive disproportionate attention.
    Scoring always uses the original node order for Rule A continuity checks.
    """
    best_mapping, best_score = None, -99999
    for run in range(num_runs):
        print(f"\n  --- Run {run + 1}/{num_runs} ---")
        shuffled_nodes = list(all_nodes)
        random.shuffle(shuffled_nodes)
        mapping = llm_end_to_end_match(all_lines, shuffled_nodes)
        if not mapping:
            continue
        score = score_mapping(mapping, all_lines, all_nodes)
        print(f"     Score: {score} (best: {best_score})")
        if score > best_score:
            best_score = score
            best_mapping = mapping
    print(f"\n  🏆 Selected run with score {best_score}")
    if best_mapping is None:
        print(f"  ⚠️  All {num_runs} LLM runs returned empty results — check DEEPSEEK_API_KEY and model availability")
    return best_mapping if best_mapping else {}


def main():
    p = argparse.ArgumentParser(description="Stage 3: Canvas Storyboard — LLM e2e matching")
    p.add_argument("--script", required=True, help="Original script JSON (ScriptInput format)")
    p.add_argument("--rewrite", default=None, help="Optional: rewrite JSON for step B")
    p.add_argument("--canvas", required=True, help="Canvas shareId or local JSON")
    p.add_argument("--output", required=True, help="Output markdown path")
    p.add_argument("--llm-runs", type=int, default=5, help="Number of LLM runs for voting (default: 5)")
    args = p.parse_args()

    print(f"Loading script: {args.script}")
    original_lines = extract_lines_from_script(args.script)
    print(f"  {len(original_lines)} lines, {len(set(l['shot_number'] for l in original_lines))} shots")

    canvas_data = load_canvas_data(args.canvas)
    nodes = video_nodes(parse_nodes(canvas_data))
    print(f"  {len(nodes)} video nodes")

    if args.llm_runs > 1:
        mapping = llm_e2e_vote(original_lines, nodes, args.llm_runs)
    else:
        print(f"\n  🤖 LLM matching...")
        mapping = llm_end_to_end_match(original_lines, nodes)

    if args.rewrite:
        print(f"Loading rewrite: {args.rewrite}")
        with open(args.rewrite, encoding="utf-8") as f:
            rw = json.load(f)
        title = rw.get("title", "Untitled")
        level = rw.get("level", "?")

        rewrite_by_id = {l["line_id"]: l for l in rw.get("lines", [])}
        for l in original_lines:
            r = rewrite_by_id.get(l["line_id"], {})
            l["rewritten"] = r.get("rewritten", l["original"])

        result = generate_rewrite_storyboard(original_lines, mapping, title, level)
    else:
        title = Path(args.script).stem
        result = generate_original_storyboard(original_lines, mapping, title)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"Done: {args.output} ({Path(args.output).stat().st_size // 1024}KB)")


if __name__ == "__main__":
    main()
