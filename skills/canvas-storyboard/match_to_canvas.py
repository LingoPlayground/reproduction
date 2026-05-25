#!/usr/bin/env python3
"""
Stage 3: Canvas Storyboard — Two-step process

Step A (no --rewrite): Match original script dialogue to canvas nodes → original storyboard
Step B (with --rewrite): Use original mapping, replace dialogue in prompts → rewrite storyboard

Usage:
  # Step A: Generate original storyboard
  python3 skills/canvas-storyboard/match_to_canvas.py \
    --script episode1_script.json \
    --canvas m2VuuIZfI \
    --output storyboard_ep1_original.md

  # Step B: Generate rewrite storyboard (uses same --script for mapping)
  python3 skills/canvas-storyboard/match_to_canvas.py \
    --script episode1_script.json \
    --rewrite rewrites/ep1_A2.json \
    --canvas m2VuuIZfI \
    --output storyboard_ep1_A2.md
"""

import argparse
import json
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


def fuzzy_match_score(dialogue_lower: str, prompt_lower: str) -> int:
    dl_norm = _normalize(dialogue_lower)
    pr_norm = _normalize(prompt_lower)
    total_words = len(dl_norm.split())
    if len(dl_norm) < 5 or total_words < 2:
        return 0
    if dl_norm in pr_norm:
        return 100
    words = [w for w in dl_norm.split() if len(w) >= 3 or w.isdigit()]
    if not words:
        return 0
    best_sub = 0
    for win in range(len(words), 1, -1):
        for i in range(len(words) - win + 1):
            sub = " ".join(words[i : i + win])
            if sub in pr_norm:
                best_sub = max(best_sub, win * 100 // len(words))
    if best_sub > 0:
        return best_sub
    for skip in range(1, min(4, len(words))):
        sub = " ".join(words[skip:])
        if sub in pr_norm:
            return 80
    hits = sum(1 for w in words if w in pr_norm)
    if len(words) < total_words:
        return hits * 100 // total_words
    return hits * 100 // max(len(words), 1)


def match_line_to_node(
    dialogue: str, scene_desc: str, nodes: list[dict], min_score: int = 55
) -> Optional[Tuple[dict, int]]:
    dl = dialogue.strip().lower()
    sd = scene_desc.strip().lower()
    if len(dl) < 5 and len(sd) < 5:
        return None

    best, best_score = None, 0
    for v in nodes:
        prompt = v["data_obj"].get("params", {}).get("prompt", "").lower()
        dialogue_score = fuzzy_match_score(dl, prompt)
        scene_score = fuzzy_match_score(sd, prompt) if sd else 0

        # Combined: dialogue is primary (weight 0.7), scene confirms (weight 0.3)
        if dialogue_score >= 80 or dialogue_score > scene_score * 2:
            combined = dialogue_score
            if scene_score >= 60:
                combined = min(100, combined + 10)  # bonus for scene match
        elif scene_score >= 60 and dialogue_score < 40:
            combined = scene_score  # scene-driven match when dialogue is weak
        else:
            combined = max(dialogue_score, scene_score)

        if combined > best_score:
            best_score, best = combined, v
        elif combined == best_score and best:
            if v.get("updatedAtMs", 0) > best.get("updatedAtMs", 0):
                best = v

    if best and best_score >= min_score:
        return (best, best_score)
    return None


def build_original_storyboard_map(
    original_lines: List[dict], nodes: list[dict], min_score: int = 55
) -> Dict[str, Tuple[Optional[dict], Optional[int]]]:
    unique_texts = {}
    for line in original_lines:
        dialogue = line["original"].strip()
        scene = line.get("shot_scene", "").strip()
        key = (dialogue, scene)
        if key not in unique_texts:
            result = match_line_to_node(dialogue, scene, nodes, min_score)
            unique_texts[key] = result if result else (None, None)

    result = {}
    matched = 0
    for line in original_lines:
        dialogue = line["original"].strip()
        scene = line.get("shot_scene", "").strip()
        key = (dialogue, scene)
        result[line["line_id"]] = unique_texts.get(key, (None, None))
        if unique_texts.get(key) and unique_texts[key][0]:
            matched += 1

    unique = len(unique_texts)
    unique_matched = sum(1 for v in unique_texts.values() if v and v[0])
    print(f"Original→node mapping: {unique_matched}/{unique} unique (dialogue+scene) mapped ({matched}/{len(original_lines)} lines)")

    # Post-processing: very short names/calls (≤10 chars after strip) inherit
    # the node from surrounding lines in the same shot
    fixed = 0
    for i, line in enumerate(original_lines):
        lid = line["line_id"]
        entry = result.get(lid, (None, None))
        if not entry or entry[0] is not None:
            continue
        dialogue = line["original"].strip()
        if len(dialogue) > 10:
            continue
        shot = line.get("shot_number")
        # Find nearest mapped line in the same shot
        for offset in [1, -1, 2, -2, 3, -3]:
            ni = i + offset
            if 0 <= ni < len(original_lines) and original_lines[ni].get("shot_number") == shot:
                neighbor = result.get(original_lines[ni]["line_id"])
                if neighbor and neighbor[0]:
                    result[lid] = (neighbor[0], 95)
                    fixed += 1
                    break
    if fixed:
        print(f"  → {fixed} short-name lines inherited node from neighbors")

    return result


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
                    nd = f"✅ {node.get('name', '?')}"
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
                out.append(f"📝 **{info['name']} Prompt**:")
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
                    nd = node.get("name", "?")
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
                out.append(f"🔄 **{info['name']} Prompt 替换**:")
                out.append("```")
                out.append(prompt[:2000])
                out.append("```")
            else:
                out.append(f"📝 **{info['name']} Prompt**:")
                out.append("```")
                out.append(prompt[:2000])
                out.append("```")
            out.append("")
        out.append("---\n")

    mapped = sum(1 for l in lines if mapping.get(l["line_id"]) and mapping[l["line_id"]][0])
    out.append(f"*{mapped}/{total} lines have canvas nodes, {replaced} replaced, {not_found} not found*")
    return "\n".join(out)


def node_summary(v: dict) -> str:
    p = v["data_obj"].get("params", {}).get("prompt", "")
    import re as _re
    quotes = _re.findall(r'"([^"]{3,60})"', p)
    quote_str = " | ".join(q[:60] for q in quotes[:3])
    summary = p[:200].replace("\n", " ")
    if quote_str:
        summary += f" [台词: {quote_str}]"
    return f"{v['name']} | {summary}"


def llm_match_unmapped(
    unmapped: List[dict],
    all_nodes: list[dict],
    env_file: Optional[str] = None,
) -> Dict[str, Tuple[Optional[dict], Optional[int]]]:
    if not unmapped:
        return {}

    node_summaries = []
    for i, n in enumerate(all_nodes):
        node_summaries.append(f"[{i}] {node_summary(n)}")

    lines_text = []
    for l in unmapped:
        lines_text.append(
            f"line_id={l['line_id']} | scene=\"{l.get('shot_scene','')[:150]}\" | "
            f"speaker={l.get('speaker','?')} | dialogue=\"{l['original']}\""
        )

    prompt = (
        "你是剧本-画布场景匹配器。请为每行未匹配的台词找到对应的画布视频节点。\n\n"
        "【最重要】按场景匹配——先看场景描述，再看台词：\n\n"
        "第1步（场景匹配-必做）：\n"
        "  对比每行的 scene（场景描述）与每个节点的 prompt 前200字符。\n"
        "  场景关键词必须一致，否则排除该节点：\n"
        '    scene 含"毕业典礼/DJ"→ 只匹配 prompt 含"毕业典礼/DJ/典礼"的节点\n'
        '    scene 含"台阶/女友哭诉"→ 只匹配 prompt 含"台阶/女孩/女友"的节点\n'
        '    scene 含"贩卖机/扫脸/校园"→ 只匹配 prompt 含"贩卖机/扫脸/校园"的节点\n'
        '    scene 含"办公室/父亲/通牒"→ 只匹配 prompt 含"办公室/父亲/通牒"的节点\n'
        '    scene 含"客厅/漆黑/关灯"→ 只匹配 prompt 含"客厅/漆黑/关灯"的节点\n'
        "  场景不匹配的节点直接跳过，绝不要选。\n\n"
        "第2步（台词匹配-辅助）：\n"
        "  在场景匹配的候选节点中，再看对话原文或近义表达是否出现。\n"
        '  "Face test okay" → "扫脸认证"；"Money has run out" → "没钱了/余额不足"\n\n'
        "第3步：场景和台词都不匹配 → node_index 设为 null\n\n"
        f"画布节点列表（共{len(all_nodes)}个）：\n{chr(10).join(node_summaries)}\n\n"
        f"未匹配的台词行（{len(unmapped)}行）：\n{chr(10).join(lines_text)}\n\n"
        "严格输出一个 JSON 数组：\n"
        '[{"line_id":"p007_l001","node_index":0,"reason":"场景匹配+台词匹配"}]'
    )

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
    model = _os.environ.get("LLM_MATCH_MODEL", "deepseek-chat")

    if not api_key:
        print("  ⚠️  No DEEPSEEK_API_KEY, skipping LLM match")
        return {}

    print(f"  🤖 LLM semantic matching {len(unmapped)} lines against {len(all_nodes)} nodes...")
    print(f"     model={model} base_url={base_url}")

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}")
        return {}

    if not text:
        print("  ⚠️  Empty LLM response")
        return {}

    # Parse JSON (may be wrapped in ```json ... ```)
    import re as _re
    json_match = _re.search(r'\[\s*\{', text)
    if json_match:
        start = json_match.start()
        depth = 0
        end = start
        for i in range(start, len(text)):
            if text[i] == '[': depth += 1
            elif text[i] == ']':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        json_str = text[start:end]
    else:
        json_str = None

    if not json_str:
        print(f"  ⚠️  Could not parse LLM response: {text[:200]}")
        return {}

    try:
        results = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Invalid JSON: {e}")
        print(f"     ...{json_str[-100:]}")
        return {}

    if isinstance(results, dict) and "line_id" in results:
        results = [results]

    mapping = {}
    for r in (results if isinstance(results, list) else [results]):
        if not isinstance(r, dict):
            continue
        lid = r.get("line_id")
        ni = r.get("node_index")
        reason = r.get("reason", "")
        if ni is not None and isinstance(ni, int) and 0 <= ni < len(all_nodes):
            mapping[lid] = (all_nodes[ni], 95)
            print(f"     ✅ {lid} → {all_nodes[ni]['name']} ({reason})")
        else:
            mapping[lid] = (None, None)
            print(f"     —  {lid} → NONE ({reason})")
    return mapping
def main():
    p = argparse.ArgumentParser(description="Stage 3: Canvas Storyboard")
    p.add_argument("--script", required=True, help="Original script JSON (ScriptInput format)")
    p.add_argument("--rewrite", default=None, help="Optional: rewrite JSON for step B")
    p.add_argument("--canvas", required=True, help="Canvas shareId or local JSON")
    p.add_argument("--output", required=True, help="Output markdown path")
    p.add_argument("--min-score", type=int, default=55)
    p.add_argument("--llm", action="store_true", help="Use LLM for semantic matching of unmapped lines")
    args = p.parse_args()

    print(f"Loading script: {args.script}")
    original_lines = extract_lines_from_script(args.script)
    print(f"  {len(original_lines)} lines, {len(set(l['shot_number'] for l in original_lines))} shots")

    canvas_data = load_canvas_data(args.canvas)
    nodes = video_nodes(parse_nodes(canvas_data))
    print(f"  {len(nodes)} video nodes")

    mapping = build_original_storyboard_map(original_lines, nodes, args.min_score)

    if args.llm:
        unmapped = [(lid, entry) for lid, entry in mapping.items() if not entry or not entry[0]]
        if unmapped:
            unmapped_lines = [l for l in original_lines if l["line_id"] in dict(unmapped)]
            llm_results = llm_match_unmapped(unmapped_lines, nodes)
            for lid, result in llm_results.items():
                mapping[lid] = result

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
