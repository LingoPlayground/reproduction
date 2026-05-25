#!/usr/bin/env python3
"""
LibLib Canvas → Script Storyboard Pipeline

用法:
  python3 pipeline.py \\
    --share-id m2VuuIZfI \\
    --script /path/to/script_lines.json \\
    --output storyboard_ep1.md \\
    --episode-name "Episode 1: 毕业典礼与最后通牒"

流程:
  1. 从 api.liblib.tv 拉取画布数据 (nodeList + connectionList)
  2. 解析所有视频节点的 prompt、参考图URL、输出视频URL
  3. 读取剧本 JSON (script_lines 格式)
  4. 模糊匹配台词到 Canvas prompt → 找到对应生成版本
  5. 按镜头+台词输出 storyboard markdown
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import Optional


# ── API ──────────────────────────────────────────────────────────────

CANVAS_API = "https://api.liblib.tv/api/canvas/project/share/detail"


def fetch_canvas(share_id: str) -> dict:
    """拉取 LibLib 画布数据。"""
    url = f"{CANVAS_API}?shareId={share_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "AnalyzeScriptWithCanvas/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        sys.exit(f"❌ 拉取画布数据失败: {e}")
    if data.get("code") != 0:
        sys.exit(f"❌ API 返回错误: {data}")
    return data["data"]


# ── Parse ────────────────────────────────────────────────────────────

def parse_nodes(data: dict) -> list[dict]:
    """解析画布节点，展开 data 字段。"""
    nodes = data.get("nodeList", [])
    parsed = []
    for n in nodes:
        nd = dict(n)
        try:
            nd["data_obj"] = json.loads(n.get("data", "{}"))
        except (json.JSONDecodeError, TypeError):
            nd["data_obj"] = None
        parsed.append(nd)
    return parsed


def video_nodes(parsed: list[dict]) -> list[dict]:
    """筛选类型=3 的视频节点。"""
    return [n for n in parsed if n.get("type") == 3 and n.get("data_obj")]


# ── Matching ─────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """去除标点，统一空格，用于子串匹配。"""
    # 去除非字母数字中文的字符，保留空格
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fuzzy_match_score(dialogue_lower: str, prompt_lower: str) -> int:
    """模糊评分 (0-100): 对话是否出现在 prompt 中。"""
    dl_norm = _normalize(dialogue_lower)
    pr_norm = _normalize(prompt_lower)

    if len(dl_norm) < 5:
        return 0

    # 直接子串匹配（标准化后）
    if dl_norm in pr_norm:
        return 100

    words = [w for w in dl_norm.split() if len(w) >= 3 or w.isdigit()]
    if not words:
        return 0

    # 滑窗子串匹配
    best_sub = 0
    for win in range(len(words), 1, -1):
        for i in range(len(words) - win + 1):
            sub = " ".join(words[i : i + win])
            if sub in pr_norm:
                best_sub = max(best_sub, win * 100 // len(words))
    if best_sub > 0:
        return best_sub

    # 去掉前导词
    for skip in range(1, min(4, len(words))):
        sub = " ".join(words[skip:])
        if sub in pr_norm:
            return 80

    # 词命中率
    hits = sum(1 for w in words if w in pr_norm)
    return hits * 100 // max(len(words), 1)


def find_best_match(
    dialogue: str, nodes: list[dict], min_score: int = 40
) -> Optional[dict]:
    """为一条台词找到最匹配的 Canvas 节点（按 updatedAtMs 取最新）。"""
    dl = dialogue.lower()
    if len(dl) < 5:
        return None

    best, best_score = None, 0
    for v in nodes:
        prompt = v["data_obj"].get("params", {}).get("prompt", "").lower()
        score = fuzzy_match_score(dl, prompt)
        if score > best_score:
            best_score = score
            best = v
        elif score == best_score and best:
            # 分数相同时取更新时间更晚的
            if v.get("updatedAtMs", 0) > best.get("updatedAtMs", 0):
                best = v

    return best if best and best_score >= min_score else None


# ── Format ───────────────────────────────────────────────────────────

def format_node(v: dict) -> str:
    """将 Canvas 节点格式化为 markdown。"""
    d = v["data_obj"]
    params = d.get("params", {})
    task = d.get("taskInfo", {})
    output_urls = d.get("url", [])
    input_imgs = params.get("imageList", []) or params.get("mixedList", [])
    settings = params.get("settings", {})
    rm = d.get("_resourceMeta", {})
    status_icon = "✅" if task.get("status") == 2 else "⏳"

    lines = [
        f"**{status_icon} {v['name']}** `{v['nodeKey'][:12]}...` | "
        f"{params.get('model','?')} | "
        f"{settings.get('ratio','?')} {settings.get('resolution','?')} | "
        f"{settings.get('duration','?')}s",
    ]

    if output_urls:
        for u in output_urls:
            lines.append(f"🎥 {u}")

    if rm.get("items"):
        for item in rm["items"]:
            lines.append(
                f"  → {item.get('width','?')}x{item.get('height','?')} | "
                f"{item.get('durationSec','?')}s | "
                f"{item.get('byteSize',0)//1024}KB"
            )

    if input_imgs:
        limit = 12
        lines.append(f"📸 参考图 ({len(input_imgs)}张):")
        for img in input_imgs[:limit]:
            lines.append(f"  - {img.get('url','?')}")
        if len(input_imgs) > limit:
            lines.append(f"  ... 等 {len(input_imgs)} 张")

    prompt = params.get("prompt", "")
    if prompt:
        lines.append(f"📝 Prompt:\n```\n{prompt}\n```")

    return "\n".join(lines)


# ── Storyboard Generator ─────────────────────────────────────────────

def generate_storyboard(
    script_lines: list[dict],
    nodes: list[dict],
    episode_name: str,
) -> str:
    """生成完整 storyboard markdown。"""
    # 按 shot_number 分组
    shots: dict[int, dict] = {}
    for line in script_lines:
        sn = line["shot_number"]
        if sn not in shots:
            shots[sn] = {
                "lines": [],
                "start": line["start_seconds"],
                "end": line["end_seconds"],
            }
        shots[sn]["lines"].append(line)
        shots[sn]["end"] = max(shots[sn]["end"], line["end_seconds"])

    out = [f"# {episode_name}\n"]

    for sn in sorted(shots):
        shot = shots[sn]
        slines = shot["lines"]
        scene = slines[0]["shot_scene"]
        duration = shot["end"] - shot["start"]
        speakers = sorted({l["speaker"] for l in slines})

        out.append(f"## 镜头 {sn} ({duration:.1f}s) — {', '.join(speakers)}")
        out.append(f"*{scene}*\n")

        last_node_key = None
        for line in slines:
            dialogue = line["dialogue"]
            out.append(
                f"### 💬 {line['speaker']}: \"{dialogue}\" ({line['start_seconds']:.1f}s)\n"
            )

            match = find_best_match(dialogue, nodes)
            if match:
                if match["nodeKey"] == last_node_key:
                    out.append("*(同上)*\n")
                else:
                    out.append(format_node(match))
                    out.append("")
                last_node_key = match["nodeKey"]
            else:
                out.append("*（未匹配到 Canvas 版本）*\n")
                last_node_key = None

        out.append("---\n")

    return "\n".join(out)


# ── CLI ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LibLib Canvas → Script Storyboard 分析管线"
    )
    parser.add_argument(
        "--share-id", required=True, help="LibLib 画布 shareId"
    )
    parser.add_argument(
        "--script", required=True, help="剧本 script_lines.json 路径"
    )
    parser.add_argument(
        "--output", default="storyboard.md", help="输出 markdown 路径"
    )
    parser.add_argument(
        "--episode-name",
        default="Untitled Episode",
        help="剧集名称",
    )
    args = parser.parse_args()

    # 1. 拉取画布
    print(f"📡 拉取画布 shareId={args.share_id} ...")
    canvas_data = fetch_canvas(args.share_id)
    nodes = video_nodes(parse_nodes(canvas_data))
    print(f"   解析到 {len(nodes)} 个视频节点")

    # 2. 读取剧本
    print(f"📖 读取剧本 {args.script} ...")
    with open(args.script, "r", encoding="utf-8") as f:
        script_lines = json.load(f)
    print(f"   {len(script_lines)} 行台词")

    # 3. 生成 storyboard
    print(f"🔗 匹配并生成 storyboard ...")
    result = generate_storyboard(script_lines, nodes, args.episode_name)

    # 4. 写入
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result)

    size_kb = os.path.getsize(args.output) // 1024
    print(f"✅ 已生成: {args.output} ({size_kb}KB)")


if __name__ == "__main__":
    main()
