#!/usr/bin/env python3
"""
Stage 4: Video Generation & Concatenation

Parses a rewrite storyboard, generates new videos via seedance for nodes
with replaced prompts, downloads unchanged originals, concatenates all.
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import tempfile
import urllib.request
import uuid
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[2] / "generated"
sys.path.insert(0, str(Path("~/workspace/lingolens/backend").expanduser()))

for env_path in [
    Path("~/workspace/lingolens/backend/.env").expanduser(),
    Path("~/workspace/shakespeare/.env").expanduser(),
]:
    if env_path.exists():
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

from utils.aqinfo_seedance import AQInfoSeedanceClient, SeedanceModel, AssetType, SeedanceRatio, SeedanceResolution

client = AQInfoSeedanceClient()


def parse_storyboard(path: str) -> list[dict]:
    """Extract all nodes with metadata from storyboard, in order of appearance."""
    with open(path) as f:
        text = f.read()

    nodes = []
    current_shot = None
    i = 0
    lines = text.split("\n")

    while i < len(lines):
        ln = lines[i]

        m = re.match(r"## 镜头 (\d+)", ln)
        if m:
            current_shot = int(m.group(1))

        m = re.match(r"(🔄|📝) \*\*([^*]+) (?:Prompt 替换|Prompt)\*\*:", ln)
        if m and current_shot:
            node_name = m.group(2)
            is_replaced = (m.group(1) == "🔄")
            i += 1
            prompt = ""
            if i < len(lines) and lines[i].strip() == "```":
                i += 1
                while i < len(lines) and lines[i].strip() != "```":
                    prompt += lines[i] + "\n"
                    i += 1

            nodes.append({
                "shot": current_shot,
                "name": node_name,
                "replaced": is_replaced,
                "prompt": prompt.strip(),
            })
        i += 1

    # Second pass: extract ref images and video URLs per node
    for nd in nodes:
        name = nd["name"]
        im = re.search(rf"📸 \*\*{re.escape(name)} 参考图\*\* \((\d+)张\):\n((?:  - [^\n]+\n?)+)", text)
        if im:
            nd["images"] = re.findall(r"  - (https://[^\n]+)", im.group(2))
        else:
            nd["images"] = []

        vm = re.search(r"🎥 (https://[^\n]+)", text)
        if vm:
            nd["video_url"] = vm.group(1)

    print(f"Parsed {len(nodes)} nodes ({sum(1 for n in nodes if n['replaced'])} replaced)")
    return nodes


def build_original_node_videos(canvas_path: str, script_path: str) -> dict:
    sys.path.insert(0, str(Path("skills/canvas-storyboard").resolve()))
    from match_to_canvas import extract_lines_from_script, video_nodes, parse_nodes, build_original_storyboard_map

    with open(canvas_path) as f:
        canvas = json.load(f)
    lines = extract_lines_from_script(script_path)
    nodes = video_nodes(parse_nodes(canvas))
    mapping = build_original_storyboard_map(lines, nodes, 55)

    info = {}
    for line in lines:
        entry = mapping.get(line["line_id"])
        if not entry or not entry[0]:
            continue
        node = entry[0]
        name = node["name"]
        if name not in info:
            d = node["data_obj"]
            s = d.get("params", {}).get("settings", {})
            info[name] = {
                "video_url": (d.get("url", [None]) or [None])[0],
                "duration": int(s.get("duration", 15)),
            }
    return info


def download_image_locally(url: str) -> str:
    """Download an image to a temp file, return local path."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            fd, path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return path
        except Exception as e:
            if attempt == 2:
                print(f"    ❌ Download image failed: {e}")
                return ""
            time.sleep(2)
    return ""


async def upload_local_image(path: str, name: str) -> str:
    """Upload a local image to OSS and return a public URL for seedance."""
    if not path or not os.path.exists(path):
        return ""
    
    import oss2
    auth = oss2.Auth(os.environ['ALIYUN_ACCESS_KEY_ID'], os.environ['ALIYUN_ACCESS_KEY_SECRET'])
    bucket = oss2.Bucket(auth, os.environ['ALIYUN_OSS_ENDPOINT'], os.environ['ALIYUN_OSS_BUCKET'])
    
    key = f"seedance_refs/{name}_{uuid.uuid4().hex[:8]}.png"
    bucket.put_object_from_file(key, path)
    url = bucket.sign_url('GET', key, 86400)
    
    # Create seedance asset from OSS URL
    result = await client.create_asset(url=url, asset_type=AssetType.IMAGE, name=name)
    aid = result.get("data", {}).get("id", result.get("id", ""))
    if aid:
        await client.wait_for_asset(aid, max_wait_time=120)
    return aid


async def generate_via_seedance(nd: dict, duration: int) -> str:
    """Upload local refs → seedance → return new video URL."""
    name = nd["name"]
    prompt = nd["prompt"]
    urls = nd.get("images", [])

    if not urls:
        print(f"  [{name}] No images, skipping")
        return ""

    print(f"  [{name}] Downloading {len(urls)} images locally...")
    local_paths = []
    for u in urls:
        lp = download_image_locally(u)
        if lp:
            local_paths.append(lp)

    if not local_paths:
        print(f"  [{name}] ❌ All image downloads failed")
        return ""

    print(f"  [{name}] Uploading {len(local_paths)} images to seedance...")
    asset_ids = []
    for i, lp in enumerate(local_paths):
        aid = await upload_local_image(lp, f"gen_{name.replace(' ','_')}_{i}")
        if aid:
            asset_ids.append(aid)
        os.unlink(lp)  # cleanup temp

    if not asset_ids:
        return ""

    asset_urls = [f"asset://{aid}" for aid in asset_ids]
    print(f"  [{name}] Generating (seedance fast, {duration}s)...")
    result = await client.multimodal_reference_to_video(
        prompt=prompt, images=asset_urls,
        model=SeedanceModel.SEEDANCE_2_0_FAST,
        duration=duration, ratio=SeedanceRatio.RATIO_9_16,
        resolution=SeedanceResolution.RESOLUTION_720P,
        generate_audio=True, wait=True, max_wait_time=900,
    )
    return result.get("video_url", "")


def download_video(url: str, path: Path) -> bool:
    if not url:
        return False
    if path.exists():
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            with open(path, "wb") as f:
                f.write(data)
            print(f"    {path.stat().st_size//1024}KB")
            return True
        except Exception as e:
            if attempt == 2:
                print(f"    ❌ {e}")
                return False
            time.sleep(3)
    return False


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--storyboard", required=True)
    p.add_argument("--canvas", required=True)
    p.add_argument("--script", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Only process first N nodes (0=all)")
    args = p.parse_args()

    WORK_DIR.mkdir(exist_ok=True)
    vdir = WORK_DIR / "videos"
    vdir.mkdir(exist_ok=True)

    nodes = parse_storyboard(args.storyboard)
    orig_info = build_original_node_videos(args.canvas, args.script)

    if args.limit > 0:
        nodes = nodes[: args.limit]
        print(f"Limited to first {args.limit} nodes")

    if args.dry_run:
        for nd in nodes:
            dur = orig_info.get(nd["name"], {}).get("duration", "?")
            label = "🔄" if nd["replaced"] else "📝"
            print(f"  {label} Shot {nd['shot']}: {nd['name']} (dur={dur}s, imgs={len(nd.get('images',[]))})")
        sys.exit(0)

    video_files = []
    for idx, nd in enumerate(nodes):
        orig = orig_info.get(nd["name"], {})
        duration = orig.get("duration", 15)
        safe = re.sub(r"[^\w]", "_", nd["name"])[:40]
        path = vdir / f"{idx:02d}_shot{nd['shot']}_{safe}.mp4"

        if nd["replaced"] and nd.get("prompt"):
            new_url = await generate_via_seedance(nd, duration)
            url = new_url or nd.get("video_url") or orig.get("video_url")
            label = "NEW" if new_url else "FALLBACK"
        else:
            url = nd.get("video_url") or orig.get("video_url")
            label = "ORIG"

        print(f"  [{label}] Shot {nd['shot']}: {nd['name']} -> {path.name}")
        if download_video(url, path):
            video_files.append(str(path))

    if not video_files:
        print("No videos")
        sys.exit(1)

    concat_file = WORK_DIR / "concat.txt"
    with open(concat_file, "w") as f:
        for vf in video_files:
            f.write(f"file '{vf}'\n")

    output = Path(args.output)
    r = subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_file), "-c", "copy", str(output)], capture_output=True, text=True)
    if r.returncode == 0:
        print(f"✅ {output} ({output.stat().st_size//1024//1024}MB)")
    else:
        print(f"❌ {r.stderr[:300]}")


if __name__ == "__main__":
    asyncio.run(main())
