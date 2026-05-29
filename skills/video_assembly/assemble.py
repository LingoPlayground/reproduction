#!/usr/bin/env python3
"""Stage 4: Video assembly from TimelinePlan."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
import tempfile
import urllib.request
import uuid
from pathlib import Path
from typing import List

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

# Lazy import — only available when lingolens is installed
try:
    from utils.aqinfo_seedance import AQInfoSeedanceClient, SeedanceModel, AssetType, SeedanceRatio, SeedanceResolution
    _seedance_client = AQInfoSeedanceClient()
    SEEDANCE_AVAILABLE = True
except ImportError:
    SEEDANCE_AVAILABLE = False


def normalize_seedance_duration(target_sec: float) -> int:
    """Round to nearest integer second, clamped to [5, 30]."""
    return max(5, min(30, round(target_sec)))


def normalize_segment_encoding(input_path: str, output_path: str) -> None:
    """Re-encode segment to consistent format: libx264 high, yuv420p, aac 44.1kHz."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-c:v", "libx264", "-profile:v", "high",
        "-pix_fmt", "yuv420p", "-crf", "18",
        "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
        output_path,
    ], capture_output=True, check=True)


def normalize_audio_loudness(input_path: str, output_path: str) -> None:
    """Apply EBU R128 loudness normalization."""
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-c:v", "copy", output_path,
    ], capture_output=True, check=True)


def _write_concat_file(segment_paths: List[str], concat_path: str) -> str:
    """Write ffmpeg concat file listing all segments."""
    with open(concat_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    return concat_path


def _download_image_locally(url: str) -> str:
    """Download a reference image to a temp file. Returns local path or empty string."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            fd, path = tempfile.mkstemp(suffix=".png")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return path
        except Exception:
            if attempt == 2:
                return ""
            time.sleep(2)
    return ""


async def _upload_local_image(path: str, name: str) -> str:
    """Upload image to OSS → create seedance asset → return asset ID."""
    if not path or not os.path.exists(path):
        return ""
    if not SEEDANCE_AVAILABLE:
        return ""
    try:
        import oss2
        auth = oss2.Auth(os.environ['ALIYUN_ACCESS_KEY_ID'], os.environ['ALIYUN_ACCESS_KEY_SECRET'])
        bucket = oss2.Bucket(auth, os.environ['ALIYUN_OSS_ENDPOINT'], os.environ['ALIYUN_OSS_BUCKET'])
        key = f"seedance_refs/{name}_{uuid.uuid4().hex[:8]}.png"
        bucket.put_object_from_file(key, path)
        url = bucket.sign_url('GET', key, 86400)
        result = await _seedance_client.create_asset(url=url, asset_type=AssetType.IMAGE, name=name)
        aid = result.get("data", {}).get("id", result.get("id", ""))
        if aid:
            await _seedance_client.wait_for_asset(aid, max_wait_time=120)
        return aid
    except Exception as e:
        print(f"    ⚠️  Upload failed: {e}")
        return ""


async def _generate_via_seedance(item: dict, duration: int) -> str:
    """Generate a video via seedance for a single TimelinePlanItem.

    Args:
        item: TimelinePlanItem as dict (from JSON).
        duration: Target duration in seconds (integer, 5-30).

    Returns:
        URL of generated video, or empty string on failure.
    """
    if not SEEDANCE_AVAILABLE:
        return ""

    prompt = item.get("rewritten_prompt", "")
    ref_images = item.get("ref_images", [])

    if not prompt or not ref_images:
        return ""

    shot_num = item.get("shot_number", 0)
    name = f"shot_{shot_num}"

    print(f"    [{name}] Downloading {len(ref_images)} images...")
    local_paths = []
    for u in ref_images:
        lp = _download_image_locally(u)
        if lp:
            local_paths.append(lp)

    if not local_paths:
        print(f"    [{name}] ❌ All image downloads failed")
        return ""

    print(f"    [{name}] Uploading {len(local_paths)} images to seedance...")
    asset_ids = []
    for i, lp in enumerate(local_paths):
        aid = await _upload_local_image(lp, f"gen_{name}_{i}")
        if aid:
            asset_ids.append(aid)
        os.unlink(lp)

    if not asset_ids:
        return ""

    asset_urls = [f"asset://{aid}" for aid in asset_ids]
    print(f"    [{name}] Generating (seedance fast, {duration}s)...")
    try:
        result = await _seedance_client.multimodal_reference_to_video(
            prompt=prompt, images=asset_urls,
            model=SeedanceModel.SEEDANCE_2_0_FAST,
            duration=duration, ratio=SeedanceRatio.RATIO_9_16,
            resolution=SeedanceResolution.RESOLUTION_720P,
            generate_audio=True, wait=True, max_wait_time=900,
        )
        return result.get("video_url", "")
    except Exception as e:
        print(f"    [{name}] ❌ seedance failed: {e}")
        return ""


def _download_video(url: str, path: Path) -> bool:
    """Download a generated video from URL to local path."""
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
            print(f"    Downloaded {path.stat().st_size//1024}KB")
            return True
        except Exception as e:
            if attempt == 2:
                print(f"    ❌ Download failed: {e}")
                return False
            time.sleep(3)
    return False


async def assemble_video(
    plan_path: str,
    original_video: str,
    output_path: str,
    skip_seedance: bool = False,
) -> str:
    """Assemble final video from TimelinePlan.

    Args:
        plan_path: Path to timeline_plan.json.
        original_video: Path to original complete video.
        output_path: Desired output path for final.mp4.
        skip_seedance: Skip seedance generation (use original segments).

    Returns:
        Path to assembled video.
    """
    with open(plan_path, encoding="utf-8") as f:
        plan_data = json.load(f)

    items = plan_data.get("items", [])
    if not items:
        raise ValueError("TimelinePlan has no items")

    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "segments"
    work_dir.mkdir(exist_ok=True)

    segment_paths: List[str] = []

    for idx, item in enumerate(items):
        source = item.get("source", "original")
        seg_path = str(work_dir / f"seg_{idx:03d}_shot{item['shot_number']}.mp4")

        if source == "original" or skip_seedance:
            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{item['start_sec']:.3f}",
                "-to", f"{item['end_sec']:.3f}",
                "-i", original_video, "-c", "copy", seg_path,
            ], capture_output=True, check=True)
            print(f"  [ORIG] Shot {item['shot_number']}: {item['start_sec']:.1f}s-{item['end_sec']:.1f}s")
        elif source == "seedance":
            duration = item.get("seedance_duration", normalize_seedance_duration(
                item["end_sec"] - item["start_sec"]
            ))
            video_url = await _generate_via_seedance(item, duration)
            if video_url and _download_video(video_url, Path(seg_path)):
                print(f"  [SEED] Shot {item['shot_number']}: seedance {duration}s → {seg_path}")
            else:
                # Fallback to original segment
                subprocess.run([
                    "ffmpeg", "-y", "-ss", f"{item['start_sec']:.3f}",
                    "-to", f"{item['end_sec']:.3f}",
                    "-i", original_video, "-c", "copy", seg_path,
                ], capture_output=True, check=True)
                print(f"  [SEED-FB] Shot {item['shot_number']}: seedance failed → original fallback")

        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
            segment_paths.append(seg_path)

    if not segment_paths:
        raise RuntimeError("No valid segments produced")

    # Normalize encoding + audio
    print(f"\n  Normalizing {len(segment_paths)} segments...")
    normalized_paths: List[str] = []
    for idx, sp in enumerate(segment_paths):
        np_path = str(work_dir / f"norm_{idx:03d}.mp4")
        normalize_segment_encoding(sp, np_path)
        loud_path = str(work_dir / f"loud_{idx:03d}.mp4")
        normalize_audio_loudness(np_path, loud_path)
        normalized_paths.append(loud_path)

    # Concatenate
    concat_file = str(work_dir / "concat.txt")
    _write_concat_file(normalized_paths, concat_file)
    print(f"\n  Concatenating {len(normalized_paths)} segments...")
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-c", "copy", output_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        r = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c:v", "libx264", "-c:a", "aac", output_path,
        ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Concat failed: {r.stderr[:300]}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final: {output_path} ({size_mb:.1f}MB)")
    return output_path


async def main():
    p = argparse.ArgumentParser(description="Stage 4: Assemble final video")
    p.add_argument("--plan", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-seedance", action="store_true")
    args = p.parse_args()
    await assemble_video(args.plan, args.video, args.output, args.skip_seedance)

if __name__ == "__main__":
    asyncio.run(main())
