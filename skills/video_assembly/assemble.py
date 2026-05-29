#!/usr/bin/env python3
"""Stage 4: Video assembly from TimelinePlan."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import List


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
            # Fallback to original segment (seedance integration deferred)
            subprocess.run([
                "ffmpeg", "-y", "-ss", f"{item['start_sec']:.3f}",
                "-to", f"{item['end_sec']:.3f}",
                "-i", original_video, "-c", "copy", seg_path,
            ], capture_output=True, check=True)
            print(f"  [SEED-FB] Shot {item['shot_number']}: {item['start_sec']:.1f}s-{item['end_sec']:.1f}s (original fallback)")

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
