#!/usr/bin/env python3
"""Stage 4: Video assembly from TimelinePlan."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time

import urllib.request
import urllib.error

from pathlib import Path


WORK_DIR = Path(__file__).resolve().parents[2] / "generated"

# ── seedance import (lazy, only when lingolens is available) ──────

def _ensure_seedance_import():
    """Lazy-import seedance client with sys.path setup."""
    sys.path.insert(0, str(Path("~/workspace/lingolens/backend").expanduser()))
    try:
        from utils.aqinfo_seedance import AQInfoSeedanceClient, SeedanceModel, AssetType, SeedanceRatio, SeedanceResolution
        return AQInfoSeedanceClient, SeedanceModel, AssetType, SeedanceRatio, SeedanceResolution
    except ImportError:
        return None, None, None, None, None

# Lazy seedance client — initialized on first use, after env is loaded
_seedance_client = None
_seedance_available_cache = None


def _get_seedance():
    """Lazy seedance init. Returns (client, Model, AssetType, Ratio, Resolution, is_available).
    
    Calls load_pipeline_env() internally to ensure env is loaded.
    Import failures (lingolens not installed) are cached permanently.
    Credential/env failures are retried on each call.
    """
    global _seedance_client, _seedance_available_cache

    if _seedance_available_cache is False:
        return None, None, None, None, None, False
    
    # Already initialized successfully
    if _seedance_client is not None:
        Client, Model, AssetType, Ratio, Resolution = _ensure_seedance_import()
        return _seedance_client, Model, AssetType, Ratio, Resolution, True

    Client, Model, AssetType, Ratio, Resolution = _ensure_seedance_import()
    if Client is None:
        _seedance_available_cache = False
        return None, None, None, None, None, False

    try:
        _seedance_client = Client()
        _seedance_available_cache = True
        return _seedance_client, Model, AssetType, Ratio, Resolution, True
    except (ValueError, Exception) as e:
        # Do NOT cache credential failures — env may be loaded later
        print(f"  [WARN] Seedance unavailable: {e}")
        return None, None, None, None, None, False

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


def _write_concat_file(segment_paths: list[str], concat_path: str) -> str:
    """Write ffmpeg concat file listing all segments (absolute paths)."""
    with open(concat_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{os.path.abspath(p)}'\n")
    return concat_path


def _probe_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", video_path],
        capture_output=True, text=True, check=True,
    )
    import json as _json
    info = _json.loads(result.stdout)
    return float(info.get("format", {}).get("duration", 0.0))


def _extract_asset_id(asset_result: dict) -> str | None:
    data = asset_result.get("data", {})
    if isinstance(data, dict):
        raw = data.get("id")
        if raw:
            return str(raw)
    return None


def _validate_external_url(url: str) -> None:
    from urllib.parse import urlparse
    import ipaddress, socket
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http/https URLs allowed, got: {parsed.scheme}")
    if not parsed.hostname:
        raise ValueError(f"URL must include a hostname: {url}")
    hostname = parsed.hostname.lower()
    if hostname in ("localhost", "127.0.0.1", "::1"):
        raise ValueError(f"Blocked hostname: {hostname}")
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise ValueError(f"Could not resolve hostname: {hostname}") from e
    for info in addrs:
        ip_str = info[4][0]
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                raise ValueError(f"Blocked IP: {ip_str}")
        except ValueError:
            continue


async def _generate_via_seedance(item: dict, duration: int) -> str:
    """Generate a video via seedance for a single TimelinePlanItem.

    Registers reference images as assets (bypasses face detection),
    then generates video using asset:// URLs.
    """
    client, Model, AssetType, Ratio, Resolution, ok = _get_seedance()
    if not ok:
        return ""

    prompt = item.get("rewritten_prompt", "")
    ref_images = item.get("ref_images", [])

    if not prompt or not ref_images:
        return ""

    shot_num = item.get("shot_number", 0)
    name = f"shot_{shot_num}"

    image_urls = [u if isinstance(u, str) else u.get("url", "") for u in ref_images]
    image_urls = [u for u in image_urls if u]

    asset_ids = []
    for i, url in enumerate(image_urls):
        try:
            result = await client.create_asset(url=url, asset_type=AssetType.IMAGE, name=f"{name}_ref_{i}")
            aid = _extract_asset_id(result)
            if aid:
                await client.wait_for_asset(aid, max_wait_time=120)
                asset_ids.append(aid)
        except Exception as e:
            print(f"    [{name}] ⚠️  Asset creation failed for ref {i}: {e}")

    if not asset_ids:
        return ""

    asset_urls = [f"asset://{aid}" for aid in asset_ids]
    duration_label = f"{duration}s" if duration > 0 else "auto"
    print(f"    [{name}] Generating (seedance fast, {duration_label}, {len(asset_urls)} refs)...")
    try:
        kwargs = dict(prompt=prompt, images=asset_urls, model=Model.SEEDANCE_2_0_FAST,
                      ratio=Ratio.RATIO_9_16, resolution=Resolution.RESOLUTION_720P,
                      generate_audio=True, wait=True, max_wait_time=900)
        if duration > 0:
            kwargs["duration"] = duration
        result = await client.multimodal_reference_to_video(**kwargs)
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
    _validate_external_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                content_length = resp.headers.get("Content-Length")
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > 500:
                        raise ValueError(f"Video too large: {size_mb:.0f}MB (max 500MB)")
                MAX_VIDEO_BYTES = 500 * 1024 * 1024
                total = 0
                with open(path, "wb") as f:
                    while chunk := resp.read(1048576):  # 1MB chunks
                        total += len(chunk)
                        if total > MAX_VIDEO_BYTES:
                            raise ValueError(f"Video download exceeded {MAX_VIDEO_BYTES} bytes")
                        f.write(chunk)
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
) -> dict:
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

    segment_paths: list[str] = []
    fallback_report: list[dict] = []

    for idx, item in enumerate(items):
        source = item.get("source", "original")
        seg_path = str(work_dir / f"seg_{idx:03d}_shot{item['shot_number']}.mp4")

        if source == "original" or skip_seedance:
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", f"{item['start_sec']:.3f}",
                "-i", original_video,
                "-t", f"{item['end_sec'] - item['start_sec']:.3f}",
                "-c:v", "libx264", "-c:a", "aac",
                seg_path,
            ], capture_output=True, check=True)
            print(f"  [ORIG] Shot {item['shot_number']}: {item['start_sec']:.1f}s-{item['end_sec']:.1f}s")
        elif source == "modified":
            planned_duration = item["end_sec"] - item["start_sec"]
            duration = max(4, int(planned_duration))
            video_url = await _generate_via_seedance(item, duration)
            if video_url and _download_video(video_url, Path(seg_path)):
                actual = _probe_duration(seg_path)
                if actual > 0 and actual < planned_duration - 0.5:
                    print(f"  [SEED-FB] Shot {item['shot_number']}: seedance output {actual:.1f}s "
                          f"too short (planned {planned_duration:.1f}s) → original fallback")
                    fallback_report.append({
                        "segment_id": item.get("shot_id", f"mod_{idx}"),
                        "planned_source": "modified",
                        "actual_source": "original_fallback",
                        "reason": "seedance_output_too_short",
                        "affected_line_ids": item.get("covered_line_ids", []),
                    })
                    # Replace the short seedance with original fallback
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-ss", f"{item['start_sec']:.3f}",
                        "-i", original_video,
                        "-t", f"{item['end_sec'] - item['start_sec']:.3f}",
                        "-c:v", "libx264", "-c:a", "aac",
                        seg_path,
                    ], capture_output=True, check=True)
                    actual = _probe_duration(seg_path)
                    print(f"  [SEED-FB] Shot {item['shot_number']}: fallback {actual:.1f}s")
                if actual > 0 and actual > planned_duration + 0.3:
                    trimmed = str(work_dir / f"seg_{idx:03d}_trimmed.mp4")
                    subprocess.run([
                        "ffmpeg", "-y", "-i", seg_path,
                        "-t", f"{planned_duration:.3f}",
                        "-c:v", "libx264", "-c:a", "aac",
                        "-preset", "ultrafast",
                        trimmed,
                    ], capture_output=True, check=True)
                    os.replace(trimmed, seg_path)
                print(f"  [SEED] Shot {item['shot_number']}: seedance {actual:.1f}s (planned {planned_duration:.1f}s)")
            else:
                # Fallback to original segment
                try:
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-ss", f"{item['start_sec']:.3f}",
                        "-i", original_video,
                        "-t", f"{item['end_sec'] - item['start_sec']:.3f}",
                        "-c:v", "libx264", "-c:a", "aac",
                        seg_path,
                    ], capture_output=True, check=True)
                    print(f"  [SEED-FB] Shot {item['shot_number']}: seedance failed → original fallback")
                    fallback_report.append({
                        "segment_id": item.get("shot_id", f"mod_{idx}"),
                        "planned_source": "modified",
                        "actual_source": "original_fallback",
                        "reason": "seedance_api_failed",
                        "affected_line_ids": item.get("covered_line_ids", []),
                    })
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Shot {item['shot_number']}: seedance + fallback both failed. "
                        f"ffmpeg stderr: {e.stderr.decode()[:200] if e.stderr else 'unknown'}"
                    ) from e

        if os.path.exists(seg_path) and os.path.getsize(seg_path) > 0:
            segment_paths.append(seg_path)

    # ── Integrity check: every plan item must produce a segment ──
    planned_modified = sum(1 for item in items if item.get("source") == "modified" and not skip_seedance)
    planned_original = sum(1 for item in items if item.get("source") != "modified" or skip_seedance)
    planned_total = planned_modified + planned_original
    if len(segment_paths) != planned_total:
        missing = planned_total - len(segment_paths)
        raise RuntimeError(
            f"Segment count mismatch: {len(segment_paths)} produced, "
            f"{planned_total} planned ({missing} missing). Aborting."
        )

    if not segment_paths:
        raise RuntimeError("No valid segments produced")

    # Normalize encoding + audio
    print(f"\n  Normalizing {len(segment_paths)} segments...")
    normalized_paths: list[str] = []
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
    first_stderr = ""
    r = subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", concat_file, "-c", "copy", output_path,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        first_stderr = r.stderr[:300]
        r = subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", concat_file, "-c:v", "libx264", "-c:a", "aac", output_path,
        ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"Concat failed (copy mode: {first_stderr[:100]}...; "
            f"re-encode mode: {r.stderr[:300]})"
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  Final: {output_path} ({size_mb:.1f}MB)")

    # ── Duration integrity check ──
    planned_total_duration = sum(item["end_sec"] - item["start_sec"] for item in items)
    actual_duration = _probe_duration(output_path)
    drift = actual_duration - planned_total_duration
    if abs(drift) > 2.0:
        raise RuntimeError(
            f"Duration drift too large: planned {planned_total_duration:.1f}s, "
            f"actual {actual_duration:.1f}s, drift {drift:+.1f}s"
        )
    print(f"  Duration OK: {actual_duration:.1f}s (planned {planned_total_duration:.1f}s)")

    # Clean up intermediate segments
    shutil.rmtree(work_dir, ignore_errors=True)

    # Write execution report
    report_path = str(out_dir / "execution_report.json")
    report = {
        "total_items": len(items),
        "planned_modified": planned_modified,
        "actual_modified": planned_modified - len(fallback_report),
        "fallback_count": len(fallback_report),
        "fallbacks": fallback_report,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if fallback_report:
        affected = set()
        for fb in fallback_report:
            affected.update(fb.get("affected_line_ids", []))
        print(f"  ⚠️  {len(fallback_report)} segments fell back to original "
              f"({len(affected)} line(s) affected) → {report_path}")

    return {"output_path": output_path, "report": report}


async def main():
    p = argparse.ArgumentParser(description="Stage 4: Assemble final video")
    p.add_argument("--plan", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--skip-seedance", action="store_true")
    args = p.parse_args()
    result = await assemble_video(args.plan, args.video, args.output, args.skip_seedance)
    print(f"Output: {result['output_path']}")
    if result["report"]["fallback_count"] > 0:
        print(f"Warning: {result['report']['fallback_count']} segments used original fallback")

if __name__ == "__main__":
    asyncio.run(main())
