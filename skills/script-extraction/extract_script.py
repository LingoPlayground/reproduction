#!/usr/bin/env python3
"""
Stage 1: Video Script Extraction

Extracts structured screenplay from video using lingolens VideoScriptExtractor.
Auto-generates ASR via Azure Speech, then runs multimodal LLM analysis.

Usage:
  python3 skills/script-extraction/extract_script.py \\
    --video /path/to/video.mp4 \\
    --output ep1_script.json
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

# ── lingolens setup ───────────────────────────────────────────────────────
LINGOLENS_ROOT = Path("~/workspace/lingolens").expanduser().resolve()

for env_path in [
    LINGOLENS_ROOT / "backend" / ".env",
    Path("~/workspace/shakespeare/.env").expanduser(),
]:
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

sys.path.insert(0, str(LINGOLENS_ROOT))
sys.path.insert(0, str(LINGOLENS_ROOT / "backend"))

try:
    from agents.script_extraction import VideoScriptExtractor
except ImportError as e:
    print(f"❌ Cannot import lingolens VideoScriptExtractor: {e}")
    sys.exit(1)

from skills.scene_detection.detect_scenes import detect_scene_boundaries


def get_multimodal_llm():
    from services.llm.factory import LLMServiceFactory
    service = LLMServiceFactory.get("doubao-seed-2-0-pro-260215")
    if not service.enabled:
        raise RuntimeError("Doubao multimodal LLM is not enabled")
    return service


async def auto_asr(video_path: str) -> list[dict[str, Any]]:
    """Extract audio from video and transcribe via Azure ASR."""
    from services.asr_service import ASRService
    from services.media_service import MediaService

    print("🎵 Extracting audio...")
    audio_path = str(Path(tempfile.gettempdir()) / f"asr_audio_{os.getpid()}.wav")
    try:
        MediaService.extract_audio_wav(video_path, audio_path)

        print("🎙️  Azure ASR transcribing...")
        asr = ASRService()
        asr_result = await asr.transcribe(audio_path)
        if not asr_result.get("success"):
            raise RuntimeError(f"ASR failed: {asr_result}")

        utterances = asr.extract_utterances(asr_result)
        print(f"   {len(utterances)} utterances")
        return utterances
    finally:
        Path(audio_path).unlink(missing_ok=True)


def get_video_duration(video_path: str, utterances: list[dict[str, Any]]) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
             video_path],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        if utterances:
            return max(u.get("end_time", 0) for u in utterances) / 1000.0
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: Extract structured screenplay from video (auto ASR)")
    parser.add_argument("--video", required=True, help="Raw video file path")
    parser.add_argument("--output", default="script_output.json", help="Output JSON path")
    parser.add_argument("--temp-dir", default="runs", help="Debug output directory")
    return parser.parse_args()


async def main() -> None:
    os.chdir(str(LINGOLENS_ROOT))
    args = parse_args()

    video_path = str(Path(args.video).resolve())
    if not Path(video_path).exists():
        sys.exit(f"❌ Video file not found: {video_path}")

    utterances = await auto_asr(video_path)

    duration = get_video_duration(video_path, utterances)
    print(f"⏱️  Video duration: {duration:.1f}s")

    print("🔍 Detecting scene cuts...")
    scene_cuts = detect_scene_boundaries(video_path)
    cut_times = [c.time_sec for c in scene_cuts]
    print(f"   {len(cut_times)} cuts: {[f'{t:.1f}s' for t in cut_times[:10]]}{'...' if len(cut_times) > 10 else ''}")

    llm = get_multimodal_llm()
    extractor = VideoScriptExtractor(multimodal_llm=llm)

    output_path = str(Path(args.output).resolve())
    temp_dir = args.temp_dir
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    print(f"🎬 Starting script extraction...")
    try:
        result = await extractor.extract(
            video_path=video_path,
            utterances=utterances,
            duration_seconds=duration,
            temp_dir=temp_dir,
            scene_cut_times=cut_times,
        )
    except TypeError:
        # lingolens extractor does not support scene_cut_times yet — fall back
        print("⚠️  lingolens does not support scene_cut_times yet, skipping cut injection")
        result = await extractor.extract(
            video_path=video_path,
            utterances=utterances,
            duration_seconds=duration,
            temp_dir=temp_dir,
        )
    except RuntimeError as e:
        sys.exit(f"❌ Extraction failed: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    size_kb = Path(output_path).stat().st_size // 1024
    print(f"✅ Script generated: {output_path} ({size_kb}KB)")
    print(f"   {len(result.script.shots)} shots, {result.meta.summary[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())