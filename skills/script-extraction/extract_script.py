#!/usr/bin/env python3
"""
Stage 1: Video Script Extraction

Wraps lingolens VideoScriptExtractor to extract structured screenplay from
AI-generated video + ASR transcription.

Usage:
  python3 skills/script-extraction/extract_script.py \\
    --video /path/to/video.mp4 \\
    --utterances /path/to/asr.json \\
    --output ep1_script.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# ── lingolens import ──────────────────────────────────────────────────────
LINGOLENS_ROOT = Path("~/workspace/lingolens").expanduser().resolve()
sys.path.insert(0, str(LINGOLENS_ROOT / "backend"))

try:
    from agents.script_extraction import VideoScriptExtractor
except ImportError as e:
    print(f"❌ 无法导入 lingolens VideoScriptExtractor: {e}")
    print(f"   请确认 {LINGOLENS_ROOT} 存在且 backend/ 可导入")
    sys.exit(1)


# ── Multimodal LLM (Doubao Seed) ──────────────────────────────────────────
# Uses lingolens' LLMServiceFactory — the same pattern as production code.
# Requires: DOUBAO_API_KEY in environment (from ~/workspace/lingolens/backend/.env)


def get_multimodal_llm():
    """Get a multimodal-capable LLM service for VideoScriptExtractor.

    Uses LLMServiceFactory.get("doubao-seed-2-0-pro-260215") which returns
    a ResponsesApiService instance. This is the same pattern used in:
    - backend/scripts/extract_scripts_batch.py (line 133)
    - backend/tasks/course_task_runners.py (line 241)
    """
    from services.llm.factory import LLMServiceFactory

    service = LLMServiceFactory.get("doubao-seed-2-0-pro-260215")
    if not service.enabled:
        raise RuntimeError(
            "Doubao multimodal LLM 未启用。请检查 DOUBAO_API_KEY 环境变量。\n"
            "配置来源: ~/workspace/lingolens/backend/.env"
        )
    return service


# ── Main ───────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 1: 从视频 + ASR 提取结构化剧本"
    )
    parser.add_argument("--video", required=True, help="原始视频文件路径")
    parser.add_argument("--utterances", required=True, help="ASR 转录 JSON 文件路径")
    parser.add_argument("--output", default="script_output.json", help="输出 JSON 路径")
    parser.add_argument("--temp-dir", default="runs", help="调试输出目录")
    return parser.parse_args()


def load_utterances(path: str) -> List[Dict[str, Any]]:
    """Load and validate ASR utterances JSON."""
    with open(path, "r", encoding="utf-8") as f:
        utterances = json.load(f)

    if not isinstance(utterances, list):
        sys.exit(f"❌ utterances 文件必须是 JSON 数组，当前类型: {type(utterances)}")

    required = {"speaker", "start_time", "end_time", "text"}
    for i, u in enumerate(utterances):
        missing = required - set(u.keys())
        if missing:
            sys.exit(f"❌ utterances[{i}] 缺少必填字段: {missing}")

    return utterances


def get_video_duration(video_path: str) -> float:
    """Get video duration using ffprobe or fallback."""
    import subprocess

    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries",
                "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        print("⚠️  ffprobe 不可用，从 ASR 数据推算时长")
        # Fallback: use max end_time from utterances
        utterances = load_utterances(
            sys.argv[sys.argv.index("--utterances") + 1]
            if "--utterances" in sys.argv else ""
        )
        if utterances:
            return max(u["end_time"] for u in utterances) / 1000.0
        return 0.0


async def main() -> None:
    args = parse_args()

    # 1. Validate inputs
    video_path = str(Path(args.video).resolve())
    if not Path(video_path).exists():
        sys.exit(f"❌ 视频文件不存在: {video_path}")

    utterances_path = str(Path(args.utterances).resolve())
    if not Path(utterances_path).exists():
        sys.exit(f"❌ ASR 文件不存在: {utterances_path}")

    utterances = load_utterances(utterances_path)
    print(f"📖 加载 {len(utterances)} 条 ASR 转录")

    duration = get_video_duration(args.video)
    print(f"⏱️  视频时长: {duration:.1f}s")

    # 2. Initialize extractor with Doubao Seed multimodal LLM
    llm = get_multimodal_llm()

    extractor = VideoScriptExtractor(multimodal_llm=llm)

    # 3. Extract
    output_path = str(Path(args.output).resolve())
    temp_dir = args.temp_dir
    Path(temp_dir).mkdir(parents=True, exist_ok=True)

    print(f"🎬 开始剧本提取...")
    try:
        result = await extractor.extract(
            video_path=video_path,
            utterances=utterances,
            duration_seconds=duration,
            temp_dir=temp_dir,
        )
    except RuntimeError as e:
        sys.exit(f"❌ 提取失败: {e}")

    # 4. Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

    size_kb = Path(output_path).stat().st_size // 1024
    print(f"✅ 剧本已生成: {output_path} ({size_kb}KB)")
    print(f"   {len(result.script.shots)} 个镜头, {result.meta.summary[:60]}...")


if __name__ == "__main__":
    asyncio.run(main())
