"""Stage 1b: Scene detection using PySceneDetect.

Provides:
- detect_scene_boundaries(): Content-aware shot boundary detection
- detect_node_internal_cuts(): Internal cut detection for canvas node videos
- extract_keyframes(): Extract representative frames at cut points
"""
from __future__ import annotations

import os
import subprocess
from typing import List

from skills.timeline_plan.models import CutPoint, KeyFrame


def detect_scene_boundaries(
    video_path: str, threshold: float = 20.0
) -> List[CutPoint]:
    """Detect shot boundaries in a video using PySceneDetect.

    Uses ContentDetector (HSV color histogram differences) for cut detection.
    AI-generated videos typically need lower thresholds (15-22) than real footage (27).

    Args:
        video_path: Path to the video file.
        threshold: ContentDetector threshold. Lower = more sensitive.

    Returns:
        List of CutPoint objects sorted by time.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)

    scene_list = scene_manager.get_scene_list()
    cuts = []
    for i, (start_tc, _end_tc) in enumerate(scene_list):
        if i > 0:  # Skip the first scene's start (it's 0.0, not a cut)
            cuts.append(CutPoint(time_sec=start_tc.get_seconds()))
    return cuts


def detect_node_internal_cuts(
    video_path: str, threshold: float = 20.0
) -> List[CutPoint]:
    """Detect internal shot boundaries within a canvas node video.

    Useful for multi-shot canvas nodes where one node's prompt covers
    multiple ScriptShots.

    Args:
        video_path: Path to the canvas node video file (downloaded locally).
        threshold: ContentDetector threshold.

    Returns:
        List of CutPoint objects sorted by time.
    """
    return detect_scene_boundaries(video_path, threshold=threshold)


def extract_keyframes(
    video_path: str,
    cut_points: List[CutPoint],
    output_dir: str,
    shot_number: int = 0,
) -> List[KeyFrame]:
    """Extract keyframes from video at given cut point times.

    Uses ffmpeg to extract single frames. Output files named
    `keyframe_{shot_number}_{index:03d}.png`.

    Args:
        video_path: Path to the source video.
        cut_points: Cut times at which to extract frames.
        output_dir: Directory to save extracted frame images.
        shot_number: Shot number for filename prefix.

    Returns:
        List of KeyFrame objects with paths to extracted images.
    """
    os.makedirs(output_dir, exist_ok=True)

    duration = _probe_duration(video_path)

    keyframes: List[KeyFrame] = []
    for idx, cp in enumerate(cut_points):
        t = max(0.0, min(cp.time_sec, max(0.0, duration - 0.1)))
        out_path = os.path.join(output_dir, f"keyframe_{shot_number}_{idx:03d}.png")
        subprocess.run(
            [
                "ffmpeg", "-y", "-ss", f"{t:.3f}",
                "-i", video_path, "-frames:v", "1",
                "-q:v", "2", out_path,
            ],
            capture_output=True, check=True,
        )
        if os.path.exists(out_path):
            keyframes.append(KeyFrame(
                time_sec=cp.time_sec,
                image_path=out_path,
                shot_number=shot_number,
            ))
    return keyframes


def _probe_duration(video_path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    import json as _json
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    info = _json.loads(result.stdout)
    return float(info.get("format", {}).get("duration", 0.0))
