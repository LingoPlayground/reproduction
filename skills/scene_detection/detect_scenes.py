"""Stage 1b: Scene detection using PySceneDetect.

Provides:
- detect_scene_boundaries(): Content-aware shot boundary detection
- detect_node_internal_cuts(): Internal cut detection for canvas node videos
- extract_keyframes(): Extract representative frames at cut points
"""
from __future__ import annotations

import os
from typing import List

from skills.timeline_plan.models import CutPoint


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
