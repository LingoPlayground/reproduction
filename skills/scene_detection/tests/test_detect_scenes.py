"""Tests for scene detection module."""
import os
import subprocess
import tempfile
from pathlib import Path
from skills.timeline_plan.models import CutPoint
from skills.scene_detection.detect_scenes import detect_scene_boundaries


class TestDetectSceneBoundaries:
    def test_empty_video_returns_empty(self):
        """Non-existent video path raises FileNotFoundError."""
        import pytest
        with pytest.raises(FileNotFoundError):
            detect_scene_boundaries("/nonexistent/video.mp4")

    def test_short_video_single_scene(self, tmp_path):
        """A very short video produces minimal cut points."""
        test_video = tmp_path / "test_2s.mp4"
        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=2 "
            f"-c:v libx264 -pix_fmt yuv420p {test_video}",
            shell=True, capture_output=True, check=True,
        )
        cuts = detect_scene_boundaries(str(test_video))
        assert len(cuts) <= 2
        if cuts:
            assert all(isinstance(c, CutPoint) for c in cuts)

    def test_two_scene_video_detects_cut(self, tmp_path):
        """A video with distinct scene change should produce at least one cut."""
        test_video = tmp_path / "two_scene.mp4"
        black_video = tmp_path / "black.mp4"
        white_video = tmp_path / "white.mp4"
        concat_file = tmp_path / "concat.txt"

        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=2 "
            f"-c:v libx264 -pix_fmt yuv420p {black_video}",
            shell=True, capture_output=True, check=True,
        )
        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=white:s=320x240:d=2 "
            f"-c:v libx264 -pix_fmt yuv420p {white_video}",
            shell=True, capture_output=True, check=True,
        )
        concat_file.write_text(
            f"file '{black_video}'\nfile '{white_video}'\n"
        )
        subprocess.run(
            f"ffmpeg -y -f concat -safe 0 -i {concat_file} "
            f"-c copy {test_video}",
            shell=True, capture_output=True, check=True,
        )
        assert test_video.exists(), f"Test video not created: {test_video}"
        cuts = detect_scene_boundaries(str(test_video))
        assert len(cuts) >= 1, f"Expected at least 1 cut, got {len(cuts)}"
        cut_times = [c.time_sec for c in cuts]
        near_2s = [t for t in cut_times if 1.5 <= t <= 2.5]
        assert len(near_2s) >= 1, f"No cut near 2.0s: {cut_times}"

    def test_content_detector_threshold_configurable(self, tmp_path):
        """ContentDetector accepts configurable threshold."""
        video_path = tmp_path / "test_thresh.mp4"
        black_video = tmp_path / "black_thresh.mp4"
        white_video = tmp_path / "white_thresh.mp4"
        concat_file = tmp_path / "concat_thresh.txt"

        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=2 "
            f"-c:v libx264 -pix_fmt yuv420p {black_video}",
            shell=True, capture_output=True, check=True,
        )
        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=white:s=320x240:d=2 "
            f"-c:v libx264 -pix_fmt yuv420p {white_video}",
            shell=True, capture_output=True, check=True,
        )
        concat_file.write_text(
            f"file '{black_video}'\nfile '{white_video}'\n"
        )
        subprocess.run(
            f"ffmpeg -y -f concat -safe 0 -i {concat_file} "
            f"-c copy {video_path}",
            shell=True, capture_output=True, check=True,
        )

        cuts_low = detect_scene_boundaries(str(video_path), threshold=10.0)
        cuts_high = detect_scene_boundaries(str(video_path), threshold=30.0)
        # Lower threshold = more sensitive = potentially more cuts
        assert len(cuts_low) >= len(cuts_high) - 1

