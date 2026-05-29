"""Tests for scene detection module."""
import os
import subprocess
import tempfile
from pathlib import Path
from skills.scene_detection.detect_scenes import (
    detect_scene_boundaries, extract_keyframes, CutPoint, KeyFrame
)


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


class TestExtractKeyframes:
    def test_extract_keyframes(self, tmp_path):
        """Extract keyframes from a simple video at given cut points."""
        video = tmp_path / "test.mp4"
        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=red:s=320x240:d=3 "
            f"-c:v libx264 -pix_fmt yuv420p {video}",
            shell=True, capture_output=True, check=True,
        )
        cut_points = [CutPoint(time_sec=1.0), CutPoint(time_sec=2.0)]
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        keyframes = extract_keyframes(
            str(video), cut_points, str(output_dir), shot_number=1
        )
        assert len(keyframes) == 2
        for kf in keyframes:
            assert isinstance(kf, KeyFrame)
            assert kf.shot_number == 1
            assert os.path.exists(kf.image_path), f"Missing: {kf.image_path}"

    def test_extract_keyframes_clamps_oob_times(self, tmp_path):
        """Out-of-bounds cut times are clamped to valid range."""
        video = tmp_path / "test_oob.mp4"
        subprocess.run(
            f"ffmpeg -y -f lavfi -i color=c=red:s=320x240:d=1 "
            f"-c:v libx264 -pix_fmt yuv420p {video}",
            shell=True, capture_output=True, check=True,
        )
        cut_points = [CutPoint(time_sec=-1.0), CutPoint(time_sec=10.0)]
        output_dir = tmp_path / "frames"
        output_dir.mkdir()
        keyframes = extract_keyframes(
            str(video), cut_points, str(output_dir), shot_number=1
        )
        assert len(keyframes) == 2
