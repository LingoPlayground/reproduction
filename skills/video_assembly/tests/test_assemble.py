"""Tests for video assembly module."""
import os
from skills.timeline_plan.models import normalize_seedance_duration
from skills.video_assembly.assemble import (
    normalize_segment_encoding,
    _write_concat_file,
)


class TestNormalizeSeedanceDuration:
    def test_smart_duration_default(self):
        """Shots >= MIN_SEEDANCE_DURATION (4s) return -1 for seedance smart-duration mode."""
        assert normalize_seedance_duration(4.0) == -1
        assert normalize_seedance_duration(5.2) == -1
        assert normalize_seedance_duration(12.3) == -1
        assert normalize_seedance_duration(30.0) == -1
        assert normalize_seedance_duration(100.0) == -1

    def test_short_shot_fallback(self):
        """Shots < MIN_SEEDANCE_DURATION (4s) use clamped explicit duration (min 4s)."""
        assert normalize_seedance_duration(0.5) == 4
        assert normalize_seedance_duration(1.0) == 4
        assert normalize_seedance_duration(3.7) == 4

    def test_edge_cases(self):
        assert normalize_seedance_duration(4.0) == -1
        assert normalize_seedance_duration(3.9) == 4
        assert normalize_seedance_duration(0.0) == 4


class TestNormalizeSegmentEncoding:
    def test_command_runs_without_error(self, tmp_path):
        test_video = tmp_path / "test_input.mp4"
        os.system(
            f"ffmpeg -y -f lavfi -i color=c=black:s=32x32:d=0.5 "
            f"-c:v libx264 -pix_fmt yuv420p {test_video} 2>/dev/null"
        )
        out_video = tmp_path / "test_output.mp4"
        normalize_segment_encoding(str(test_video), str(out_video))
        assert out_video.exists()
        assert out_video.stat().st_size > 0

    def test_idempotent(self, tmp_path):
        test_video = tmp_path / "double.mp4"
        os.system(
            f"ffmpeg -y -f lavfi -i color=c=black:s=32x32:d=0.5 "
            f"-c:v libx264 {test_video} 2>/dev/null"
        )
        mid = tmp_path / "mid.mp4"
        final = tmp_path / "final.mp4"
        normalize_segment_encoding(str(test_video), str(mid))
        normalize_segment_encoding(str(mid), str(final))
        assert final.exists()


class TestWriteConcatFile:
    def test_writes_file_list(self, tmp_path):
        paths = ["/tmp/a.mp4", "/tmp/b.mp4", "/tmp/c.mp4"]
        concat_path = tmp_path / "concat.txt"
        _write_concat_file(paths, str(concat_path))
        content = concat_path.read_text()
        assert "file '/tmp/a.mp4'" in content
        assert "file '/tmp/b.mp4'" in content
        assert "file '/tmp/c.mp4'" in content
