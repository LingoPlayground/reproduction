"""Tests for cut point fusion algorithm."""
from skills.timeline_plan.models import CutPoint
from skills.timeline_plan.cut_fusion import (
    find_nearest_cut, fuse_cut_boundary, determine_cut_points,
)


class FakeShot:
    def __init__(self, start, end):
        self.start_seconds = start
        self.end_seconds = end


class TestFindNearestCut:
    def test_exact_match(self):
        cuts = [CutPoint(5.0), CutPoint(10.0)]
        result = find_nearest_cut(cuts, 5.0, tolerance=0.5)
        assert result is not None
        assert result.time_sec == 5.0

    def test_within_tolerance(self):
        cuts = [CutPoint(5.2)]
        result = find_nearest_cut(cuts, 5.0, tolerance=0.5)
        assert result is not None
        assert result.time_sec == 5.2

    def test_outside_tolerance(self):
        cuts = [CutPoint(6.0)]
        result = find_nearest_cut(cuts, 5.0, tolerance=0.5)
        assert result is None

    def test_picks_closest(self):
        cuts = [CutPoint(4.8), CutPoint(5.3)]
        result = find_nearest_cut(cuts, 5.0, tolerance=0.5)
        assert result is not None
        assert result.time_sec == 4.8

    def test_empty_cuts(self):
        result = find_nearest_cut([], 5.0, tolerance=0.5)
        assert result is None


class TestFuseCutBoundary:
    def test_llm_only_no_nearby_cut(self):
        shot = FakeShot(10.0, 20.0)
        cuts = [CutPoint(5.0), CutPoint(25.0)]
        start, end = fuse_cut_boundary(shot, cuts, tolerance=0.5)
        assert start == 10.0
        assert end == 20.0

    def test_scenedetect_refines_both(self):
        shot = FakeShot(10.0, 20.0)
        cuts = [CutPoint(10.1), CutPoint(19.8)]
        start, end = fuse_cut_boundary(shot, cuts, tolerance=0.5)
        assert start == 10.1
        assert end == 19.8


class TestDetermineCutPoints:
    def test_basic_flow(self):
        shots = [FakeShot(0.0, 10.0), FakeShot(10.0, 20.0)]
        cuts = [CutPoint(10.2)]
        results = determine_cut_points(shots, cuts, video_duration=20.0)
        assert len(results) == 2
        assert results[1][0] == 10.2

    def test_clamp_negative_start(self):
        shots = [FakeShot(-5.0, 10.0)]
        results = determine_cut_points(shots, [], video_duration=20.0)
        assert results[0][0] == 0.0

    def test_clamp_over_duration_end(self):
        shots = [FakeShot(5.0, 25.0)]
        results = determine_cut_points(shots, [], video_duration=20.0)
        assert results[0][1] == 20.0

    def test_minimum_duration_1s(self):
        shots = [FakeShot(10.0, 9.0)]
        results = determine_cut_points(shots, [], video_duration=20.0)
        assert results[0][1] - results[0][0] >= 1.0

    def test_gap_filling(self):
        shots = [FakeShot(0.0, 5.0), FakeShot(10.0, 15.0)]
        cuts = [CutPoint(5.0), CutPoint(7.5), CutPoint(10.0)]
        results = determine_cut_points(shots, cuts, video_duration=20.0)
        assert results[0][1] <= results[1][0]
