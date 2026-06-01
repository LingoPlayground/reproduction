"""Tests for timeline_normalizer.py — two-stage MatchResult input."""
from skills.timeline_plan.models import CanvasNode, CutPoint, TimelinePlanItem
from skills.timeline_plan.planner_models import (
    MatchResult, NodeGeneration, LineNodeMatch, SourceTimeRange, UnmatchedLine,
)
from skills.timeline_plan.timeline_normalizer import normalize_plan, _carve_out


def make_shot(num, start=0.0, end=10.0):
    class S:
        def __init__(self):
            self.shot_number = num; self.start_seconds = start
            self.end_seconds = end; self.scene_description = f"s{num}"
            self.lines = []
    return S()


class TestNormalize:
    def test_basic(self):
        draft = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           matched_node_ids=["n1"],
                           source_time_range=SourceTimeRange(2, 6),
                           rewritten_prompt="test",
                           line_matches=[
                               LineNodeMatch(line_id="L1", original_line="a", rewritten_line="b",
                                             node_id="n1", match_reasoning="v", confidence=0.9)
                           ])
        ])
        plan = normalize_plan(draft, [make_shot(1, 0, 15)], [], [], 15)
        mod = [i for i in plan.items if i.source == "modified"]
        assert len(mod) == 1
        assert mod[0].rewritten_prompt == "test"

    def test_short_padded(self):
        draft = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           matched_node_ids=["n1"],
                           source_time_range=SourceTimeRange(5, 6),
                           rewritten_prompt="test",
                           line_matches=[
                               LineNodeMatch(line_id="L1", original_line="a", rewritten_line="b",
                                             node_id="n1", match_reasoning="v", confidence=0.9)
                           ])
        ])
        plan = normalize_plan(draft, [make_shot(1, 0, 20)], [], [], 20)
        mod = [i for i in plan.items if i.source == "modified"]
        assert mod[0].end_sec - mod[0].start_sec >= 4.0

    def test_unmatched_raises(self):
        draft = MatchResult(unmatched_lines=[
            UnmatchedLine(line_id="L1", reason="no match", original="a", rewritten="b")
        ])
        try:
            normalize_plan(draft, [make_shot(1, 0, 15)], [], [], 15)
            assert False, "should raise"
        except ValueError as e:
            assert "unmatched" in str(e).lower()

    def test_overlap_raises(self):
        draft = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           source_time_range=SourceTimeRange(0, 5), rewritten_prompt="a",
                           line_matches=[
                               LineNodeMatch(line_id="L1", original_line="x", rewritten_line="y",
                                             node_id="n1", match_reasoning="v", confidence=0.9)
                           ]),
            NodeGeneration(group_id="G2", covered_line_ids=["L2"],
                           source_time_range=SourceTimeRange(3, 8), rewritten_prompt="b",
                           line_matches=[
                               LineNodeMatch(line_id="L2", original_line="x", rewritten_line="y",
                                             node_id="n2", match_reasoning="v", confidence=0.9)
                           ]),
        ])
        try:
            normalize_plan(draft, [make_shot(1, 0, 15)], [], [], 15)
            assert False, "should raise"
        except ValueError as e:
            assert "overlap" in str(e).lower()


class TestCarveOut:
    def test_middle(self): assert _carve_out([(0,10)], 3, 7) == [(0,3),(7,10)]
    def test_start(self): assert _carve_out([(0,10)], 0, 4) == [(4,10)]
    def test_whole(self): assert _carve_out([(0,5)], 0, 5) == []
    def test_no_overlap(self): assert _carve_out([(0,5)], 10, 15) == [(0,5)]


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestNormalize, TestCarveOut]:
        t = cls()
        for name in sorted(dir(t)):
            if name.startswith("test_"):
                try:
                    getattr(t, name)()
                    print(f"  PASS: {cls.__name__}.{name}")
                except AssertionError as e:
                    print(f"  FAIL: {cls.__name__}.{name} - {e}")
                    failed += 1
    sys.exit(failed)
