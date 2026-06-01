"""Tests for planner_models.py — two-stage API."""
from skills.timeline_plan.planner_models import (
    MatchResult, NodeGeneration, LineNodeMatch, SourceTimeRange,
    UnmatchedLine, RewriteInput,
)


class TestMatchResult:
    def test_empty(self):
        r = MatchResult()
        assert r.node_generations == []
        assert r.unmatched_lines == []

    def test_with_generations(self):
        r = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           matched_node_ids=["n1"],
                           source_time_range=SourceTimeRange(10.0, 15.0),
                           line_matches=[
                               LineNodeMatch(line_id="L1", original_line="hello",
                                             rewritten_line="hi", node_id="n1",
                                             match_reasoning="verbatim", confidence=0.95)
                           ],
                           grouping_reasoning="single line", confidence=0.94)
        ])
        assert len(r.node_generations) == 1
        g = r.node_generations[0]
        assert g.source_time_range.start_sec == 10.0
        assert g.line_matches[0].confidence == 0.95
        assert not g.has_prompt  # No rewritten_prompt yet

    def test_with_unmatched(self):
        r = MatchResult(unmatched_lines=[
            UnmatchedLine(line_id="L99", reason="no match", original="hello", rewritten="hi")
        ])
        assert len(r.unmatched_lines) == 1

    def test_has_prompt_false_by_default(self):
        g = NodeGeneration(group_id="G1")
        assert not g.has_prompt

    def test_has_prompt_true_after_rewrite(self):
        g = NodeGeneration(group_id="G1", rewritten_prompt="new prompt text")
        assert g.has_prompt


class TestRewriteInput:
    def test_basic(self):
        ri = RewriteInput(group_id="G1", original_prompt="original", covered_lines=[
            {"line_id": "L1", "original": "hello", "rewritten": "hi", "speaker": "S"}
        ])
        assert ri.group_id == "G1"
        assert len(ri.covered_lines) == 1


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestMatchResult, TestRewriteInput]:
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
