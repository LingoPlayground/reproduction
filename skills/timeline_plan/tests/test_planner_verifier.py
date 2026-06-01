"""Tests for planner_verifier.py — two-stage verify_match + verify_prompt."""
from skills.timeline_plan.planner_models import (
    MatchResult, NodeGeneration, LineNodeMatch, SourceTimeRange, UnmatchedLine,
)
from skills.timeline_plan.planner_verifier import verify_match, verify_prompt, _loose_match


def make_evidence(line_ids):
    return {
        "rewrite_lines": [
            {"line_id": lid, "original": f"o_{lid}", "rewritten": f"r_{lid}",
             "speaker": "S", "start_sec": float(i*3), "end_sec": float(i*3+2)}
            for i, lid in enumerate(line_ids)
        ],
        "canvas_nodes": [{"node_id": f"n{i}", "prompt": f"p{i}"} for i in range(3)],
        "neighbor_lines": [],
    }

def m(line_id):
    return LineNodeMatch(line_id=line_id, original_line=f"o_{line_id}",
                         rewritten_line=f"r_{line_id}", node_id="n0",
                         match_reasoning="v", confidence=0.9)


class TestVerifyMatch:
    def test_all_covered_passes(self):
        ev = make_evidence(["L1", "L2"])
        r = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1", "L2"],
                           source_time_range=SourceTimeRange(0, 5),
                           line_matches=[m("L1"), m("L2")])
        ])
        assert verify_match(r, ev) == []

    def test_missing_caught(self):
        ev = make_evidence(["L1", "L2"])
        r = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           source_time_range=SourceTimeRange(0, 2),
                           line_matches=[m("L1")])
        ])
        assert any("L2" in e for e in verify_match(r, ev))

    def test_unknown_id_caught(self):
        ev = make_evidence(["L1"])
        r = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L_FAKE"],
                           source_time_range=SourceTimeRange(0, 4),
                           line_matches=[m("L_FAKE")])
        ])
        assert any("unknown" in e for e in verify_match(r, ev))

    def test_duplicate_caught(self):
        ev = make_evidence(["L1"])
        r = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           source_time_range=SourceTimeRange(0, 2),
                           line_matches=[m("L1")]),
            NodeGeneration(group_id="G2", covered_line_ids=["L1"],
                           source_time_range=SourceTimeRange(2, 4),
                           line_matches=[m("L1")]),
        ])
        assert any("duplicate" in e for e in verify_match(r, ev))

    def test_unmatched_ok(self):
        ev = make_evidence(["L1", "L2"])
        r = MatchResult(
            node_generations=[
                NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                               source_time_range=SourceTimeRange(0, 2),
                               line_matches=[m("L1")])
            ],
            unmatched_lines=[UnmatchedLine(line_id="L2", reason="no match",
                                            original="o_L2", rewritten="r_L2")]
        )
        assert verify_match(r, ev) == []

    def test_time_range_mismatch(self):
        ev = make_evidence(["L1"])
        r = MatchResult(node_generations=[
            NodeGeneration(group_id="G1", covered_line_ids=["L1"],
                           source_time_range=SourceTimeRange(100, 104),
                           line_matches=[m("L1")])
        ])
        assert any("too late" in e for e in verify_match(r, ev))


class TestVerifyPrompt:
    def test_passes_when_dialogue_present(self):
        g = NodeGeneration(group_id="G1", line_matches=[
            LineNodeMatch(line_id="L1", original_line="hello", rewritten_line="hi",
                          node_id="n1", match_reasoning="v", confidence=0.9)
        ])
        assert verify_prompt(g, "scene with hi dialogue") == []

    def test_fails_when_dialogue_missing(self):
        g = NodeGeneration(group_id="G1", line_matches=[
            LineNodeMatch(line_id="L1", original_line="hello", rewritten_line="hi",
                          node_id="n1", match_reasoning="v", confidence=0.9)
        ])
        assert len(verify_prompt(g, "completely different text")) > 0

    def test_skips_unchanged_lines(self):
        g = NodeGeneration(group_id="G1", line_matches=[
            LineNodeMatch(line_id="L1", original_line="same", rewritten_line="same",
                          node_id="n1", match_reasoning="neighbor", confidence=0.5)
        ])
        assert verify_prompt(g, "some prompt without same") == []

    def test_loose_match_punctuation(self):
        g = NodeGeneration(group_id="G1", line_matches=[
            LineNodeMatch(line_id="L1", original_line="hello", rewritten_line="Hi there",
                          node_id="n1", match_reasoning="v", confidence=0.9)
        ])
        # "Hi there" should be found in the prompt
        assert verify_prompt(g, "scene with Hi there dialogue") == []


class TestLooseMatch:
    def test_exact(self):
        assert _loose_match("hello", "say hello world")

    def test_case_insensitive(self):
        assert _loose_match("Hello", "say hello world")

    def test_trailing_punctuation(self):
        assert _loose_match("hello.", "say hello world")

    def test_not_found(self):
        assert not _loose_match("goodbye", "say hello world")


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestVerifyMatch, TestVerifyPrompt, TestLooseMatch]:
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
