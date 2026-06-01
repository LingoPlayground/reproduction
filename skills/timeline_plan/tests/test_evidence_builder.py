"""Tests for evidence_builder.py — v3.0 input packaging."""
from skills.timeline_plan.evidence_builder import build_evidence
from skills.timeline_plan.models import CanvasNode, CutPoint


class TestEvidence:
    def test_separates_rewritten(self):
        lines = [
            {"line_id": "L1", "original": "a", "rewritten": "b", "speaker": "S",
             "start_seconds": 0, "end_seconds": 2, "shot_number": 1, "shot_scene": "s1"},
            {"line_id": "L2", "original": "x", "rewritten": "x", "speaker": "S",
             "start_seconds": 3, "end_seconds": 4, "shot_number": 1, "shot_scene": "s1"},
        ]
        ev = build_evidence([], lines, [], [])
        assert len(ev["rewrite_lines"]) == 1
        assert ev["rewrite_lines"][0]["line_id"] == "L1"
        assert len(ev["neighbor_lines"]) == 1
        assert ev["neighbor_lines"][0]["line_id"] == "L2"

    def test_scene_context(self):
        lines = [
            {"line_id": "L1", "original": "a", "rewritten": "b", "speaker": "S",
             "start_seconds": 0, "end_seconds": 2, "shot_number": 1,
             "shot_scene": "Donny at party"},
            {"line_id": "L2", "original": "c", "rewritten": "d", "speaker": "S",
             "start_seconds": 3, "end_seconds": 4, "shot_number": 2,
             "shot_scene": "Rachel in apartment"},
        ]
        ev = build_evidence([], lines, [], [])
        assert len(ev["scene_context"]) == 2
        assert ev["scene_context"][0]["shot_number"] == 1
        assert "Donny" in ev["scene_context"][0]["description"]

    def test_passes_full_node_prompt(self):
        nodes = [CanvasNode(node_id="n1", prompt="Scene: Rachel says hi.", video_url="",
                            reference_images=["img.jpg"])]
        ev = build_evidence([], [], nodes, [])
        assert len(ev["canvas_nodes"]) == 1
        assert ev["canvas_nodes"][0]["prompt"] == "Scene: Rachel says hi."
        assert "detected_quoted_dialogue" not in ev["canvas_nodes"][0]

    def test_timeline_has_scene_cuts(self):
        cuts = [CutPoint(time_sec=1.0), CutPoint(time_sec=5.0)]
        ev = build_evidence([], [], [], cuts)
        assert ev["timeline"]["scene_cuts"] == [1.0, 5.0]

    def test_constraints(self):
        ev = build_evidence([], [], [], [])
        c = ev["constraints"]
        assert c["must_cover_every_rewritten_line"] is True
        assert c["min_modified_duration_sec"] == 4.0


if __name__ == "__main__":
    import sys
    failed = 0
    tests = TestEvidence()
    for name in sorted(dir(tests)):
        if name.startswith("test_"):
            try:
                getattr(tests, name)()
                print(f"  PASS: {name}")
            except AssertionError as e:
                print(f"  FAIL: {name} - {e}")
                failed += 1
    if failed:
        print(f"\n{failed} FAILED")
        sys.exit(1)
    print("\nAll tests passed!")
