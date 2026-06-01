"""Integration test: full v4 pipeline from input -> TimelinePlan."""
import json
from unittest.mock import patch, MagicMock
from skills.timeline_plan.models import CanvasNode, CutPoint, Stage3Input
from skills.timeline_plan.generate_plan import generate_timeline_plan


class FakeShot:
    def __init__(self, shot_number, start_sec, end_sec, scene_desc="test scene"):
        self.shot_number = shot_number
        self.start_seconds = start_sec
        self.end_seconds = end_sec
        self.scene_description = scene_desc


class FakeScript:
    def __init__(self, shots):
        self.shots = shots


class FakeScriptOutput:
    def __init__(self, shots, title="Test Episode"):
        self.script = FakeScript(shots)
        self.title = title


class TestV4Pipeline:
    @patch("skills.timeline_plan.segment_matcher._get_client")
    @patch("skills.timeline_plan.prompt_rewriter._get_client")
    def test_full_pipeline_single_rewritten_line(self, mock_rewrite_client, mock_match_client):
        """End-to-end: one line rewritten -> matched -> window -> prompt -> plan."""
        mock_match = MagicMock()
        mock_match.chat.completions.create.return_value.choices = [MagicMock()]
        mock_match.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "matches": [{"atom_id": "atom_001", "node_id": "n1", "confidence": 0.9, "reasoning": "match"}],
            "unmatched": [],
        })
        mock_match_client.return_value = mock_match

        mock_rewrite = MagicMock()
        mock_rewrite.chat.completions.create.return_value.choices = [MagicMock()]
        mock_rewrite.chat.completions.create.return_value.choices[0].message.content = "Rewritten prompt: hi there"
        mock_rewrite_client.return_value = mock_rewrite

        script_output = FakeScriptOutput([FakeShot(1, 0.0, 10.0, "Mia in kitchen")])
        rewrite_json = {
            "lines": [{"line_id": "L1", "speaker": "Mia", "original": "hello", "rewritten": "hi",
                       "start_seconds": 2.0, "end_seconds": 4.0, "shot_number": 1,
                       "shot_scene": "Mia in kitchen"}]
        }
        canvas_nodes = [CanvasNode(node_id="n1", prompt="Scene: Mia says hello in kitchen",
                                   video_url="", reference_images=["img.jpg"])]

        inp = Stage3Input(script_output=script_output, rewrite_json=rewrite_json,
                          canvas_nodes=canvas_nodes, level="B2")
        plan = generate_timeline_plan(inp)

        modified = [i for i in plan.items if i.source == "modified"]
        assert len(modified) == 1
        assert modified[0].matched_node_id == "n1"
        assert modified[0].rewritten_prompt is not None
        assert "L1" in modified[0].covered_line_ids
        assert modified[0].ref_images == ["img.jpg"]
        assert plan.title == "Test Episode"

        # No gaps
        sorted_items = sorted(plan.items, key=lambda i: i.start_sec)
        for i in range(len(sorted_items) - 1):
            assert sorted_items[i].end_sec <= sorted_items[i + 1].start_sec + 0.1

    def test_no_rewritten_lines_returns_all_original(self):
        script_output = FakeScriptOutput([FakeShot(1, 0.0, 10.0, "scene")])
        rewrite_json = {
            "lines": [{"line_id": "L1", "speaker": "M", "original": "hello", "rewritten": "hello",
                       "start_seconds": 2.0, "end_seconds": 4.0, "shot_number": 1, "shot_scene": "s"}]
        }
        inp = Stage3Input(script_output=script_output, rewrite_json=rewrite_json, level="B2")
        plan = generate_timeline_plan(inp)
        assert all(i.source == "original" for i in plan.items)


if __name__ == "__main__":
    import sys
    failed = 0
    tests = TestV4Pipeline()
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
