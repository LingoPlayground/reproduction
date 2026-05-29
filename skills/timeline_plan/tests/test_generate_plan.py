"""Tests for timeline plan generator orchestrator."""
import json
from dataclasses import asdict
from skills.timeline_plan.models import TimelinePlan, CanvasNode, Stage3Input
from skills.timeline_plan.generate_plan import generate_timeline_plan


class FakeLine:
    def __init__(self, line_id, dialogue, start_s=0.0, end_s=1.0):
        self.line_id = line_id
        self.dialogue = dialogue
        self.start_seconds = start_s
        self.end_seconds = end_s


class FakeShot:
    def __init__(self, shot_number, start, end, scene_desc="", lines=None):
        self.shot_number = shot_number
        self.start_seconds = start
        self.end_seconds = end
        self.scene_description = scene_desc
        self.lines = lines or []


class FakeScript:
    def __init__(self, shots):
        self.shots = shots


class FakeScriptOutput:
    def __init__(self, shots):
        self.script = FakeScript(shots)


def make_rewrite(line_id, original, rewritten, shot_num, start_s, end_s):
    return {"line_id": line_id, "original": original, "rewritten": rewritten,
            "shot_number": shot_num, "start_seconds": start_s, "end_seconds": end_s,
            "shot_scene": "", "speaker": "Speaker"}


class TestGenerateTimelinePlan:
    def test_produces_original_when_no_rewrite(self):
        shots = [FakeShot(1, 0.0, 10.0, "Opening", [FakeLine("p1_l1", "hello", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hello", "hello", 1, 1.0, 2.0)]}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) == 1
        assert plan.items[0].source == "original"

    def test_produces_seedance_when_rewritten(self):
        shots = [FakeShot(1, 0.0, 10.0, "Opening", [FakeLine("p1_l1", "hello", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hello", "hi there", 1, 1.0, 2.0)]}
        nodes = [CanvasNode(node_id="n1", prompt='He says "hello"', video_url="http://x.com/v.mp4", reference_images=["http://x.com/r.png"])]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) == 1
        item = plan.items[0]
        assert item.source == "seedance"
        assert item.rewritten_prompt is not None
        assert "hi there" in item.rewritten_prompt

    def test_mixed_original_and_seedance(self):
        shots = [
            FakeShot(1, 0.0, 5.0, "Scene A", [FakeLine("p1_l1", "hello", 1.0, 2.0)]),
            FakeShot(2, 5.0, 10.0, "Scene B", [FakeLine("p2_l1", "goodbye", 6.0, 7.0)]),
        ]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [
            make_rewrite("p1_l1", "hello", "hello", 1, 1.0, 2.0),
            make_rewrite("p2_l1", "goodbye", "farewell", 2, 6.0, 7.0),
        ]}
        nodes = [CanvasNode(node_id="n2", prompt='She says "goodbye"', video_url="http://x.com/v2.mp4", reference_images=["http://x.com/r2.png"])]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) == 2
        assert plan.items[0].source == "original"
        assert plan.items[1].source == "seedance"

    def test_degradation_level_tracking(self):
        shots = [FakeShot(1, 0.0, 10.0, "Scene", [FakeLine("p1_l1", "unique text", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "unique text", "rewritten unique", 1, 1.0, 2.0)]}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=[], level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) == 1
        assert plan.items[0].source == "seedance"
        assert plan.items[0].degradation_level > 0

    def test_json_serializable_output(self):
        shots = [FakeShot(1, 0.0, 10.0, "Scene", [FakeLine("p1_l1", "hi", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hi", "hello", 1, 1.0, 2.0)]}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, level="B2")
        plan = generate_timeline_plan(inp)
        json_str = json.dumps(asdict(plan), indent=2)
        parsed = json.loads(json_str)
        assert parsed["pipeline_version"] == "2.0"
        assert len(parsed["items"]) == 1

    def test_empty_shots(self):
        script = FakeScriptOutput([])
        rewrite = {"level": "B2", "lines": []}
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, level="B2")
        plan = generate_timeline_plan(inp)
        assert len(plan.items) == 0
