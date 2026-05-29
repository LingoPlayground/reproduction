"""Tests for canvas node matcher."""
from skills.timeline_plan.models import CanvasNode
from skills.timeline_plan.canvas_matcher import (
    text_overlap_score, match_canvas_node_for_shot,
)


class FakeShot:
    def __init__(self, lines=None, scene_description=""):
        self.lines = lines or []
        self.scene_description = scene_description


class FakeLine:
    def __init__(self, dialogue=""):
        self.dialogue = dialogue


class TestTextOverlapScore:
    def test_exact_match(self):
        score = text_overlap_score("this ceremony is boring", "this ceremony is boring")
        assert score > 0.8

    def test_partial_match(self):
        score = text_overlap_score(
            "this ceremony is boring",
            "He says this ceremony is boring and walks away"
        )
        assert score > 0.5

    def test_no_match(self):
        score = text_overlap_score("hello world", "xyz abc def ghi")
        assert score < 0.3

    def test_case_insensitive(self):
        score = text_overlap_score("Hello World", "hello world")
        assert score > 0.8


class TestMatchCanvasNodeForShot:
    def test_matches_node_with_overlapping_dialogue(self):
        shot = FakeShot(
            lines=[FakeLine("this ceremony is boring"), FakeLine("let's see who wants me")],
            scene_description="Graduation ceremony"
        )
        nodes = [
            CanvasNode(node_id="node1", prompt="random content", video_url="http://x.com/v1.mp4"),
            CanvasNode(node_id="node2", prompt="He says this ceremony is boring and let's see who wants me", video_url="http://x.com/v2.mp4"),
        ]
        matched, confidence = match_canvas_node_for_shot(shot, nodes)
        assert matched is not None
        assert matched.node_id == "node2"
        assert confidence > 0.5

    def test_no_match_below_threshold(self):
        shot = FakeShot(lines=[FakeLine("unique dialogue text")], scene_description="A scene")
        nodes = [CanvasNode(node_id="node1", prompt="totally unrelated content", video_url="http://x.com/v1.mp4")]
        matched, _ = match_canvas_node_for_shot(shot, nodes)
        assert matched is None

    def test_empty_nodes_returns_none(self):
        shot = FakeShot(lines=[FakeLine("hello")])
        matched, _ = match_canvas_node_for_shot(shot, [])
        assert matched is None
