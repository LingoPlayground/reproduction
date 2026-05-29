"""Tests for timeline plan data models."""
import json
from dataclasses import asdict
from skills.timeline_plan.models import (
    CutPoint, KeyFrame, CanvasNode, TimelinePlanItem, TimelinePlan, Stage3Input
)


class TestCutPoint:
    def test_basic_creation(self):
        cp = CutPoint(time_sec=32.5, confidence=1.0)
        assert cp.time_sec == 32.5
        assert cp.confidence == 1.0

    def test_default_confidence(self):
        cp = CutPoint(time_sec=10.0)
        assert cp.confidence == 1.0


class TestKeyFrame:
    def test_basic_creation(self):
        kf = KeyFrame(time_sec=5.0, image_path="/tmp/frame.png", shot_number=1)
        assert kf.shot_number == 1
        assert kf.image_path == "/tmp/frame.png"


class TestCanvasNode:
    def test_creation_with_optional_duration(self):
        node = CanvasNode(
            node_id="abc123", prompt="test prompt",
            video_url="http://example.com/v.mp4",
            reference_images=["http://example.com/img.png"],
        )
        assert node.duration_sec is None

    def test_creation_with_duration(self):
        node = CanvasNode(
            node_id="abc123", prompt="test prompt",
            video_url="http://example.com/v.mp4",
            reference_images=["http://example.com/img.png"],
            duration_sec=15.5,
        )
        assert node.duration_sec == 15.5


class TestTimelinePlanItem:
    def test_original_source_minimal(self):
        item = TimelinePlanItem(
            shot_id="shot_1", shot_number=1,
            source="original", start_sec=0.0, end_sec=26.7,
            scene_description="Opening scene",
        )
        assert item.source == "original"
        assert item.ref_images == []
        assert item.rewritten_prompt is None
        assert item.matched_node_id is None
        assert item.match_confidence is None
        assert item.degradation_level == 0
        assert item.seedance_duration is None
        assert item.original_duration is None

    def test_seedance_source_with_all_fields(self):
        item = TimelinePlanItem(
            shot_id="shot_3", shot_number=3,
            source="seedance", start_sec=32.5, end_sec=43.5,
            scene_description="Breakup scene",
            ref_images=["http://example.com/ref1.png"],
            rewritten_prompt="A man asks...",
            matched_node_id="node_abc",
            match_confidence=0.85,
            degradation_level=1,
            seedance_duration=11,
            original_duration=10.8,
        )
        assert item.source == "seedance"
        assert len(item.ref_images) == 1
        assert item.degradation_level == 1
        assert item.seedance_duration == 11
        assert item.original_duration == 10.8

    def test_serialization_roundtrip(self):
        item = TimelinePlanItem(
            shot_id="shot_1", shot_number=1,
            source="original", start_sec=0.0, end_sec=26.7,
            scene_description="Opening scene",
        )
        d = asdict(item)
        assert d["shot_id"] == "shot_1"
        assert d["source"] == "original"


class TestTimelinePlan:
    def test_creation_with_items(self):
        items = [
            TimelinePlanItem(
                shot_id="shot_1", shot_number=1,
                source="original", start_sec=0.0, end_sec=26.7,
                scene_description="Opening",
            ),
            TimelinePlanItem(
                shot_id="shot_2", shot_number=2,
                source="seedance", start_sec=26.7, end_sec=45.0,
                scene_description="Conflict",
                ref_images=["http://example.com/r.png"],
                rewritten_prompt="Conflict prompt",
            ),
        ]
        plan = TimelinePlan(
            title="Test Episode", level="B2",
            original_video_path="/videos/ep1.mp4",
            total_duration_sec=45.0, items=items,
        )
        assert plan.pipeline_version == "2.0"
        assert len(plan.items) == 2
        assert plan.items[0].source == "original"
        assert plan.items[1].source == "seedance"

    def test_json_roundtrip(self):
        items = [
            TimelinePlanItem(
                shot_id="shot_1", shot_number=1,
                source="original", start_sec=0.0, end_sec=10.0,
                scene_description="Test scene",
            )
        ]
        plan = TimelinePlan(
            title="Test", level="B2",
            original_video_path="/v/test.mp4",
            total_duration_sec=10.0, items=items,
        )
        serialized = json.dumps(asdict(plan), indent=2)
        deserialized = json.loads(serialized)
        assert deserialized["pipeline_version"] == "2.0"
        assert deserialized["items"][0]["source"] == "original"


class TestStage3Input:
    def test_all_fields(self):
        inp = Stage3Input(
            script_output=None,
            video_cut_points=[],
            keyframes=[],
            node_cut_points={},
            rewrite_json={"level": "B2", "lines": []},
            canvas_nodes=[],
            level="B2",
        )
        assert inp.level == "B2"
        assert inp.rewrite_json["level"] == "B2"
