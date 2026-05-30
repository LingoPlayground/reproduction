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


# ── v3 dataclasses ──────────────────────────────────────────────────

from skills.timeline_plan.models import (
    PromptPatchPlan, CoveragePlan, MatchEvidence,
)


class TestPromptPatchPlan:
    def test_literal_replace_creation(self):
        plan = PromptPatchPlan(
            operation_type="literal_replace",
            global_style="8k, 超高清, 电影级布光",
            local_visual_context="镜头 1：三层景深",
            dialogue_patches=[],
            discarded_sections=[],
            final_prompt="8k, 超高清...镜头1...Donny says: 'hi'",
        )
        assert plan.operation_type == "literal_replace"
        assert plan.global_style == "8k, 超高清, 电影级布光"
        assert plan.final_prompt is not None

    def test_semantic_insert_defaults(self):
        plan = PromptPatchPlan(
            operation_type="semantic_insert",
            global_style="",
            local_visual_context="",
            dialogue_patches=[],
            discarded_sections=[],
            final_prompt="",
        )
        assert plan.operation_type == "semantic_insert"
        assert plan.dialogue_patches == []


class TestCoveragePlan:
    def test_direct_strategy(self):
        cp = CoveragePlan(
            start_sec=17.47,
            end_sec=29.55,
            included_rewritten_line_ids=["p001_l003", "p001_l004"],
            borrowed_original_line_ids=[],
            duration_strategy="direct",
        )
        assert cp.duration_strategy == "direct"
        assert cp.duration_expansion_sec == 0.0
        assert cp.end_sec - cp.start_sec >= 4.0

    def test_pad_after_strategy(self):
        cp = CoveragePlan(
            start_sec=2.83,
            end_sec=6.83,
            included_rewritten_line_ids=["p001_l001", "p001_l002"],
            borrowed_original_line_ids=[],
            duration_strategy="pad_after",
            duration_expansion_sec=0.40,
        )
        assert cp.duration_strategy == "pad_after"
        assert cp.duration_expansion_sec == 0.40


class TestMatchEvidence:
    def test_quoted_dialogue_signal(self):
        ev = MatchEvidence(
            signal="quoted_dialogue",
            detail='Found "This ceremony is boring." in node prompt',
            confidence=0.97,
        )
        assert ev.signal == "quoted_dialogue"
        assert ev.confidence > 0.9

    def test_implicit_visual_scene_signal(self):
        ev = MatchEvidence(
            signal="implicit_visual_scene",
            detail="Prompt describes Donny's breakdown close-up",
            confidence=0.85,
        )
        assert ev.signal == "implicit_visual_scene"
        assert ev.confidence < 1.0


class TestTimelinePlanItemV3Fields:
    """Verify new v3 optional fields have correct defaults and serialization."""
    def test_defaults_for_backward_compat(self):
        item = TimelinePlanItem(
            shot_id="shot_1", shot_number=1,
            source="original", start_sec=0.0, end_sec=10.0,
            scene_description="Test",
        )
        assert item.operation_type is None
        assert item.duration_strategy is None
        assert item.covered_line_ids == []
        assert item.borrowed_line_ids == []
        assert item.source_node_ids == []
        assert item.degradation_reason == ""

    def test_v3_fields_serialize(self):
        from dataclasses import asdict
        import json as _json
        item = TimelinePlanItem(
            shot_id="shot_1", shot_number=1,
            source="seedance", start_sec=0.0, end_sec=10.0,
            scene_description="Test",
            operation_type="semantic_insert",
            duration_strategy="pad_after",
            covered_line_ids=["p001_l001"],
            degradation_reason="duration_padded_to_meet_min_4s",
        )
        d = asdict(item)
        assert d["operation_type"] == "semantic_insert"
        assert d["duration_strategy"] == "pad_after"
        assert d["covered_line_ids"] == ["p001_l001"]
        # Verify old JSON deserialization still works (new fields optional)
        old_json = '{"shot_id": "s1", "shot_number": 1, "source": "original", "start_sec": 0.0, "end_sec": 10.0, "scene_description": "T"}'
        parsed = _json.loads(old_json)
        assert parsed["source"] == "original"
