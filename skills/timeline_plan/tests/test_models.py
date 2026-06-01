"""Tests for models.py — v3.0 deterministic execution models."""
import json
from dataclasses import asdict
from skills.timeline_plan.models import (
    CutPoint, CanvasNode, TimelinePlanItem, TimelinePlan,
    Stage3Input, MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
)


class TestTimelinePlanItem:
    def test_original_source(self):
        item = TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                                start_sec=0.0, end_sec=5.0, scene_description="test")
        assert item.source == "original"
        assert item.duration_sec == 5.0
        assert item.ref_images == []
        assert item.rewritten_prompt is None

    def test_modified_source(self):
        item = TimelinePlanItem(shot_id="s1", shot_number=1, source="modified",
                                start_sec=10.0, end_sec=15.0, scene_description="test",
                                rewritten_prompt="new prompt", matched_node_id="n1",
                                match_confidence=0.9, covered_line_ids=["L1", "L2"],
                                source_node_ids=["n1"])
        assert item.source == "modified"
        assert item.rewritten_prompt == "new prompt"
        assert item.covered_line_ids == ["L1", "L2"]
        assert item.source_node_ids == ["n1"]
        assert item.match_confidence == 0.9

    def test_duration_sec_zero_clamped(self):
        item = TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                                start_sec=5.0, end_sec=3.0, scene_description="test")
        assert item.duration_sec == 0.0


class TestTimelinePlan:
    def test_no_version_field(self):
        plan = TimelinePlan(title="Test", level="B2")
        assert plan.items == []

    def test_with_items(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                             start_sec=0.0, end_sec=5.0, scene_description="s1"),
            TimelinePlanItem(shot_id="s2", shot_number=2, source="modified",
                             start_sec=5.0, end_sec=10.0, scene_description="s2"),
        ]
        plan = TimelinePlan(title="Test", level="B2", items=items)
        assert len(plan.items) == 2
        assert plan.items[0].source == "original"
        assert plan.items[1].source == "modified"


class TestStage3Input:
    def test_defaults(self):
        inp = Stage3Input(script_output=None)
        assert inp.level == "B2"
        assert inp.canvas_nodes == []


class TestConstants:
    def test_duration_constants(self):
        assert MIN_MODIFIED_DURATION == 4.0
        assert MAX_MODIFIED_DURATION == 30.0


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestTimelinePlanItem, TestTimelinePlan, TestStage3Input, TestConstants]:
        t = cls()
        for name in sorted(dir(t)):
            if name.startswith("test_"):
                try:
                    getattr(t, name)()
                    print(f"  PASS: {cls.__name__}.{name}")
                except AssertionError as e:
                    print(f"  FAIL: {cls.__name__}.{name} - {e}")
                    failed += 1
    if failed:
        print(f"\n{failed} FAILED")
        sys.exit(1)
    print("\nAll tests passed!")
