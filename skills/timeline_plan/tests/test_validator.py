"""Tests for multilayer prompt validator."""
from skills.timeline_plan.validator import (
    extract_style_anchors,
    validate_style_preservation,
)


class TestExtractStyleAnchors:
    def test_extracts_chinese_anchors(self):
        prompt = "美式情景喜剧，真实短剧，柔光雾化，8k，超高清，电影级布光。镜头 1：..."
        anchors = extract_style_anchors(prompt)
        assert "美式情景喜剧" in anchors
        assert "电影级布光" in anchors
        assert "8k" in anchors

    def test_no_duplicates(self):
        prompt = "8k 8k 8k resolution"
        anchors = extract_style_anchors(prompt)
        assert anchors.count("8k") == 1

    def test_empty_for_no_style(self):
        prompt = "Donny says: hello world"
        anchors = extract_style_anchors(prompt)
        assert len(anchors) == 0

    def test_extracts_english_anchors(self):
        prompt = "cinematic lighting, shallow depth of field, 8k"
        anchors = extract_style_anchors(prompt)
        assert "cinematic" in anchors
        assert "shallow depth of field" in anchors


class TestValidateStylePreservation:
    def test_passes_when_all_preserved(self):
        original = "美式情景喜剧，8k，电影级布光。镜头 1"
        rewritten = "美式情景喜剧，8k，电影级布光。Donny says: hi"
        passes, missing, ratio = validate_style_preservation(original, rewritten)
        assert passes
        assert len(missing) == 0
        assert ratio == 1.0

    def test_fails_when_most_missing(self):
        original = "美式情景喜剧，真实短剧，柔光雾化，8k，超高清，电影级布光"
        rewritten = "8k. Donny says: hi"
        passes, missing, ratio = validate_style_preservation(original, rewritten)
        assert not passes
        assert len(missing) >= 3
        assert ratio < 0.6

    def test_passes_at_threshold_boundary(self):
        original = "美式情景喜剧，8k，超高清，柔光雾化"
        rewritten = "美式情景喜剧，8k. Donny says: hi"
        passes, _, ratio = validate_style_preservation(original, rewritten)
        assert ratio == 0.5
        assert not passes  # 2/4 < 0.6

    def test_passes_when_no_anchors(self):
        original = "Donny says: hello world"
        rewritten = "Donny says: hi"
        passes, missing, ratio = validate_style_preservation(original, rewritten)
        assert passes
        assert ratio == 1.0

    def test_case_insensitive(self):
        original = "CINEMATIC LIGHTING"
        rewritten = "cinematic lighting"
        passes, _, _ = validate_style_preservation(original, rewritten)
        assert passes


# ── L1/L2 timeline validation ────────────────────────────────────────

from skills.timeline_plan.validator import (
    validate_timeline_item,
    validate_timeline_items,
)
from skills.timeline_plan.models import TimelinePlanItem


class TestValidateTimelineItem:
    def test_valid_seedance_item(self):
        item = TimelinePlanItem(
            shot_id="shot_1", shot_number=1, source="modified",
            start_sec=2.0, end_sec=6.0, scene_description="Test",
            rewritten_prompt="A prompt with dialogue",
        )
        errors = validate_timeline_item(item)
        assert len(errors) == 0

    def test_valid_original_item(self):
        item = TimelinePlanItem(
            shot_id="shot_2", shot_number=2, source="original",
            start_sec=6.0, end_sec=15.0, scene_description="Scene",
        )
        errors = validate_timeline_item(item)
        assert len(errors) == 0

    def test_inverted_time_range(self):
        item = TimelinePlanItem(
            shot_id="bad", shot_number=1, source="modified",
            start_sec=10.0, end_sec=5.0, scene_description="T",
            rewritten_prompt="prompt",
        )
        errors = validate_timeline_item(item)
        assert any("start_sec" in e for e in errors)

    def test_seedance_without_prompt(self):
        item = TimelinePlanItem(
            shot_id="bad", shot_number=1, source="modified",
            start_sec=2.0, end_sec=6.0, scene_description="T",
        )
        errors = validate_timeline_item(item)
        assert any("rewritten_prompt" in e for e in errors)

    def test_empty_shot_id(self):
        item = TimelinePlanItem(
            shot_id="", shot_number=1, source="original",
            start_sec=0.0, end_sec=5.0, scene_description="T",
        )
        errors = validate_timeline_item(item)
        assert any("shot_id" in e for e in errors)


class TestValidateTimelineItems:
    def test_no_errors_for_clean_timeline(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="modified",
                             start_sec=0.0, end_sec=4.0, scene_description="A",
                             rewritten_prompt="prompt"),
            TimelinePlanItem(shot_id="s2", shot_number=2, source="original",
                             start_sec=4.0, end_sec=10.0, scene_description="B"),
        ]
        errors = validate_timeline_items(items, video_duration=10.0)
        assert len(errors) == 0

    def test_detects_overlap(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="modified",
                             start_sec=0.0, end_sec=5.0, scene_description="A",
                             rewritten_prompt="p", covered_line_ids=["l1"]),
            TimelinePlanItem(shot_id="s2", shot_number=2, source="original",
                             start_sec=4.0, end_sec=10.0, scene_description="B"),
        ]
        errors = validate_timeline_items(items, video_duration=10.0)
        assert any("Overlap" in e for e in errors)

    def test_detects_duplicate_line_coverage(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="modified",
                             start_sec=0.0, end_sec=5.0, scene_description="A",
                             rewritten_prompt="p", covered_line_ids=["l1", "l2"]),
            TimelinePlanItem(shot_id="s2", shot_number=2, source="modified",
                             start_sec=5.0, end_sec=10.0, scene_description="B",
                             rewritten_prompt="p", covered_line_ids=["l2", "l3"]),
        ]
        errors = validate_timeline_items(items, video_duration=10.0)
        assert any("covered by both" in e for e in errors)

    def test_detects_gap_at_start(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                             start_sec=2.0, end_sec=10.0, scene_description="A"),
        ]
        errors = validate_timeline_items(items, video_duration=10.0)
        assert any("gap at start" in e for e in errors)

    def test_detects_gap_at_end(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                             start_sec=0.0, end_sec=8.0, scene_description="A"),
        ]
        errors = validate_timeline_items(items, video_duration=10.0)
        assert any("gap at end" in e for e in errors)
