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
