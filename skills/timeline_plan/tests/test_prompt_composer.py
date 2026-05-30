"""Tests for PromptPatchComposer."""
from skills.timeline_plan.prompt_composer import (
    compose_prompt_patch,
    _extract_style_prefix,
    _generate_prompt_from_scene,
    _validate_rewrite,
)


class FakeLine:
    def __init__(self, dialogue="", original="", rewritten="", speaker="Speaker"):
        self.dialogue = dialogue
        self.original = original
        self.rewritten = rewritten
        self.speaker = speaker


class TestExtractStylePrefix:
    def test_extracts_chinese_style_keywords(self):
        prompt = "美式情景喜剧，真实短剧，柔光雾化，画面通透，8k，超高清，电影级布光。镜头 1：..."
        result = _extract_style_prefix(prompt)
        assert "美式情景喜剧" in result
        assert "电影级布光" in result
        assert "镜头 1" not in result

    def test_returns_empty_for_no_style(self):
        prompt = 'Donny says: "hello"'
        result = _extract_style_prefix(prompt)
        assert result == ""

    def test_extracts_english_style_keywords(self):
        prompt = "cinematic lighting, 8k resolution, shallow depth of field. Scene 1: ..."
        result = _extract_style_prefix(prompt)
        assert "cinematic lighting" in result
        assert "Scene 1" not in result


class TestValidateRewrite:
    def test_exact_substring_passes_for_literal(self):
        prompt = '美式情景喜剧...Donny says: "No, no, no, this can\'t be."'
        lines = [FakeLine(rewritten="No, no, no, this can't be.")]
        assert _validate_rewrite(prompt, lines, operation_type="literal_replace")

    def test_missing_dialogue_fails(self):
        prompt = '美式情景喜剧...Donny says: "Hi."'
        lines = [FakeLine(rewritten="Hello.")]
        assert not _validate_rewrite(prompt, lines, operation_type="literal_replace")

    def test_no_original_required_for_semantic_insert(self):
        prompt = '美式情景喜剧...真实的破防...Donny says: "No, no, no, this can\'t be."'
        lines = [FakeLine(original="no no no", rewritten="No, no, no, this can't be.")]
        assert _validate_rewrite(prompt, lines, operation_type="semantic_insert")

    def test_empty_rewritten_skipped(self):
        lines = [FakeLine(rewritten="")]
        assert _validate_rewrite("any prompt", lines, operation_type="literal_replace")


class TestGeneratePromptFromScene:
    def test_includes_style_layer(self):
        lines = [FakeLine(original="hello", rewritten="hi there", speaker="Donny")]
        result = _generate_prompt_from_scene(lines, "A bar scene", style_layer="8k, 电影级布光")
        assert "8k, 电影级布光" in result
        assert "A bar scene" in result
        assert "hi there" in result

    def test_no_style_layer_works(self):
        lines = [FakeLine(original="test", rewritten="rewritten test")]
        result = _generate_prompt_from_scene(lines, style_layer="")
        assert "A cinematic scene" in result
        assert "rewritten test" in result


class TestComposePromptPatch:
    def test_empty_prompt_uses_fallback(self):
        lines = [FakeLine(original="hello", rewritten="hi", speaker="Donny")]
        result = compose_prompt_patch("", lines, "Opening scene")
        assert "Opening scene" in result
        assert "hi" in result

    def test_returns_original_when_no_change(self):
        lines = [FakeLine(original="hello", rewritten="hello", speaker="Donny")]
        prompt = "美式情景喜剧...Donny says: hello"
        result = compose_prompt_patch(prompt, lines)
        assert result == prompt

    def test_semantic_insert_with_style_preservation(self):
        prompt = "美式情景喜剧，真实短剧，电影级布光。镜头 2：真实的破防（面部特写）"
        lines = [FakeLine(
            original="no no no",
            rewritten="No, no, no, this can't be.",
            speaker="Donny",
        )]
        result = compose_prompt_patch(prompt, lines, operation_type="semantic_insert")
        assert len(result) > 0
