"""Tests for prompt rewriting."""
from skills.timeline_plan.prompt_extractor import (
    extract_and_rewrite_prompt,
    _generate_prompt_from_scene,
)


class FakeLine:
    def __init__(self, dialogue="", original="", rewritten="", speaker="Speaker"):
        self.dialogue = dialogue
        self.original = original
        self.rewritten = rewritten
        self.speaker = speaker


class TestGeneratePromptFromScene:
    def test_basic_generation(self):
        lines = [FakeLine(original="hello", rewritten="hi there", speaker="Donny")]
        result = _generate_prompt_from_scene(lines, "A bar scene")
        assert "A bar scene" in result
        assert "hi there" in result
        assert "Donny" in result

    def test_default_scene_description(self):
        lines = [FakeLine(original="test", rewritten="rewritten test")]
        result = _generate_prompt_from_scene(lines)
        assert "A cinematic scene" in result

    def test_empty_lines(self):
        result = _generate_prompt_from_scene([], "Scene")
        assert "Scene" in result


class TestExtractAndRewritePrompt:
    def test_empty_prompt_uses_fallback(self):
        """When full_prompt is empty, use scene_description fallback."""
        lines = [FakeLine(original="hello", rewritten="hi", speaker="Donny")]
        result = extract_and_rewrite_prompt("", lines, "Opening scene")
        assert "Opening scene" in result
        assert "hi" in result

    def test_empty_prompt_no_scene_desc(self):
        """Even without scene_description, fallback produces output."""
        lines = [FakeLine(original="x", rewritten="y")]
        result = extract_and_rewrite_prompt("", lines)
        assert len(result) > 0
