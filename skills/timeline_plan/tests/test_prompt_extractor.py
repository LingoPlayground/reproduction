"""Tests for prompt fragment extraction."""
from skills.timeline_plan.prompt_extractor import (
    extract_by_section_headers,
    extract_by_dialogue_keywords,
    replace_dialogue_in_fragment,
    extract_and_rewrite_prompt,
)


class FakeShot:
    def __init__(self, scene_description="", shot_number=1, lines=None):
        self.scene_description = scene_description
        self.shot_number = shot_number
        self.lines = lines or []


class FakeLine:
    def __init__(self, dialogue="", original="", rewritten=""):
        self.dialogue = dialogue
        self.original = original
        self.rewritten = rewritten


class TestExtractBySectionHeaders:
    def test_extracts_by_shot_number_label(self):
        prompt = "镜头 1：Opening scene\n镜头 2：Conflict scene\n镜头 3：Resolution"
        result = extract_by_section_headers(prompt, 2)
        assert result is not None
        assert "镜头 2" in result
        assert "Conflict" in result

    def test_extracts_single_shot(self):
        prompt = "镜头 5：A man walks into a room."
        result = extract_by_section_headers(prompt, 5)
        assert result is not None
        assert "镜头 5" in result

    def test_no_match_returns_none(self):
        prompt = "镜头 1: Scene one\n镜头 2: Scene two"
        result = extract_by_section_headers(prompt, 99)
        assert result is None

    def test_english_shot_labels(self):
        prompt = "Shot 1: Opening\nShot 2: Middle\nShot 3: End"
        result = extract_by_section_headers(prompt, 2)
        assert result is not None
        assert "Shot 2" in result


class TestExtractByDialogueKeywords:
    def test_finds_surrounding_context(self):
        shot = FakeShot(lines=[
            FakeLine(dialogue="Are they going crazy"),
            FakeLine(dialogue="Donnie"),
        ])
        prompt = "Scene setup.\nThe man says: no no no they all refused me.\nThen: Are they going crazy Donnie?\nCamera pulls back."
        result = extract_by_dialogue_keywords(prompt, shot)
        assert result is not None
        lowered = result.lower()
        assert "going crazy" in lowered or "donnie" in lowered

    def test_no_match_returns_none(self):
        shot = FakeShot(lines=[FakeLine(dialogue="completely unique")])
        prompt = "Nothing here matches at all."
        result = extract_by_dialogue_keywords(prompt, shot)
        assert result is None


class TestReplaceDialogueInFragment:
    def test_replaces_in_quotes(self):
        fragment = 'He says: "this ceremony is boring" and walks away.'
        result = replace_dialogue_in_fragment(fragment, [
            FakeLine(original="this ceremony is boring", rewritten="This ceremony is not entertaining at all.")
        ])
        assert "This ceremony is not entertaining at all" in result
        assert "this ceremony is boring" not in result


class TestExtractAndRewritePrompt:
    def test_level1_header_extraction(self):
        shot = FakeShot(shot_number=1, scene_description="Opening graduation scene", lines=[
            FakeLine(dialogue="this ceremony is boring", original="this ceremony is boring", rewritten="This ceremony is not entertaining at all.")
        ])
        prompt = '镜头 1：Opening graduation scene\nThe man says "this ceremony is boring" and checks phone.\n镜头 2：Next scene'
        result = extract_and_rewrite_prompt(prompt, shot, shot.lines)
        assert result is not None
        assert "not entertaining" in result.lower() or "This ceremony" in result

    def test_fallback_when_no_match(self):
        shot = FakeShot(shot_number=99, scene_description="A unique scene description", lines=[
            FakeLine(dialogue="test", original="test", rewritten="rewritten test")
        ])
        prompt = "Completely unrelated prompt text."
        result = extract_and_rewrite_prompt(prompt, shot, shot.lines)
        assert result is not None
        assert len(result) > 0
