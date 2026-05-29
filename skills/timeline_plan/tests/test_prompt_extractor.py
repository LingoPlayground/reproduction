"""Tests for prompt fragment extraction."""
from skills.timeline_plan.prompt_extractor import (
    _find_section_boundaries,
    _find_lines_containing_dialogue,
    extract_prompt_fragment_for_lines,
    replace_dialogue_in_fragment,
    extract_and_rewrite_prompt,
)


class FakeLine:
    def __init__(self, dialogue="", original="", rewritten="", speaker="Speaker"):
        self.dialogue = dialogue
        self.original = original
        self.rewritten = rewritten
        self.speaker = speaker


class TestFindSectionBoundaries:
    def test_chinese_shot_labels(self):
        prompt = "Some intro\n镜头 1：Opening scene\n镜头 2：Conflict scene\n镜头 3：Resolution"
        boundaries = _find_section_boundaries(prompt)
        assert len(boundaries) >= 3  # 0 + 镜头1 + 镜头2 + 镜头3

    def test_english_shot_labels(self):
        prompt = "Shot 1: Opening\nShot 2: Middle\nShot 3: End"
        boundaries = _find_section_boundaries(prompt)
        assert len(boundaries) >= 3

    def test_no_boundaries(self):
        prompt = "Just plain text\nwithout any headers.\nNothing here."
        boundaries = _find_section_boundaries(prompt)
        assert boundaries == [0]  # Only the implicit section 0

    def test_markdown_headings(self):
        prompt = "text before\n## Section One\ntext\n### Section Two"
        boundaries = _find_section_boundaries(prompt)
        assert len(boundaries) >= 2


class TestFindLinesContainingDialogue:
    def test_finds_matching_lines(self):
        prompt = "Scene setup.\nThe man says: Are they going crazy Donnie?\nCamera pulls back."
        fragments = ["Are they going crazy"]
        hits = _find_lines_containing_dialogue(prompt, fragments)
        assert len(hits) > 0
        assert 1 in hits  # Line index 1

    def test_no_match(self):
        prompt = "Nothing here matches at all."
        fragments = ["completely unique"]
        hits = _find_lines_containing_dialogue(prompt, fragments)
        assert len(hits) == 0

    def test_normalized_matching(self):
        prompt = "He says, \"This. Ceremony. Is. Boring!\""
        fragments = ["this ceremony is boring"]
        hits = _find_lines_containing_dialogue(prompt, fragments)
        assert len(hits) > 0


class TestExtractPromptFragmentForLines:
    def test_extracts_section_with_dialogue(self):
        prompt = (
            "镜头 1：Opening graduation scene\n"
            'The man says "this ceremony is boring" and checks phone.\n'
            "镜头 2：Next scene with different characters\n"
            "Different dialogue here."
        )
        target = [FakeLine(original="this ceremony is boring")]
        fragment = extract_prompt_fragment_for_lines(prompt, target)
        assert fragment is not None
        assert "镜头 1" in fragment
        assert "this ceremony is boring" in fragment.lower()

    def test_extracts_only_relevant_section(self):
        prompt = (
            "镜头 1：Scene one dialogue\nShot 1 content here.\n"
            "镜头 2：Scene two dialogue\n"
            'He says "some other words" and walks away.\n'
            "镜头 3：Scene three\nFinal scene content."
        )
        target = [FakeLine(original="some other words")]
        fragment = extract_prompt_fragment_for_lines(prompt, target)
        assert fragment is not None
        assert "镜头 2" in fragment
        assert "镜头 1" not in fragment
        assert "镜头 3" not in fragment

    def test_context_window_fallback(self):
        prompt = (
            "Line 0\nLine 1\nLine 2\n"
            'The man says "unique dialogue here"\n'
            "Line 4\nLine 5\nLine 6"
        )
        target = [FakeLine(original="unique dialogue here")]
        fragment = extract_prompt_fragment_for_lines(prompt, target, context_lines=2)
        assert fragment is not None
        assert "unique dialogue here" in fragment.lower()
        # Should include ~2 lines of context before/after
        assert "Line 1" in fragment

    def test_no_dialogue_match_returns_none(self):
        prompt = "Nothing here at all."
        target = [FakeLine(original="completely missing dialogue")]
        fragment = extract_prompt_fragment_for_lines(prompt, target)
        assert fragment is None

    def test_empty_target_lines(self):
        prompt = "Some prompt text."
        fragment = extract_prompt_fragment_for_lines(prompt, [])
        assert fragment is None

    def test_finds_original_fallback_from_dialogue_attr(self):
        """Should use .original first, then fall back to .dialogue."""
        prompt = 'He says "hello world"'
        # Line with .original set but dialogue empty
        target = [FakeLine(dialogue="", original="hello world")]
        fragment = extract_prompt_fragment_for_lines(prompt, target)
        assert fragment is not None
        assert "hello world" in fragment.lower()

    def test_multiple_dialogue_fragments(self):
        prompt = (
            "镜头 1\nFirst line of dialogue with key phrase.\n"
            "Also has another important sentence here.\n"
            "镜头 2\nDifferent content."
        )
        target = [
            FakeLine(original="key phrase"),
            FakeLine(original="important sentence"),
        ]
        fragment = extract_prompt_fragment_for_lines(prompt, target)
        assert fragment is not None
        assert "镜头 1" in fragment

    def test_multi_section_span(self):
        """When dialogue hits span multiple sections, return the union."""
        prompt = (
            "镜头 1：First section with dialogue one.\n"
            "More section 1 content.\n"
            "镜头 2：Second section with dialogue two.\n"
            "More section 2 content.\n"
            "镜头 3：Third section without target dialogue."
        )
        target = [
            FakeLine(original="dialogue one"),
            FakeLine(original="dialogue two"),
        ]
        fragment = extract_prompt_fragment_for_lines(prompt, target)
        assert fragment is not None
        assert "镜头 1" in fragment
        assert "镜头 2" in fragment
        assert "镜头 3" not in fragment


class TestReplaceDialogueInFragment:
    def test_replaces_in_quotes(self):
        fragment = 'He says: "this ceremony is boring" and walks away.'
        result = replace_dialogue_in_fragment(fragment, [
            FakeLine(original="this ceremony is boring", rewritten="This ceremony is not entertaining at all.")
        ])
        assert "This ceremony is not entertaining at all" in result
        assert "this ceremony is boring" not in result

    def test_replaces_after_colon(self):
        fragment = "Character A: hello there my friend."
        result = replace_dialogue_in_fragment(fragment, [
            FakeLine(original="hello there", rewritten="greetings")
        ])
        assert "greetings" in result

    def test_exact_substring_fallback(self):
        fragment = "The character says hello world quietly."
        result = replace_dialogue_in_fragment(fragment, [
            FakeLine(original="hello world", rewritten="goodbye world")
        ])
        assert "goodbye world" in result
        assert "hello world" not in result

    def test_no_change_when_same_text(self):
        fragment = 'He says "hello".'
        result = replace_dialogue_in_fragment(fragment, [
            FakeLine(original="hello", rewritten="hello")
        ])
        assert result == fragment


class TestExtractAndRewritePrompt:
    def test_content_driven_extraction_with_rewrite(self):
        prompt = (
            '镜头 1：Opening scene\n'
            'The man says "this ceremony is boring" and checks phone.\n'
            '镜头 2：Next scene'
        )
        lines = [FakeLine(original="this ceremony is boring", rewritten="This ceremony is not entertaining at all.")]
        result = extract_and_rewrite_prompt(prompt, lines, scene_description="Opening")
        assert result is not None
        assert "not entertaining" in result.lower()

    def test_fallback_when_no_prompt(self):
        """Empty prompt → generates from scene description."""
        lines = [FakeLine(original="test", rewritten="rewritten test")]
        result = extract_and_rewrite_prompt("", lines, scene_description="A unique scene description")
        assert result is not None
        assert len(result) > 0
        assert "A unique scene description" in result or "rewritten test" in result

    def test_fallback_when_no_dialogue_match(self):
        """Prompt exists but no dialogue match → fallback to scene generation."""
        lines = [FakeLine(original="unique text not in prompt", rewritten="rewritten")]
        result = extract_and_rewrite_prompt("completely different prompt", lines, "Scene context")
        assert result is not None
        assert len(result) > 0

    def test_no_duplicate_items(self):
        """Verify that unchanged lines don't create replacement cruft."""
        prompt = 'He says "hello".'
        lines = [FakeLine(original="hello", rewritten="hello")]
        result = extract_and_rewrite_prompt(prompt, lines)
        assert result is not None
        assert "hello" in result.lower()
