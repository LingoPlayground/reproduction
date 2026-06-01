"""Tests for prompt_rewriter.py — window-level prompt rewriting."""
from unittest.mock import patch, MagicMock
from skills.timeline_plan.models import EditAtom, AtomLine, GenerationWindow, CanvasNode
from skills.timeline_plan.prompt_rewriter import (
    rewrite_prompts_for_windows,
    _make_rewrite_prompt,
    _check_rewritten_prompt,
)


def _make_atom(aid, lines, shot=1):
    return EditAtom(atom_id=aid, primary_shot_number=shot, start_sec=0.0, end_sec=3.0,
                    scene_description="test", lines=lines, shot_numbers=[shot])


def _make_line(lid, original, rewritten, speaker="Mia"):
    return AtomLine(line_id=lid, speaker=speaker, original=original,
                    rewritten=rewritten, start_sec=0.0, end_sec=1.0)


def _make_window(wid, atoms, node_id="n1"):
    return GenerationWindow(window_id=wid, start_sec=0.0, end_sec=5.0, atoms=atoms,
                            matched_node_id=node_id)


class TestMakeRewritePrompt:
    def test_includes_rewritten_lines(self):
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "hello", "hi there")])])
        node = CanvasNode(node_id="n1", prompt="A scene: hello", video_url="")
        prompt = _make_rewrite_prompt(window, node, "B2")
        assert "hello" in prompt
        assert "hi there" in prompt

    def test_includes_level(self):
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "a", "b")])])
        node = CanvasNode(node_id="n1", prompt="test", video_url="")
        prompt = _make_rewrite_prompt(window, node, "B2")
        assert "B2" in prompt


class TestCheckRewrittenPrompt:
    def test_all_lines_present(self):
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "hello", "hi")])])
        errors = _check_rewritten_prompt(window, "Someone says: hi")
        assert len(errors) == 0

    def test_missing_line_reported(self):
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "hello", "uniquephrase123")])])
        errors = _check_rewritten_prompt(window, "Someone says: hi")
        assert len(errors) == 1
        assert "uniquephrase123" in errors[0]

    def test_unchanged_line_not_required(self):
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "hello", "hello")])])
        errors = _check_rewritten_prompt(window, "Some prompt without hello")
        assert len(errors) == 0


class TestRewritePromptsForWindows:
    @patch("skills.timeline_plan.prompt_rewriter._get_client")
    def test_sets_rewritten_prompt(self, mock_get_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "A rewritten scene: hi there"
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "hello", "hi there")])])
        nodes = [CanvasNode(node_id="n1", prompt="A scene: hello", video_url="")]
        rewrite_prompts_for_windows([window], nodes, "B2")
        assert window.rewritten_prompt == "A rewritten scene: hi there"

    def test_no_client_skips(self):
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "a", "b")])])
        rewrite_prompts_for_windows([window], [], "B2")
        assert window.rewritten_prompt is None

    def test_empty_windows_noop(self):
        rewrite_prompts_for_windows([], [], "B2")

    @patch("skills.timeline_plan.prompt_rewriter._get_client")
    def test_degraded_fallback_window_skips_llm(self, mock_get_client):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        window = _make_window("W1", [_make_atom("A1", [_make_line("L1", "a", "b")])])
        window.degradation_level = 5
        nodes = [CanvasNode(node_id="n1", prompt="A scene: a", video_url="")]

        rewrite_prompts_for_windows([window], nodes, "B2")

        assert window.rewritten_prompt is None
        mock_client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestMakeRewritePrompt, TestCheckRewrittenPrompt, TestRewritePromptsForWindows]:
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
