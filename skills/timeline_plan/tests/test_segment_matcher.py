"""Tests for segment_matcher.py — EditAtom -> Canvas Node matching."""
import json
from unittest.mock import patch, MagicMock
from skills.timeline_plan.models import EditAtom, AtomLine, CanvasNode, WindowPlanDraft
from skills.timeline_plan.segment_matcher import (
    match_atoms_to_nodes,
    _build_matching_prompt,
    _parse_match_response,
)


def _make_atom(aid, lines, scene="kitchen", primary_shot=1):
    return EditAtom(
        atom_id=aid, primary_shot_number=primary_shot, start_sec=0.0, end_sec=3.0,
        scene_description=scene, lines=lines, shot_numbers=[primary_shot],
    )


def _make_line(lid, original, rewritten, speaker="Mia", start=0.0, end=1.0):
    return AtomLine(line_id=lid, speaker=speaker, original=original,
                    rewritten=rewritten, start_sec=start, end_sec=end)


def _make_node(nid, prompt, ref_images=None):
    return CanvasNode(node_id=nid, prompt=prompt, video_url="",
                      reference_images=ref_images or [])


class TestBuildMatchingPrompt:
    def test_includes_atom_dialogue(self):
        atom = _make_atom("A1", [_make_line("L1", "hello", "hi")])
        nodes = [_make_node("n1", "Scene: hello world")]
        prompt = _build_matching_prompt([atom], nodes)
        assert "hello" in prompt
        assert "hi" in prompt
        assert "n1" in prompt
        assert "A1" in prompt

    def test_includes_scene_description(self):
        atom = _make_atom("A1", [_make_line("L1", "a", "b")], scene="classroom")
        nodes = [_make_node("n1", "classroom scene")]
        prompt = _build_matching_prompt([atom], nodes)
        assert "classroom" in prompt

    def test_includes_canvas_node_prompts(self):
        atom = _make_atom("A1", [_make_line("L1", "x", "y")])
        nodes = [_make_node("n1", "cats"), _make_node("n2", "dogs")]
        prompt = _build_matching_prompt([atom], nodes)
        assert "cats" in prompt
        assert "dogs" in prompt


class TestParseMatchResponse:
    def test_parses_valid_response(self):
        response = json.dumps({
            "matches": [{"atom_id": "A1", "node_id": "n1", "confidence": 0.9, "reasoning": "good"}],
            "window_drafts": [{"draft_id": "D1", "atom_ids": ["A1"], "node_id": "n1", "confidence": 0.9}],
            "unmatched": [],
        })
        matches, unmatched, drafts = _parse_match_response(response)
        assert len(matches) == 1
        assert matches[0]["atom_id"] == "A1"
        assert len(unmatched) == 0
        assert drafts[0]["atom_ids"] == ["A1"]

    def test_parses_unmatched(self):
        response = json.dumps({
            "matches": [],
            "unmatched": [{"atom_id": "A2", "reason": "no match"}],
        })
        matches, unmatched, drafts = _parse_match_response(response)
        assert len(matches) == 0
        assert len(unmatched) == 1
        assert drafts == []

    def test_parses_markdown_fenced_json(self):
        response = '```json\n{"matches": [], "unmatched": []}\n```'
        matches, unmatched, _ = _parse_match_response(response)
        assert matches == []
        assert unmatched == []

    def test_returns_empty_on_invalid(self):
        matches, unmatched, drafts = _parse_match_response("not json at all")
        assert matches == []
        assert unmatched == []
        assert drafts == []


class TestMatchAtomsToNodes:
    @patch("skills.timeline_plan.segment_matcher._get_client")
    def test_populates_matched_node(self, mock_get_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "matches": [{"atom_id": "A1", "node_id": "n1", "confidence": 0.85, "reasoning": "match"}],
            "window_drafts": [{"draft_id": "D1", "atom_ids": ["A1"], "node_id": "n1", "confidence": 0.8, "reasoning": "one intent"}],
            "unmatched": [],
        })
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        atom = _make_atom("A1", [_make_line("L1", "hello", "hi")])
        nodes = [_make_node("n1", "hello scene")]
        drafts = match_atoms_to_nodes([atom], nodes)

        assert atom.matched_node_id == "n1"
        assert atom.match_confidence == 0.85
        assert len(drafts) == 1
        assert isinstance(drafts[0], WindowPlanDraft)
        assert drafts[0].atom_ids == ["A1"]

    @patch("skills.timeline_plan.segment_matcher._get_client")
    def test_unmatched_atom_stays_none(self, mock_get_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "matches": [],
            "unmatched": [{"atom_id": "A1", "reason": "no canvas matches"}],
        })
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        atom = _make_atom("A1", [_make_line("L1", "hello", "hi")])
        drafts = match_atoms_to_nodes([atom], [])
        assert atom.matched_node_id is None
        assert drafts == []

    def test_empty_atoms_noop(self):
        assert match_atoms_to_nodes([], [_make_node("n1", "test")]) == []
        # should not raise

    @patch("skills.timeline_plan.segment_matcher._get_client")
    def test_missing_window_drafts_falls_back_to_single_atom_drafts(self, mock_get_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "matches": [{"atom_id": "A1", "node_id": "n1", "confidence": 0.75, "reasoning": "match"}],
            "unmatched": [],
        })
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        atom = _make_atom("A1", [_make_line("L1", "hello", "hi")])
        drafts = match_atoms_to_nodes([atom], [_make_node("n1", "hello")])

        assert len(drafts) == 1
        assert drafts[0].atom_ids == ["A1"]
        assert drafts[0].node_id == "n1"

    @patch("skills.timeline_plan.segment_matcher._get_client")
    def test_window_drafts_must_match_atom_node(self, mock_get_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "matches": [{"atom_id": "A1", "node_id": "n1", "confidence": "0.75", "reasoning": "match"}],
            "window_drafts": [{"draft_id": "D1", "atom_ids": ["A1"], "node_id": "n2", "confidence": "high"}],
            "unmatched": [],
        })
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        atom = _make_atom("A1", [_make_line("L1", "hello", "hi")])
        drafts = match_atoms_to_nodes([atom], [_make_node("n1", "hello"), _make_node("n2", "other")])

        assert atom.matched_node_id == "n1"
        assert atom.match_confidence == 0.75
        assert len(drafts) == 1
        assert drafts[0].atom_ids == ["A1"]
        assert drafts[0].node_id == "n1"


class TestCoarseRecall:
    def test_coarse_recall_finds_candidates(self):
        from skills.timeline_plan.segment_matcher import _coarse_recall
        atom = _make_atom("A1", [_make_line("L1", "I am calling the police.", "rewritten")])
        nodes = [
            _make_node("n1", "office scene with police investigation"),
            _make_node("n2", "kitchen cooking show"),
        ]
        result = _coarse_recall([atom], nodes)
        assert "n1" in result.get("A1", [])


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestBuildMatchingPrompt, TestParseMatchResponse, TestMatchAtomsToNodes, TestCoarseRecall]:
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
