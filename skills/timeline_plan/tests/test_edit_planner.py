"""Tests for evidence builder and edit planner."""
from skills.timeline_plan.models import CanvasNode, EvidencePack
from skills.timeline_plan.evidence_builder import build_evidence_pack
from skills.timeline_plan.edit_planner import plan_edit, _fallback_plan, _fuzzy_word_match


def _group(entries):
    return entries


class TestEvidenceBuilder:
    def test_builds_pack_from_rewrite_lines(self):
        lines = [{
            "line_id": "p001_l001", "speaker": "Donny",
            "original": "hello", "rewritten": "hi",
            "start_seconds": 1.0, "end_seconds": 2.0,
            "shot_number": 1, "shot_scene": "a scene",
        }]
        node = CanvasNode(
            node_id="n1", prompt="A prompt", video_url="http://x.com/v.mp4",
            reference_images=["http://x.com/r.png"],
        )
        pack = build_evidence_pack(
            group_id="g1", rewrite_lines=lines, all_lines_map={},
            node=node,
        )
        assert isinstance(pack, EvidencePack)
        assert pack.group_id == "g1"
        assert len(pack.target_lines) == 1
        assert pack.target_lines[0].line_id == "p001_l001"
        assert pack.canvas_node is not None
        assert pack.canvas_node.node_id == "n1"

    def test_builds_pack_without_node(self):
        lines = [{
            "line_id": "p001_l001", "speaker": "Donny",
            "original": "hello", "rewritten": "hi",
            "start_seconds": 1.0, "end_seconds": 2.0,
            "shot_number": 1, "shot_scene": "",
        }]
        pack = build_evidence_pack(
            group_id="g1", rewrite_lines=lines, all_lines_map={},
            node=None,
        )
        assert pack.canvas_node is None
        assert pack.group_id == "g1"

    def test_includes_neighbor_lines(self):
        lines = [{
            "line_id": "l1", "speaker": "A",
            "original": "hello", "rewritten": "hi",
            "start_seconds": 1.0, "end_seconds": 2.0,
            "shot_number": 1, "shot_scene": "",
        }]
        all_lines = {
            "l2": {"line_id": "l2", "speaker": "B", "dialogue": "world",
                    "start_seconds": 3.0, "end_seconds": 4.0},
        }
        pack = build_evidence_pack(
            group_id="g1", rewrite_lines=lines, all_lines_map=all_lines,
            node=None,
        )
        assert len(pack.neighbor_lines) == 1
        assert pack.neighbor_lines[0].line_id == "l2"


class TestEditPlanner:
    def test_fallback_returns_operation_type(self):
        from types import SimpleNamespace

        target = SimpleNamespace(
            line_id="l1", speaker="A", original="hello", rewritten="hi",
            start_seconds=1.0, end_seconds=2.0, shot_number=1,
            shot_scene="", rewrite_status="rewritten",
        )
        node_evidence = SimpleNamespace(
            node_id="n1", name="n1", full_prompt='He says "hello"',
            sections=[], reference_images=[], node_video_url=None,
        )
        pack = SimpleNamespace(
            group_id="g1", target_lines=[target], neighbor_lines=[],
            canvas_node=node_evidence, matched_section_id=None,
            video=None, constraints=None,
        )
        plan = _fallback_plan(pack)
        assert plan["operation_type"] == "literal_replace"

    def test_fallback_for_implicit_dialogue_is_semantic_insert(self):
        from types import SimpleNamespace

        target = SimpleNamespace(
            line_id="l1", speaker="A", original="no no no",
            rewritten="No, no, no!", start_seconds=1.0, end_seconds=2.0,
            shot_number=1, shot_scene="", rewrite_status="rewritten",
        )
        node_evidence = SimpleNamespace(
            node_id="n1", name="n1", full_prompt="美式情景喜剧，真实的破防",
            sections=[], reference_images=[], node_video_url=None,
        )
        pack = SimpleNamespace(
            group_id="g1", target_lines=[target], neighbor_lines=[],
            canvas_node=node_evidence, matched_section_id=None,
            video=None, constraints=None,
        )
        plan = _fallback_plan(pack)
        assert plan["operation_type"] == "semantic_insert"

    def test_fallback_without_node(self):
        from types import SimpleNamespace
        pack = SimpleNamespace(
            group_id="g1", target_lines=[], neighbor_lines=[],
            canvas_node=None, matched_section_id=None,
            video=None, constraints=None,
        )
        plan = _fallback_plan(pack)
        assert plan["operation_type"] == "full_fallback"


class TestFuzzyWordMatch:
    def test_exact_match(self):
        assert _fuzzy_word_match("hello world", "Donny says: hello world")

    def test_punctuation_drift(self):
        assert _fuzzy_word_match("no no no", 'Donny says: "no, no, no!"')

    def test_partial_match_below_threshold(self):
        assert not _fuzzy_word_match("hello world", "Donny says: goodbye")

    def test_short_words_ignored(self):
        assert not _fuzzy_word_match("no no", "nothing here")
