"""Tests for llm_planner.py — JSON parsing."""
from skills.timeline_plan.llm_planner import _parse_json, _make_coarse_prompt, _make_rewrite_prompt
from skills.timeline_plan.evidence_builder import build_evidence
from skills.timeline_plan.models import CanvasNode
from skills.timeline_plan.planner_models import RewriteInput


class TestParseJson:
    def test_plain(self):
        assert _parse_json('{"a": 1}') == {"a": 1}
    def test_fences(self):
        assert _parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    def test_empty(self):
        assert _parse_json("") is None
    def test_invalid(self):
        assert _parse_json("not json") is None


class TestPrompts:
    def test_coarse_prompt_has_dialogue(self):
        nodes = [CanvasNode(node_id='n1', prompt='test prompt', video_url='', reference_images=[])]
        lines = [{'line_id':'L1','original':'hello','rewritten':'hi','speaker':'D','start_seconds':1,'end_seconds':2,'shot_number':1,'shot_scene':'s1'}]
        evidence = build_evidence([], lines, nodes, [])
        p = _make_coarse_prompt(evidence)
        assert 'hello' in p and 'n1' in p and 'L1' in p

    def test_rewrite_prompt_has_change_lines(self):
        ri = RewriteInput(group_id='G1', original_prompt='Donny says: \"hello\"',
            covered_lines=[{'line_id':'L1','original':'hello','rewritten':'hi','speaker':'D'}])
        p = _make_rewrite_prompt(ri)
        assert 'Lines to Change' in p and 'hello' in p and 'hi' in p


if __name__ == "__main__":
    import sys; failed = 0
    for cls in [TestParseJson, TestPrompts]:
        for name in sorted(dir(cls())):
            if name.startswith("test_"):
                try: getattr(cls(), name)(); print(f"  PASS: {name}")
                except Exception as e: print(f"  FAIL: {name} - {e}"); failed += 1
    sys.exit(failed)
