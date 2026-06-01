"""Tests for models.py — v3.0 deterministic execution models."""
from skills.timeline_plan.models import (
    CutPoint, AtomLine, EditAtom, GenerationWindow,
    CanvasNode, TimelinePlanItem, TimelinePlan,
    Stage3Input, MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
)


class TestTimelinePlanItem:
    def test_original_source(self):
        item = TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                                start_sec=0.0, end_sec=5.0, scene_description="test")
        assert item.source == "original"
        assert item.duration_sec == 5.0
        assert item.ref_images == []
        assert item.rewritten_prompt is None

    def test_modified_source(self):
        item = TimelinePlanItem(shot_id="s1", shot_number=1, source="modified",
                                start_sec=10.0, end_sec=15.0, scene_description="test",
                                rewritten_prompt="new prompt", matched_node_id="n1",
                                match_confidence=0.9, covered_line_ids=["L1", "L2"],
                                source_node_ids=["n1"])
        assert item.source == "modified"
        assert item.rewritten_prompt == "new prompt"
        assert item.covered_line_ids == ["L1", "L2"]
        assert item.source_node_ids == ["n1"]
        assert item.match_confidence == 0.9

    def test_duration_sec_zero_clamped(self):
        item = TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                                start_sec=5.0, end_sec=3.0, scene_description="test")
        assert item.duration_sec == 0.0


class TestTimelinePlan:
    def test_no_version_field(self):
        plan = TimelinePlan(title="Test", level="B2")
        assert plan.items == []

    def test_with_items(self):
        items = [
            TimelinePlanItem(shot_id="s1", shot_number=1, source="original",
                             start_sec=0.0, end_sec=5.0, scene_description="s1"),
            TimelinePlanItem(shot_id="s2", shot_number=2, source="modified",
                             start_sec=5.0, end_sec=10.0, scene_description="s2"),
        ]
        plan = TimelinePlan(title="Test", level="B2", items=items)
        assert len(plan.items) == 2
        assert plan.items[0].source == "original"
        assert plan.items[1].source == "modified"


class TestStage3Input:
    def test_defaults(self):
        inp = Stage3Input(script_output=None)
        assert inp.level == "B2"
        assert inp.canvas_nodes == []


class TestConstants:
    def test_duration_constants(self):
        assert MIN_MODIFIED_DURATION == 4.0
        assert MAX_MODIFIED_DURATION == 30.0


class TestCutPoint:
    def test_no_confidence_field(self):
        cp = CutPoint(time_sec=5.0)
        assert cp.time_sec == 5.0
        assert not hasattr(cp, 'confidence')


class TestAtomLine:
    def test_is_rewritten_true(self):
        line = AtomLine(line_id="L1", speaker="A", original="hello", rewritten="hi",
                        start_sec=1.0, end_sec=2.0)
        assert line.is_rewritten is True
        assert line.shot_scene == ""

    def test_is_rewritten_false(self):
        line = AtomLine(line_id="L1", speaker="A", original="hello", rewritten="hello",
                        start_sec=1.0, end_sec=2.0)
        assert line.is_rewritten is False

    def test_is_rewritten_punctuation_only_diff(self):
        line = AtomLine(line_id="L1", speaker="A", original="Hello.", rewritten="Hello!",
                        start_sec=1.0, end_sec=2.0)
        assert line.is_rewritten is False

    def test_is_rewritten_case_insensitive(self):
        line = AtomLine(line_id="L1", speaker="A", original="HELLO", rewritten="hello",
                        start_sec=1.0, end_sec=2.0)
        assert line.is_rewritten is False

    def test_with_shot_scene(self):
        line = AtomLine(line_id="L1", speaker="A", original="a", rewritten="b",
                        start_sec=1.0, end_sec=2.0, shot_scene="kitchen")
        assert line.shot_scene == "kitchen"


class TestEditAtom:
    def _make_line(self, lid, original, rewritten, start=0.0, end=1.0):
        return AtomLine(line_id=lid, speaker="S", original=original, rewritten=rewritten,
                        start_sec=start, end_sec=end)

    def _make_atom(self, atom_id="A1", lines=None, primary_shot_number=1,
                   start_sec=0.0, end_sec=5.0, scene_description="test",
                   shot_numbers=None, matched_node_id=None,
                   match_confidence=None, match_reasoning="",
                   boundary_reason="", source_cut_times=None):
        return EditAtom(
            atom_id=atom_id, primary_shot_number=primary_shot_number,
            start_sec=start_sec, end_sec=end_sec,
            scene_description=scene_description,
            shot_numbers=shot_numbers if shot_numbers is not None else [1],
            lines=lines or [], matched_node_id=matched_node_id,
            match_confidence=match_confidence, match_reasoning=match_reasoning,
            boundary_reason=boundary_reason,
            source_cut_times=source_cut_times if source_cut_times is not None else [],
        )

    def test_has_rewritten_lines_true(self):
        atom = self._make_atom(lines=[
            self._make_line("L1", "a", "b"),
            self._make_line("L2", "x", "x"),
        ])
        assert atom.has_rewritten_lines is True

    def test_has_rewritten_lines_false(self):
        atom = self._make_atom(lines=[self._make_line("L1", "a", "a")])
        assert atom.has_rewritten_lines is False

    def test_rewritten_lines_only_returns_changed(self):
        atom = self._make_atom(lines=[
            self._make_line("L1", "a", "b"),
            self._make_line("L2", "x", "x"),
        ])
        rw = atom.rewritten_lines
        assert len(rw) == 1
        assert rw[0].line_id == "L1"

    def test_duration_sec(self):
        atom = self._make_atom(start_sec=3.0, end_sec=7.0)
        assert atom.duration_sec == 4.0

    def test_duration_sec_zero_clamped(self):
        atom = self._make_atom(start_sec=5.0, end_sec=3.0)
        assert atom.duration_sec == 0.0

    def test_matched_node_defaults(self):
        atom = self._make_atom()
        assert atom.matched_node_id is None
        assert atom.match_confidence is None
        assert atom.match_reasoning == ""

    def test_debug_fields_default(self):
        atom = self._make_atom()
        assert atom.boundary_reason == ""
        assert atom.source_cut_times == []


class TestGenerationWindow:
    def _make_line(self, lid, original, rewritten):
        return AtomLine(line_id=lid, speaker="S", original=original, rewritten=rewritten,
                        start_sec=0.0, end_sec=1.0)

    def _make_atom(self, atom_id="A1", lines=None):
        return EditAtom(atom_id=atom_id, primary_shot_number=1,
                        start_sec=0.0, end_sec=5.0, scene_description="test",
                        shot_numbers=[1], lines=lines or [])

    def test_covered_line_ids(self):
        atom = self._make_atom(lines=[
            self._make_line("L1", "a", "b"),
            self._make_line("L2", "x", "x"),
            self._make_line("L3", "c", "d"),
        ])
        window = GenerationWindow(window_id="W1", start_sec=0.0, end_sec=6.0, atoms=[atom])
        assert window.covered_line_ids == ["L1", "L3"]

    def test_covered_line_ids_multiple_atoms(self):
        a1 = self._make_atom("A1", lines=[self._make_line("L1", "a", "b")])
        a2 = self._make_atom("A2", lines=[self._make_line("L2", "c", "d")])
        window = GenerationWindow(window_id="W1", start_sec=0.0, end_sec=10.0, atoms=[a1, a2])
        assert window.covered_line_ids == ["L1", "L2"]

    def test_duration_sec(self):
        window = GenerationWindow(window_id="W1", start_sec=3.0, end_sec=7.0, atoms=[])
        assert window.duration_sec == 4.0

    def test_duration_sec_zero_clamped(self):
        window = GenerationWindow(window_id="W1", start_sec=5.0, end_sec=3.0, atoms=[])
        assert window.duration_sec == 0.0

    def test_defaults(self):
        window = GenerationWindow(window_id="W1", start_sec=0.0, end_sec=4.0, atoms=[])
        assert window.matched_node_id is None
        assert window.rewritten_prompt is None
        assert window.ref_images == []
        assert window.degradation_level == 0
        assert window.degradation_reason == ""


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestCutPoint, TestAtomLine, TestEditAtom, TestGenerationWindow,
            TestTimelinePlanItem, TestTimelinePlan, TestStage3Input, TestConstants]:
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
