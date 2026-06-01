"""Tests for plan_finalizer.py — GenerationWindow -> TimelinePlan assembly."""
from skills.timeline_plan.models import EditAtom, AtomLine, GenerationWindow, CanvasNode, TimelinePlan
from skills.timeline_plan.plan_finalizer import finalize_timeline_plan


def _make_atom(aid, lines, shot=1, scene="test"):
    return EditAtom(atom_id=aid, primary_shot_number=shot, start_sec=0.0, end_sec=3.0,
                    scene_description=scene, lines=lines, shot_numbers=[shot])


def _make_line(lid, original, rewritten, speaker="Mia"):
    return AtomLine(line_id=lid, speaker=speaker, original=original,
                    rewritten=rewritten, start_sec=0.0, end_sec=1.0)


def _make_window(wid, start, end, atoms, node_id="n1", prompt="test prompt", ref_images=None, degradation=0):
    return GenerationWindow(window_id=wid, start_sec=start, end_sec=end, atoms=atoms,
                            matched_node_id=node_id, match_confidence=0.9,
                            rewritten_prompt=prompt, ref_images=ref_images or [],
                            degradation_level=degradation, degradation_reason="")


class FakeShot:
    def __init__(self, shot_number, start, end, scene_desc=""):
        self.shot_number = shot_number
        self.start_seconds = start
        self.end_seconds = end
        self.scene_description = scene_desc


class TestFinalizeTimelinePlan:
    def test_single_window_produces_modified_and_originals(self):
        window = _make_window("W1", start=4.0, end=8.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")], shot=1),
        ])
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        plan = finalize_timeline_plan(windows=[window], shots=shots, video_duration=10.0, title="Test", level="B2")
        modified = [i for i in plan.items if i.source == "modified"]
        original = [i for i in plan.items if i.source == "original"]
        assert len(modified) == 1
        assert modified[0].start_sec == 4.0
        assert modified[0].end_sec == 8.0
        assert modified[0].rewritten_prompt == "test prompt"
        assert modified[0].covered_line_ids == ["L1"]
        assert plan.total_duration_sec == 10.0
        assert plan.title == "Test"

    def test_modified_item_has_ref_images(self):
        window = _make_window("W1", start=2.0, end=6.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ], ref_images=["img1.jpg"])
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(windows=[window], shots=shots, video_duration=10.0, title="T", level="A2")
        modified = [i for i in plan.items if i.source == "modified"]
        assert modified[0].ref_images == ["img1.jpg"]

    def test_modified_item_has_degradation(self):
        window = _make_window("W1", start=2.0, end=6.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ], degradation=1)
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(windows=[window], shots=shots, video_duration=10.0, title="T", level="B2")
        modified = [i for i in plan.items if i.source == "modified"]
        assert modified[0].degradation_level == 1

    def test_no_overlap_in_output(self):
        window = _make_window("W1", start=3.0, end=7.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ])
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(windows=[window], shots=shots, video_duration=10.0, title="T", level="B2")
        sorted_items = sorted(plan.items, key=lambda i: i.start_sec)
        for i in range(len(sorted_items) - 1):
            assert sorted_items[i].end_sec <= sorted_items[i + 1].start_sec + 0.1

    def test_full_coverage_zero_to_duration(self):
        window = _make_window("W1", start=2.0, end=5.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ])
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(windows=[window], shots=shots, video_duration=10.0, title="T", level="B2")
        sorted_items = sorted(plan.items, key=lambda i: i.start_sec)
        assert sorted_items[0].start_sec <= 0.1
        assert sorted_items[-1].end_sec >= 9.9

    def test_no_rewritten_lines_returns_all_original(self):
        shots = [FakeShot(1, 0.0, 5.0, "scene")]
        plan = finalize_timeline_plan(windows=[], shots=shots, video_duration=5.0, title="T", level="B2")
        assert len(plan.items) == 1
        assert plan.items[0].source == "original"

    def test_unmatched_window_becomes_original(self):
        window = GenerationWindow(
            window_id="W1", start_sec=2.0, end_sec=6.0, atoms=[],
            degradation_level=5, degradation_reason="unmatched_atom",
        )
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(windows=[window], shots=shots, video_duration=10.0, title="T", level="B2")
        modified = [i for i in plan.items if i.source == "modified"]
        assert len(modified) == 0
        assert all(i.source == "original" for i in plan.items)
