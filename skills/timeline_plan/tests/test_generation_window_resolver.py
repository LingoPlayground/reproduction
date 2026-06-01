"""Tests for generation_window_resolver.py — atom -> >=4s executable windows."""
from __future__ import annotations

from typing import Optional

from skills.timeline_plan.models import EditAtom, AtomLine, GenerationWindow, CanvasNode
from skills.timeline_plan.generation_window_resolver import resolve_generation_windows


def _make_atom(aid, start, end, lines=None, matched_node: "Optional[str]" = "n1", shot=1, scene="test"):
    return EditAtom(
        atom_id=aid, primary_shot_number=shot, start_sec=start, end_sec=end,
        scene_description=scene, lines=lines or [], shot_numbers=[shot],
        matched_node_id=matched_node, match_confidence=0.9, match_reasoning="test",
    )


def _make_line(lid, original, rewritten, start=0.0, end=1.0):
    return AtomLine(line_id=lid, speaker="S", original=original, rewritten=rewritten,
                    start_sec=start, end_sec=end, shot_scene="test")


class TestResolveGenerationWindows:
    def test_atom_geq_4s_direct_window(self):
        atom = _make_atom("A1", start=0.0, end=5.0, lines=[
            _make_line("L1", "a", "b", start=1.0, end=2.0),
        ])
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="",
                            reference_images=["img.jpg"])]
        windows = resolve_generation_windows(
            atoms=[atom], all_lines=[], canvas_nodes=nodes, video_duration=10.0,
        )
        assert len(windows) == 1
        assert windows[0].duration_sec == 5.0
        assert windows[0].matched_node_id == "n1"

    def test_ref_images_from_canvas_node(self):
        atom = _make_atom("A1", start=0.0, end=5.0, lines=[_make_line("L1", "a", "b")])
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="",
                            reference_images=["img1.jpg", "img2.jpg"])]
        windows = resolve_generation_windows(
            atoms=[atom], all_lines=[], canvas_nodes=nodes, video_duration=10.0,
        )
        assert windows[0].ref_images == ["img1.jpg", "img2.jpg"]

    def test_short_atom_expands_to_min_duration(self):
        atom = _make_atom("A1", start=2.0, end=4.5, lines=[
            _make_line("L1", "a", "b", start=2.5, end=3.5),
        ])
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(
            atoms=[atom], all_lines=[], canvas_nodes=nodes, video_duration=10.0,
        )
        assert windows[0].duration_sec >= 4.0

    def test_short_atom_merge_same_node(self):
        a1 = _make_atom("A1", start=0.0, end=1.5, matched_node="n1", lines=[
            _make_line("L1", "a", "b", start=0.2, end=0.8),
        ])
        a2 = _make_atom("A2", start=2.0, end=3.5, matched_node="n1", lines=[
            _make_line("L2", "c", "d", start=2.2, end=3.0),
        ])
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(
            atoms=[a1, a2], all_lines=[], canvas_nodes=nodes, video_duration=10.0,
        )
        assert len(windows) == 1
        assert len(windows[0].atoms) == 2
        assert windows[0].duration_sec >= 4.0

    def test_short_atom_different_node_stays_separate(self):
        a1 = _make_atom("A1", start=0.0, end=1.5, matched_node="n1", lines=[
            _make_line("L1", "a", "b"),
        ])
        a2 = _make_atom("A2", start=2.0, end=3.5, matched_node="n2", lines=[
            _make_line("L2", "c", "d"),
        ])
        nodes = [
            CanvasNode(node_id="n1", prompt="test1", video_url=""),
            CanvasNode(node_id="n2", prompt="test2", video_url=""),
        ]
        windows = resolve_generation_windows(
            atoms=[a1, a2], all_lines=[], canvas_nodes=nodes, video_duration=10.0,
        )
        assert len(windows) == 2

    def test_unmatched_atom_creates_degraded_window(self):
        atom = _make_atom("A1", start=2.0, end=5.0, matched_node=None, lines=[
            _make_line("L1", "a", "b"),
        ])
        windows = resolve_generation_windows(
            atoms=[atom], all_lines=[], canvas_nodes=[], video_duration=10.0,
        )
        assert len(windows) == 1
        assert windows[0].matched_node_id is None
        assert windows[0].degradation_level > 0
        assert windows[0].ref_images == []

    def test_window_id_unique(self):
        a1 = _make_atom("A1", start=0.0, end=5.0, lines=[_make_line("L1", "a", "b")])
        a2 = _make_atom("A2", start=6.0, end=11.0, lines=[_make_line("L2", "c", "d")])
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(
            atoms=[a1, a2], all_lines=[], canvas_nodes=nodes, video_duration=15.0,
        )
        ids = [w.window_id for w in windows]
        assert len(ids) == len(set(ids))

    def test_empty_atoms_returns_empty(self):
        windows = resolve_generation_windows(atoms=[], all_lines=[], canvas_nodes=[], video_duration=10.0)
        assert windows == []

    def test_overlapping_different_node_windows_snapped(self):
        a1 = _make_atom("A1", start=1.0, end=5.5, matched_node="n1", lines=[_make_line("L1", "a", "b")])
        a2 = _make_atom("A2", start=5.0, end=9.5, matched_node="n2", lines=[_make_line("L2", "c", "d")])
        nodes = [CanvasNode(node_id="n1", prompt="t1", video_url=""), CanvasNode(node_id="n2", prompt="t2", video_url="")]
        windows = resolve_generation_windows(atoms=[a1, a2], all_lines=[], canvas_nodes=nodes, video_duration=10.0)
        sorted_w = sorted(windows, key=lambda w: w.start_sec)
        for i in range(len(sorted_w) - 1):
            assert sorted_w[i].end_sec <= sorted_w[i + 1].start_sec + 0.1
        assert all(w.duration_sec >= 4.0 for w in sorted_w if w.degradation_level < 5)

    def test_tight_different_node_windows_fallback_instead_of_short_modified(self):
        a1 = _make_atom("A1", start=2.0, end=2.5, matched_node="n1", lines=[_make_line("L1", "a", "b")])
        a2 = _make_atom("A2", start=3.0, end=3.5, matched_node="n2", lines=[_make_line("L2", "c", "d")])
        nodes = [CanvasNode(node_id="n1", prompt="t1", video_url=""), CanvasNode(node_id="n2", prompt="t2", video_url="")]
        windows = resolve_generation_windows(atoms=[a1, a2], all_lines=[], canvas_nodes=nodes, video_duration=10.0)
        valid = [w for w in windows if w.degradation_level < 5]
        fallback = [w for w in windows if w.degradation_level >= 5]
        assert len(valid) == 1
        assert valid[0].duration_sec >= 4.0
        assert len(fallback) == 1
        assert "overlap_too_tight" in fallback[0].degradation_reason

    def test_max_duration_enforced(self):
        atoms = []
        for i in range(5):
            atoms.append(_make_atom(f"A{i}", start=i*8.0, end=i*8.0+3.0, matched_node="n1", lines=[_make_line(f"L{i}", "a", "b")]))
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(atoms=atoms, all_lines=[], canvas_nodes=nodes, video_duration=50.0)
        for w in windows:
            assert w.duration_sec <= 30.0, f"Window {w.window_id} exceeds max: {w.duration_sec}"

    def test_asr_boundary_snapping(self):
        atom = _make_atom("A1", start=2.0, end=3.5, matched_node="n1", lines=[_make_line("L1", "a", "b", start=2.5, end=3.0)])
        all_lines = [
            AtomLine(line_id="L2", speaker="S", original="x", rewritten="x", start_sec=4.0, end_sec=4.5, shot_scene="test"),
        ]
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(atoms=[atom], all_lines=all_lines, canvas_nodes=nodes, video_duration=10.0)
        w = windows[0]
        assert not (4.0 < w.end_sec < 4.5), f"Window end {w.end_sec} is inside line L2"

    def test_asr_boundary_snap_keeps_min_duration(self):
        atom_line = _make_line("L1", "a", "b", start=2.0, end=2.5)
        atom = _make_atom("A1", start=2.0, end=2.5, matched_node="n1", lines=[atom_line])
        all_lines = [
            atom_line,
            AtomLine(line_id="U1", speaker="S", original="x", rewritten="x",
                     start_sec=5.5, end_sec=6.5, shot_scene="test"),
        ]
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(atoms=[atom], all_lines=all_lines, canvas_nodes=nodes, video_duration=10.0)
        assert windows[0].duration_sec >= 4.0
        assert not (5.5 < windows[0].end_sec < 6.5)


if __name__ == "__main__":
    import sys
    failed = 0
    tests = TestResolveGenerationWindows()
    for name in sorted(dir(tests)):
        if name.startswith("test_"):
            try:
                getattr(tests, name)()
                print(f"  PASS: {name}")
            except AssertionError as e:
                print(f"  FAIL: {name} - {e}")
                failed += 1
    if failed:
        print(f"\n{failed} FAILED")
        sys.exit(1)
    print("\nAll tests passed!")
