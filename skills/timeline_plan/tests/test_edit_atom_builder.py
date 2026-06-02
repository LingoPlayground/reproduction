"""Tests for edit_atom_builder.py — Stage 3 v4 atom construction."""
from skills.timeline_plan.models import CutPoint, EditAtom, AtomLine
from skills.timeline_plan.edit_atom_builder import build_edit_atoms


def _make_rl(line_id, original, rewritten, speaker="S",
             start_sec=0.0, end_sec=1.0, shot_number=1, shot_scene="kitchen"):
    return dict(line_id=line_id, original=original, rewritten=rewritten,
                speaker=speaker, start_seconds=start_sec, end_seconds=end_sec,
                shot_number=shot_number, shot_scene=shot_scene)


class FakeShot:
    def __init__(self, shot_number, start_sec, end_sec, scene_desc="", lines=None):
        self.shot_number = shot_number
        self.start_seconds = start_sec
        self.end_seconds = end_sec
        self.scene_description = scene_desc
        self.lines = lines or []


class TestBuildEditAtoms:
    def test_single_rewritten_line_creates_atom(self):
        rls = [_make_rl("L1", "hello", "hi", start_sec=1.0, end_sec=2.0)]
        shots = [FakeShot(1, 0.0, 5.0, "kitchen scene")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 1
        assert atoms[0].atom_id == "atom_001"
        assert atoms[0].primary_shot_number == 1
        assert atoms[0].start_sec == 1.0
        assert atoms[0].end_sec == 2.0
        assert atoms[0].has_rewritten_lines is True

    def test_unchanged_line_no_atom(self):
        rls = [_make_rl("L1", "hello", "hello", start_sec=1.0, end_sec=2.0)]
        shots = [FakeShot(1, 0.0, 5.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 0

    def test_contiguous_rewritten_lines_merge(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0),
            _make_rl("L2", "c", "d", start_sec=2.5, end_sec=3.5),
        ]
        shots = [FakeShot(1, 0.0, 5.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 1
        assert atoms[0].start_sec == 1.0
        assert atoms[0].end_sec == 3.5
        assert len(atoms[0].rewritten_lines) == 2

    def test_large_gap_splits_clusters(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0),
            _make_rl("L2", "c", "d", start_sec=5.0, end_sec=6.0),
        ]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 2

    def test_different_shots_split_atoms(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0, shot_number=1, speaker="Mia"),
            _make_rl("L2", "c", "d", start_sec=2.5, end_sec=3.5, shot_number=2, speaker="Ben"),
        ]
        shots = [
            FakeShot(1, 0.0, 3.0, "kitchen"),
            FakeShot(2, 3.0, 6.0, "living room"),
        ]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 2
        assert atoms[0].primary_shot_number == 1
        assert atoms[1].primary_shot_number == 2

    def test_boundary_snaps_to_scene_cut(self):
        rls = [_make_rl("L1", "a", "b", start_sec=5.0, end_sec=6.0)]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        cuts = [CutPoint(time_sec=4.7)]
        atoms = build_edit_atoms(shots, rls, cuts, video_duration=10.0)
        assert atoms[0].start_sec == 4.7

    def test_boundary_snap_does_not_cut_line(self):
        rls = [_make_rl("L1", "a", "b", start_sec=5.0, end_sec=6.0)]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        cuts = [CutPoint(time_sec=5.3)]
        atoms = build_edit_atoms(shots, rls, cuts, video_duration=10.0)
        assert atoms[0].start_sec == 5.0

    def test_no_duplicate_atom_ids(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0),
            _make_rl("L2", "c", "d", start_sec=5.0, end_sec=6.0),
        ]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        ids = [a.atom_id for a in atoms]
        assert len(ids) == len(set(ids))

    def test_rewritten_lines_covered(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0),
            _make_rl("L2", "c", "d", start_sec=5.0, end_sec=6.0),
        ]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        covered_ids = set()
        for a in atoms:
            for l in a.rewritten_lines:
                covered_ids.add(l.line_id)
        assert covered_ids == {"L1", "L2"}

    def test_empty_rewrite_lines(self):
        atoms = build_edit_atoms([], [], [], video_duration=10.0)
        assert atoms == []

    def test_scene_description_from_shot(self):
        rls = [_make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0)]
        shots = [FakeShot(1, 0.0, 5.0, "Mia in kitchen cooking")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert atoms[0].scene_description == "Mia in kitchen cooking"

    def test_shot_numbers_tracked(self):
        rls = [_make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0, shot_number=3)]
        shots = [FakeShot(3, 0.0, 5.0, "scene")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert atoms[0].shot_numbers == [3]
        assert atoms[0].primary_shot_number == 3

    def test_alternating_rewritten_unchanged_not_auto_merged(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=1.8),
            _make_rl("L2", "x", "x", start_sec=2.0, end_sec=2.5),
            _make_rl("L3", "c", "d", start_sec=2.8, end_sec=3.5),
        ]
        shots = [FakeShot(1, 0.0, 5.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) >= 2

    def test_cross_shot_merge_similar_scene(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0, shot_number=1, shot_scene="kitchen cooking"),
            _make_rl("L2", "c", "d", start_sec=2.5, end_sec=3.5, shot_number=2, shot_scene="kitchen"),
        ]
        shots = [
            FakeShot(1, 0.0, 2.5, "kitchen cooking scene"),
            FakeShot(2, 2.5, 5.0, "kitchen scene continues"),
        ]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 1
        assert atoms[0].shot_numbers == [1, 2]

    def test_cross_shot_no_merge_different_scene(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0, shot_number=1, shot_scene="kitchen", speaker="Mia"),
            _make_rl("L2", "c", "d", start_sec=2.5, end_sec=3.5, shot_number=2, shot_scene="classroom", speaker="Ben"),
        ]
        shots = [
            FakeShot(1, 0.0, 2.5, "kitchen scene"),
            FakeShot(2, 2.5, 5.0, "classroom lecture"),
        ]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 2


if __name__ == "__main__":
    import sys
    failed = 0
    tests = TestBuildEditAtoms()
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
