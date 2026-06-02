# Stage 3 v4: Segment-First Edit Atom Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Design spec:** `docs/superpowers/specs/2026-06-01-segment-first-edit-atom-design.md`
>
> **Execution order:** Tasks 1-6 are parallelizable. Tasks 7+ are sequential (each depends on prior).

**Goal:** Replace v3's "per-line matching → post-hoc grouping → normalizer semantic patching" with EditAtom-based segment matching + GenerationWindow execution resolution.

**Architecture:** Five new modules (atom builder, segment matcher, window resolver, prompt rewriter, plan finalizer) replace four v3 modules (evidence_builder, llm_planner, timeline_normalizer, cut_fusion). The pipeline order is: build atoms → match to canvas nodes → resolve generation windows → rewrite prompts per window → finalize timeline plan.

**Tech Stack:** Python 3.12+, dataclasses, openai (DeepSeek API), PySceneDetect, existing v3 validator.py and fetch_canvas.py are kept.

---

## Phase 1: Models + Foundation (parallelizable)

### Task 1: Strip `CutPoint.confidence`, add `AtomLine`, `EditAtom`, `GenerationWindow` to models.py

**Files:**
- Modify: `skills/timeline_plan/models.py`
- Modify: `skills/timeline_plan/tests/test_models.py`

- [ ] **Step 1: Remove `confidence` from CutPoint, add new models**

In `skills/timeline_plan/models.py`, after the existing imports and before `CutPoint`, add a `normalize_text` helper:

```python
import re

def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
```

Change `CutPoint` to remove `confidence`:

```python
@dataclass
class CutPoint:
    time_sec: float
```

After `CanvasNode` and before `TimelinePlanItem`, add three new dataclasses:

```python
@dataclass
class AtomLine:
    line_id: str
    speaker: str
    original: str
    rewritten: str
    start_sec: float
    end_sec: float
    shot_scene: str = ""

    @property
    def is_rewritten(self) -> bool:
        return _normalize_text(self.original) != _normalize_text(self.rewritten)


@dataclass
class EditAtom:
    atom_id: str
    shot_numbers: list[int] = field(default_factory=list)
    primary_shot_number: int = 0
    start_sec: float = 0.0
    end_sec: float = 0.0
    scene_description: str = ""
    lines: list[AtomLine] = field(default_factory=list)

    # Matching result — set by segment_matcher
    matched_node_id: str | None = None
    match_confidence: float | None = None
    match_reasoning: str = ""

    # Debug metadata
    boundary_reason: str = ""
    source_cut_times: list[float] = field(default_factory=list)

    @property
    def rewritten_lines(self) -> list[AtomLine]:
        return [line for line in self.lines if line.is_rewritten]

    @property
    def has_rewritten_lines(self) -> bool:
        return bool(self.rewritten_lines)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


@dataclass
class GenerationWindow:
    window_id: str
    start_sec: float
    end_sec: float
    atoms: list[EditAtom] = field(default_factory=list)

    matched_node_id: str | None = None
    match_confidence: float | None = None
    rewritten_prompt: str | None = None
    ref_images: list[str] = field(default_factory=list)

    degradation_level: int = 0
    degradation_reason: str = ""

    @property
    def covered_line_ids(self) -> list[str]:
        ids: list[str] = []
        for atom in self.atoms:
            for line in atom.rewritten_lines:
                ids.append(line.line_id)
        return sorted(set(ids))

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)
```

Keep everything else in models.py unchanged.

- [ ] **Step 2: Update test_models.py — add tests for new models**

In `skills/timeline_plan/tests/test_models.py`, add new test classes after the existing imports. Keep all existing tests.

Add import:
```python
from skills.timeline_plan.models import (
    CutPoint, CanvasNode, TimelinePlanItem, TimelinePlan,
    Stage3Input, MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
    AtomLine, EditAtom, GenerationWindow,
)
```

Add these test classes at the end of the file (before `if __name__ == "__main__":`):

```python
class TestCutPoint:
    def test_no_confidence_field(self):
        cp = CutPoint(time_sec=5.0)
        assert cp.time_sec == 5.0
        # confidence should no longer exist
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

    def _make_atom(self, atom_id="A1", lines=None, **kwargs):
        defaults = dict(atom_id=atom_id, primary_shot_number=1, start_sec=0.0, end_sec=5.0,
                        scene_description="test", shot_numbers=[1])
        defaults.update(kwargs)
        return EditAtom(lines=lines or [], **defaults)

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
        a1 = self._make_atom("A1", lines=[
            self._make_line("L1", "a", "b"),
        ])
        a2 = self._make_atom("A2", lines=[
            self._make_line("L2", "c", "d"),
        ])
        window = GenerationWindow(window_id="W1", start_sec=0.0, end_sec=10.0, atoms=[a1, a2])
        assert window.covered_line_ids == ["L1", "L2"]

    def test_duration_sec(self):
        window = GenerationWindow(window_id="W1", start_sec=3.0, end_sec=7.0, atoms=[])
        assert window.duration_sec == 4.0

    def test_defaults(self):
        window = GenerationWindow(window_id="W1", start_sec=0.0, end_sec=4.0, atoms=[])
        assert window.matched_node_id is None
        assert window.rewritten_prompt is None
        assert window.ref_images == []
        assert window.degradation_level == 0
        assert window.degradation_reason == ""


class TestConstants:
    def test_duration_constants(self):
        assert MIN_MODIFIED_DURATION == 4.0
        assert MAX_MODIFIED_DURATION == 30.0
```

Also update the `if __name__ == "__main__":` block to include the new test classes:

```python
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
```

- [ ] **Step 3: Run tests**

```bash
python skills/timeline_plan/tests/test_models.py
```

Expected: All tests PASS (including both old and new).

- [ ] **Step 4: Fix any downstream CutPoint usage that references confidence**

Search for `CutPoint(` constructor calls with confidence argument:

```bash
grep -rn "CutPoint(time_sec" skills/ --include="*.py"
```

In `skills/timeline_plan/generate_plan.py` line 181, change:
```python
cuts = [CutPoint(time_sec=c["time_sec"], confidence=c.get("confidence", 1.0)) for c in json.load(f)]
```
to:
```python
cuts = [CutPoint(time_sec=c["time_sec"]) for c in json.load(f)]
```

In `skills/timeline_plan/tests/test_cut_fusion.py`, existing tests that use `CutPoint(5.0)` (no confidence kwarg) are fine — they already work because confidence had a default. Nothing to change there.

In `skills/timeline_plan/tests/test_evidence_builder.py` line 43: `CutPoint(time_sec=1.0)` already uses keyword-only — same as above, fine.

- [ ] **Step 5: Run full test suite to confirm no regressions**

```bash
python skills/timeline_plan/tests/test_models.py && python skills/timeline_plan/tests/test_cut_fusion.py && python skills/timeline_plan/tests/test_evidence_builder.py && python skills/timeline_plan/tests/test_validator.py
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add skills/timeline_plan/models.py skills/timeline_plan/tests/test_models.py skills/timeline_plan/generate_plan.py
git commit -m "feat(models): add AtomLine, EditAtom, GenerationWindow; strip CutPoint.confidence"
```

---

### Task 2: Simplify scene detection — remove KeyFrame and extract_keyframes

**Files:**
- Modify: `skills/scene_detection/detect_scenes.py`
- Modify: `skills/scene_detection/tests/test_detect_scenes.py`

- [ ] **Step 1: Remove KeyFrame and extract_keyframes from detect_scenes.py**

In `skills/scene_detection/detect_scenes.py`:

1. Delete the `KeyFrame` dataclass (lines 19-23)
2. Delete the `extract_keyframes()` function (lines 78-119)
3. Delete the `_probe_duration()` function (lines 123-133) — only used by extract_keyframes
4. Remove unused `subprocess` import (only used by extract_keyframes)
5. Remove unused `os` import if only used by extract_keyframes (check: `os.path.exists` is used in detect_scene_boundaries line 41, so keep `os`)

The file should end up with only: `detect_scene_boundaries()` and `detect_node_internal_cuts()`.

Remove these lines from imports:
```python
import subprocess  # remove
```

- [ ] **Step 2: Update test_detect_scenes.py — remove KeyFrame tests**

In `skills/scene_detection/tests/test_detect_scenes.py`:

1. Remove `KeyFrame` from the import (line 8):
   Change: `from skills.scene_detection.detect_scenes import (detect_scene_boundaries, extract_keyframes, KeyFrame,)`
   To: `from skills.scene_detection.detect_scenes import detect_scene_boundaries`

2. Delete the entire `TestExtractKeyframes` class (lines 96-131).

- [ ] **Step 3: Run scene detection tests**

```bash
python -m pytest skills/scene_detection/tests/test_detect_scenes.py -v
```

Expected: All remaining tests PASS.

- [ ] **Step 4: Commit**

```bash
git add skills/scene_detection/detect_scenes.py skills/scene_detection/tests/test_detect_scenes.py
git commit -m "refactor(scene_detection): remove KeyFrame and extract_keyframes; CutPoint is time-only"
```

---

### Task 3: Update extract_script.py to run scene detection before lingolens extract

**Files:**
- Modify: `skills/script-extraction/extract_script.py`

- [ ] **Step 1: Add scene detection to the extract flow**

In `skills/script-extraction/extract_script.py`, after the `from agents.script_extraction import VideoScriptExtractor` import (around line 43), add:

```python
from skills.scene_detection.detect_scenes import detect_scene_boundaries
```

In `async def main()`, after getting `duration` (around line 113-114), add scene cut detection:

```python
    print("🔍 检测场景切点...")
    scene_cuts = detect_scene_boundaries(video_path)
    cut_times = [c.time_sec for c in scene_cuts]
    print(f"   {len(cut_times)} 个切点: {[f'{t:.1f}s' for t in cut_times[:10]]}{'...' if len(cut_times) > 10 else ''}")
```

Then in the `extractor.extract()` call (around line 125-130), add `scene_cut_times`:

```python
        result = await extractor.extract(
            video_path=video_path,
            utterances=utterances,
            duration_seconds=duration,
            temp_dir=temp_dir,
            scene_cut_times=cut_times,
        )
```

Note: `extractor.extract()` already accepts `**kwargs` in many implementations, or may need a signature update in lingolens. If the current lingolens `extract()` does NOT accept `scene_cut_times`, add a `try/except` fallback:

```python
        try:
            result = await extractor.extract(
                video_path=video_path,
                utterances=utterances,
                duration_seconds=duration,
                temp_dir=temp_dir,
                scene_cut_times=cut_times,
            )
        except TypeError:
            # lingolens extractor does not support scene_cut_times yet — fall back
            print("⚠️  lingolens 暂不支持 scene_cut_times，跳过切点注入")
            result = await extractor.extract(
                video_path=video_path,
                utterances=utterances,
                duration_seconds=duration,
                temp_dir=temp_dir,
            )
```

- [ ] **Step 2: Commit**

```bash
git add skills/script-extraction/extract_script.py
git commit -m "feat(extract_script): run scene detection before lingolens extract; pass cut times"
```

---

## Phase 2: Edit Atom Builder

### Task 4: Implement edit_atom_builder.py

**Files:**
- Create: `skills/timeline_plan/edit_atom_builder.py`
- Create: `skills/timeline_plan/tests/test_edit_atom_builder.py`

- [ ] **Step 1: Write the failing test**

Create `skills/timeline_plan/tests/test_edit_atom_builder.py`:

```python
"""Tests for edit_atom_builder.py — Stage 3 v4 atom construction."""
from skills.timeline_plan.models import CutPoint, EditAtom, AtomLine
from skills.timeline_plan.edit_atom_builder import build_edit_atoms


def _make_rl(line_id, original, rewritten, speaker="S",
             start_sec=0.0, end_sec=1.0, shot_number=1, shot_scene="kitchen"):
    return dict(line_id=line_id, original=original, rewritten=rewritten,
                speaker=speaker, start_seconds=start_sec, end_seconds=end_sec,
                shot_number=shot_number, shot_scene=shot_scene)


class FakeShotLine:
    def __init__(self, line_id, dialogue, speaker, start, end):
        self.line_id = line_id
        self.dialogue = dialogue
        self.speaker = speaker
        self.start_seconds = start
        self.end_seconds = end


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
            _make_rl("L2", "c", "d", start_sec=2.5, end_sec=3.5),  # gap = 0.5s < 1.5s
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
            _make_rl("L2", "c", "d", start_sec=5.0, end_sec=6.0),  # gap = 3.0s > 1.5s
        ]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        assert len(atoms) == 2

    def test_different_shots_split_atoms(self):
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=2.0, shot_number=1),
            _make_rl("L2", "c", "d", start_sec=2.5, end_sec=3.5, shot_number=2),
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
        cuts = [CutPoint(time_sec=4.7)]  # within 0.5s of raw_start=5.0
        atoms = build_edit_atoms(shots, rls, cuts, video_duration=10.0)
        assert atoms[0].start_sec == 4.7

    def test_boundary_snap_does_not_cut_line(self):
        rls = [_make_rl("L1", "a", "b", start_sec=5.0, end_sec=6.0)]
        shots = [FakeShot(1, 0.0, 10.0, "kitchen")]
        cuts = [CutPoint(time_sec=5.3)]  # inside the line — must NOT snap
        atoms = build_edit_atoms(shots, rls, cuts, video_duration=10.0)
        assert atoms[0].start_sec == 5.0  # unchanged

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
        """Line 1 rewritten, line 2 unchanged, line 3 rewritten.
        Should NOT auto-merge just because gap is small."""
        rls = [
            _make_rl("L1", "a", "b", start_sec=1.0, end_sec=1.8),
            _make_rl("L2", "x", "x", start_sec=2.0, end_sec=2.5),  # unchanged
            _make_rl("L3", "c", "d", start_sec=2.8, end_sec=3.5),
        ]
        shots = [FakeShot(1, 0.0, 5.0, "kitchen")]
        atoms = build_edit_atoms(shots, rls, [], video_duration=10.0)
        # With gap 2.8-1.8=1.0s, still within 1.5s but L2 unchanged breaks contiguity
        # The default conservative approach: don't merge across unchanged lines
        assert len(atoms) >= 2


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python skills/timeline_plan/tests/test_edit_atom_builder.py
```

Expected: FAIL because `build_edit_atoms` is not defined.

- [ ] **Step 3: Implement edit_atom_builder.py**

Create `skills/timeline_plan/edit_atom_builder.py`:

```python
"""Edit Atom Builder: Stage 1 shot + rewrite lines → EditAtom list.

Builds semantic edit units from script shots and rewritten dialogue lines.
Each EditAtom represents a coherent local scene fragment suitable for
canvas node prompt matching.
"""
from __future__ import annotations

import logging
from typing import Any

from skills.timeline_plan.models import CutPoint, EditAtom, AtomLine

logger = logging.getLogger(__name__)

CLUSTER_GAP_SEC = 1.5       # max gap between rewritten lines to stay in one cluster
SNAP_TOLERANCE_SEC = 0.5    # max distance for boundary snapping to scene cut


def build_edit_atoms(
    script_shots: list[Any],
    rewrite_lines: list[dict],
    scene_cuts: list[CutPoint],
    video_duration: float,
) -> list[EditAtom]:
    """Build EditAtoms from Stage 1 shots and rewrite lines.

    Algorithm:
      1. Group rewrite lines by Stage 1 shot.
      2. Within each shot, cluster contiguous rewritten lines.
      3. For each cluster, build an EditAtom with boundary snapping.
      4. Optionally merge adjacent-shot clusters (deferred to v4.1).

    Design rules:
      - ASR line boundaries are never cut.
      - Default: atoms don't cross shot boundaries.
      - Scene cuts only snap existing boundaries, never create new atoms.
    """
    if not rewrite_lines:
        return []

    # Index lines by shot_number
    lines_by_shot: dict[int, list[dict]] = {}
    for rl in rewrite_lines:
        sn = int(rl.get("shot_number", 0))
        lines_by_shot.setdefault(sn, []).append(rl)

    atoms: list[EditAtom] = []
    atom_counter = 0

    cut_times = sorted(c.time_sec for c in scene_cuts)

    for shot in script_shots:
        sn = getattr(shot, "shot_number", 0)
        shot_start = float(getattr(shot, "start_seconds", 0.0))
        shot_end = float(getattr(shot, "end_seconds", 0.0))
        scene_desc = getattr(shot, "scene_description", "") or ""

        shot_lines = sorted(
            lines_by_shot.get(sn, []),
            key=lambda rl: float(rl.get("start_seconds", 0.0)),
        )

        # Build clusters: contiguous rewritten lines
        clusters: list[list[dict]] = []
        current: list[dict] = []

        for rl in shot_lines:
            original = str(rl.get("original", ""))
            rewritten = str(rl.get("rewritten", ""))
            is_changed = _is_rewritten(original, rewritten)

            if is_changed:
                if current:
                    prev_end = float(current[-1].get("end_seconds", 0.0))
                    curr_start = float(rl.get("start_seconds", 0.0))
                    gap = curr_start - prev_end
                    if gap > CLUSTER_GAP_SEC:
                        clusters.append(current)
                        current = [rl]
                    else:
                        current.append(rl)
                else:
                    current.append(rl)

        if current:
            clusters.append(current)

        # Build EditAtom per cluster
        for cluster in clusters:
            atom_counter += 1
            start_sec = min(float(rl.get("start_seconds", 0.0)) for rl in cluster)
            end_sec = max(float(rl.get("end_seconds", 0.0)) for rl in cluster)

            # Boundary snapping
            snapped_start = _snap_boundary(start_sec, cut_times, -1, cluster_lines=cluster)
            snapped_end = _snap_boundary(end_sec, cut_times, 1, cluster_lines=cluster)

            # Don't snap into the middle of any line
            all_line_starts = {float(rl.get("start_seconds", 0.0)) for rl in cluster}
            all_line_ends = {float(rl.get("end_seconds", 0.0)) for rl in cluster}
            # Only snap if the new boundary doesn't cut a line
            if not _cuts_any_line(snapped_start, cluster, is_start=True):
                start_sec = snapped_start
            if not _cuts_any_line(snapped_end, cluster, is_start=False):
                end_sec = snapped_end

            atom_lines = [
                AtomLine(
                    line_id=str(rl.get("line_id", "")),
                    speaker=str(rl.get("speaker", "")),
                    original=str(rl.get("original", "")),
                    rewritten=str(rl.get("rewritten", "")),
                    start_sec=float(rl.get("start_seconds", 0.0)),
                    end_sec=float(rl.get("end_seconds", 0.0)),
                    shot_scene=str(rl.get("shot_scene", "")),
                )
                for rl in cluster
            ]

            atoms.append(EditAtom(
                atom_id=f"atom_{atom_counter:03d}",
                shot_numbers=[sn],
                primary_shot_number=sn,
                start_sec=start_sec,
                end_sec=end_sec,
                scene_description=scene_desc,
                lines=atom_lines,
                boundary_reason="asr_line_range",
            ))

    # Sort by start time
    atoms.sort(key=lambda a: a.start_sec)
    return atoms


def _is_rewritten(original: str, rewritten: str) -> bool:
    """Check if text was actually changed (normalized comparison)."""
    import re

    def norm(t: str) -> str:
        t = t.strip().lower()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"[^\w\s]", "", t)
        return t

    return norm(original) != norm(rewritten)


def _snap_boundary(
    target: float,
    cut_times: list[float],
    direction: int,
    cluster_lines: list[dict],
) -> float:
    """Snap target to nearest cut within tolerance.

    direction: -1 for start (prefer earlier snap), +1 for end (prefer later snap).
    Returns snapped value or original target.
    """
    best = target
    best_dist = float("inf")
    for ct in cut_times:
        dist = abs(ct - target)
        if dist <= SNAP_TOLERANCE_SEC and dist < best_dist:
            best = ct
            best_dist = dist
    return best


def _cuts_any_line(
    boundary: float,
    cluster_lines: list[dict],
    is_start: bool,
) -> bool:
    """Check if a boundary time falls inside any ASR line's [start, end]."""
    for rl in cluster_lines:
        ls = float(rl.get("start_seconds", 0.0))
        le = float(rl.get("end_seconds", 0.0))
        if ls < boundary < le:
            return True
        # For start boundary: if snap would be AFTER line starts, it cuts
        if is_start and boundary > ls:
            return True
        # For end boundary: if snap would be BEFORE line ends, it cuts
        if not is_start and boundary < le:
            return True
    return False
```

- [ ] **Step 4: Run tests**

```bash
python skills/timeline_plan/tests/test_edit_atom_builder.py
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/edit_atom_builder.py skills/timeline_plan/tests/test_edit_atom_builder.py
git commit -m "feat(edit_atom_builder): build EditAtoms from shots + rewrite lines with boundary snapping"
```

---

## Phase 3: Segment Matcher

### Task 5: Implement segment_matcher.py

**Files:**
- Create: `skills/timeline_plan/segment_matcher.py`
- Create: `skills/timeline_plan/tests/test_segment_matcher.py`

- [ ] **Step 1: Write the failing test**

Create `skills/timeline_plan/tests/test_segment_matcher.py`:

```python
"""Tests for segment_matcher.py — EditAtom → Canvas Node matching."""
import json
from unittest.mock import patch, MagicMock
from skills.timeline_plan.models import EditAtom, AtomLine, CanvasNode
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
        nodes = [
            _make_node("n1", "A video prompt about cats"),
            _make_node("n2", "Another prompt about dogs"),
        ]
        prompt = _build_matching_prompt([atom], nodes)
        assert "cats" in prompt
        assert "dogs" in prompt


class TestParseMatchResponse:
    def test_parses_valid_response(self):
        response = json.dumps({
            "matches": [
                {"atom_id": "A1", "node_id": "n1", "confidence": 0.9, "reasoning": "good match"},
            ],
            "unmatched": [],
        })
        matches, unmatched = _parse_match_response(response)
        assert len(matches) == 1
        assert matches[0]["atom_id"] == "A1"
        assert matches[0]["node_id"] == "n1"
        assert len(unmatched) == 0

    def test_parses_unmatched(self):
        response = json.dumps({
            "matches": [],
            "unmatched": [{"atom_id": "A2", "reason": "no match"}],
        })
        matches, unmatched = _parse_match_response(response)
        assert len(matches) == 0
        assert len(unmatched) == 1
        assert unmatched[0]["atom_id"] == "A2"

    def test_parses_markdown_fenced_json(self):
        response = '```json\n{"matches": [], "unmatched": []}\n```'
        matches, unmatched = _parse_match_response(response)
        assert matches == []
        assert unmatched == []

    def test_returns_empty_on_invalid(self):
        matches, unmatched = _parse_match_response("not json at all")
        assert matches == []
        assert unmatched == []


class TestMatchAtomsToNodes:
    @patch("skills.timeline_plan.segment_matcher._get_client")
    def test_populates_matched_node(self, mock_get_client):
        mock_client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = json.dumps({
            "matches": [{"atom_id": "A1", "node_id": "n1", "confidence": 0.85, "reasoning": "match"}],
            "unmatched": [],
        })
        mock_client.chat.completions.create.return_value = mock_resp
        mock_get_client.return_value = mock_client

        atom = _make_atom("A1", [_make_line("L1", "hello", "hi")])
        nodes = [_make_node("n1", "hello scene")]
        match_atoms_to_nodes([atom], nodes)

        assert atom.matched_node_id == "n1"
        assert atom.match_confidence == 0.85
        assert "match" in atom.match_reasoning

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
        match_atoms_to_nodes([atom], [])
        assert atom.matched_node_id is None

    def test_empty_atoms_noop(self):
        match_atoms_to_nodes([], [_make_node("n1", "test")])
        # should not raise


if __name__ == "__main__":
    import sys
    failed = 0
    for cls in [TestBuildMatchingPrompt, TestParseMatchResponse, TestMatchAtomsToNodes]:
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python skills/timeline_plan/tests/test_segment_matcher.py
```

Expected: FAIL because `match_atoms_to_nodes` is not defined.

- [ ] **Step 3: Implement segment_matcher.py**

Create `skills/timeline_plan/segment_matcher.py`:

```python
"""Segment Matcher: match EditAtoms to Canvas Node prompts via LLM.

Sends all target atoms and all canvas nodes in one LLM call for global
matching. Unmatched atoms keep matched_node_id=None.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from skills.timeline_plan.models import EditAtom, CanvasNode

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("LLM_PLANNER_MODEL", "deepseek-v4-pro")
_LOG_DIR = Path("runs/v4_plans/matcher_logs")
_log_counter = 0


def _get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def _log_llm(prompt: str, resp: str, dur: float):
    global _log_counter
    _log_counter += 1
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _LOG_DIR / f"{_log_counter:03d}_matcher_{time.strftime('%H%M%S')}.json"
    with open(fpath, "w") as f:
        json.dump({
            "prompt_chars": len(prompt), "response_chars": len(resp),
            "duration_sec": round(dur, 1),
            "prompt": prompt, "response": resp,
        }, f, ensure_ascii=False, indent=2)


def _build_matching_prompt(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
) -> str:
    """Build LLM prompt for atom-to-node matching."""
    atoms_json = json.dumps([
        {
            "atom_id": a.atom_id,
            "shot_numbers": a.shot_numbers,
            "primary_shot_number": a.primary_shot_number,
            "scene": a.scene_description[:400],
            "dialogue": [
                {"line_id": l.line_id, "speaker": l.speaker,
                 "original": l.original, "rewritten": l.rewritten}
                for l in a.rewritten_lines
            ],
        }
        for a in atoms
    ], ensure_ascii=False, indent=2)

    nodes_json = json.dumps([
        {"node_id": n.node_id, "prompt": n.prompt[:1200]}
        for n in canvas_nodes
    ], ensure_ascii=False, indent=2)

    return f"""## Role
Match each Edit Atom to the canvas node prompt that best fits its
dialogue + scene + character context.

Canvas node prompts describe original video generation intent.
Focus on matching spoken dialogue (exact or semantic) AND the
scene/environment/action described.

## Edit Atoms ({len(atoms)})
```json
{atoms_json}
```

## Canvas Nodes ({len(canvas_nodes)})
```json
{nodes_json}
```

## Rules
- Prefer dialogue match over scene keyword match.
- If no node reasonably matches an atom, put it in unmatched.
- Every atom must appear in either matches or unmatched.
- Semantic similarity is acceptable when exact text differs.

## Output
Return ONLY JSON:
```json
{{
  "matches": [
    {{"atom_id": "A1", "node_id": "n1", "confidence": 0.9, "reasoning": "..."}}
  ],
  "unmatched": [
    {{"atom_id": "A2", "reason": "no canvas prompt matches this scene"}}
  ]
}}
```"""


def _parse_match_response(text: str) -> tuple[list[dict], list[dict]]:
    """Parse LLM response into (matches, unmatched) lists."""
    if not text or not text.strip():
        return [], []
    t = text.strip()
    # Strip markdown fences
    if t.startswith("```"):
        ls = t.split("\n")
        if ls[0].startswith("```"):
            ls = ls[1:]
        if ls and ls[-1].strip() in ("```", "```json"):
            ls = ls[:-1]
        t = "\n".join(ls).strip()
    try:
        data = json.loads(t)
    except json.JSONDecodeError:
        # Try to extract JSON from garbled response
        b = t.find("{")
        if b < 0:
            return [], []
        d = 0
        for i in range(b, len(t)):
            if t[i] in "[{":
                d += 1
            elif t[i] in "]}":
                d -= 1
            if d == 0:
                try:
                    data = json.loads(t[b:i + 1])
                    break
                except json.JSONDecodeError:
                    pass
        else:
            return [], []

    return data.get("matches", []), data.get("unmatched", [])


def match_atoms_to_nodes(
    atoms: list[EditAtom],
    canvas_nodes: list[CanvasNode],
) -> None:
    """Match each EditAtom to a CanvasNode via LLM. Updates atoms in-place.

    Args:
        atoms: Target atoms to match (only those with has_rewritten_lines=True).
               matched_node_id, match_confidence, match_reasoning are set in-place.
        canvas_nodes: All available canvas nodes.
    """
    if not atoms:
        return

    client = _get_client()
    if not client:
        logger.warning("No LLM client available — skipping matching")
        return

    prompt = _build_matching_prompt(atoms, canvas_nodes)
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)

    t0 = time.time()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=32768,
            reasoning_effort="low",
            extra_body={"thinking": {"type": "enabled"}},
        )
        text = resp.choices[0].message.content or ""
        _log_llm(prompt, text, time.time() - t0)
    except Exception as e:
        logger.error("Segment matcher LLM call failed: %s", e)
        return

    matches, unmatched = _parse_match_response(text)

    match_map = {m["atom_id"]: m for m in matches}
    unmatched_ids = {u["atom_id"] for u in unmatched}

    for atom in atoms:
        if atom.atom_id in match_map:
            m = match_map[atom.atom_id]
            atom.matched_node_id = m.get("node_id")
            atom.match_confidence = float(m.get("confidence", 0.0))
            atom.match_reasoning = m.get("reasoning", "")
        elif atom.atom_id in unmatched_ids:
            # Already defaults to None; log it
            logger.warning("Atom %s unmatched: %s", atom.atom_id,
                           next((u.get("reason", "") for u in unmatched if u.get("atom_id") == atom.atom_id), "unknown"))
        else:
            logger.warning("Atom %s missing from LLM response — treated as unmatched", atom.atom_id)

    matched_count = sum(1 for a in atoms if a.matched_node_id)
    logger.info("Matcher: %d/%d atoms matched to nodes", matched_count, len(atoms))
```

- [ ] **Step 4: Run tests**

```bash
python skills/timeline_plan/tests/test_segment_matcher.py
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/segment_matcher.py skills/timeline_plan/tests/test_segment_matcher.py
git commit -m "feat(segment_matcher): LLM-based EditAtom → CanvasNode matching"
```

---

## Phase 4: Generation Window Resolver

### Task 6: Implement generation_window_resolver.py

**Files:**
- Create: `skills/timeline_plan/generation_window_resolver.py`
- Create: `skills/timeline_plan/tests/test_generation_window_resolver.py`

- [ ] **Step 1: Write the failing test**

Create `skills/timeline_plan/tests/test_generation_window_resolver.py`:

```python
"""Tests for generation_window_resolver.py — atom → >=4s executable windows."""
from skills.timeline_plan.models import EditAtom, AtomLine, GenerationWindow, CanvasNode
from skills.timeline_plan.generation_window_resolver import resolve_generation_windows


def _make_atom(aid, start, end, lines=None, matched_node="n1", shot=1, scene="test"):
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
        atom = _make_atom("A1", start=0.0, end=5.0, lines=[
            _make_line("L1", "a", "b"),
        ])
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
        # Two short atoms with same node should merge into one window
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
        # Different nodes should NOT merge
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
        a1 = _make_atom("A1", start=0.0, end=5.0, lines=[
            _make_line("L1", "a", "b"),
        ])
        a2 = _make_atom("A2", start=6.0, end=11.0, lines=[
            _make_line("L2", "c", "d"),
        ])
        nodes = [CanvasNode(node_id="n1", prompt="test", video_url="")]
        windows = resolve_generation_windows(
            atoms=[a1, a2], all_lines=[], canvas_nodes=nodes, video_duration=15.0,
        )
        ids = [w.window_id for w in windows]
        assert len(ids) == len(set(ids))

    def test_empty_atoms_returns_empty(self):
        windows = resolve_generation_windows(atoms=[], all_lines=[], canvas_nodes=[], video_duration=10.0)
        assert windows == []


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python skills/timeline_plan/tests/test_generation_window_resolver.py
```

Expected: FAIL because `resolve_generation_windows` is not defined.

- [ ] **Step 3: Implement generation_window_resolver.py**

Create `skills/timeline_plan/generation_window_resolver.py`:

```python
"""Generation Window Resolver: EditAtoms → executable >=4s GenerationWindows.

Converts semantic EditAtoms into execution-ready GenerationWindows that
satisfy Stage 4 constraints (min 4s duration, valid ref_images, etc.).
"""
from __future__ import annotations

import logging
from typing import Optional

from skills.timeline_plan.models import (
    EditAtom, AtomLine, GenerationWindow, CanvasNode,
    MIN_MODIFIED_DURATION, MAX_MODIFIED_DURATION,
)

logger = logging.getLogger(__name__)


def resolve_generation_windows(
    atoms: list[EditAtom],
    all_lines: list[AtomLine],
    canvas_nodes: list[CanvasNode],
    video_duration: float,
    min_duration_sec: float = MIN_MODIFIED_DURATION,
    max_duration_sec: float = MAX_MODIFIED_DURATION,
) -> list[GenerationWindow]:
    """Resolve EditAtoms into executable GenerationWindows.

    Strategy:
      1. Sort atoms by start_sec.
      2. Group atoms: atoms with same matched_node_id and small gap → one window.
      3. For atoms >= 4s: direct 1:1 atom → window.
      4. For short atoms: expand boundaries to at least 4s using adjacent
         original content. Never cut ASR lines.
      5. For unmatched atoms: emit degraded window with fallback reason.
      6. Populate ref_images from matched CanvasNode.

    Returns:
        List of GenerationWindows ready for prompt rewrite and finalization.
    """
    if not atoms:
        return []

    node_map: dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    windows: list[GenerationWindow] = []
    window_counter = 0

    sorted_atoms = sorted(atoms, key=lambda a: a.start_sec)

    # Step 1: Group atoms that should share a window
    groups: list[list[EditAtom]] = []
    current: list[EditAtom] = [sorted_atoms[0]]

    for atom in sorted_atoms[1:]:
        prev = current[-1]
        gap = atom.start_sec - prev.end_sec
        same_node = (
            prev.matched_node_id
            and atom.matched_node_id
            and prev.matched_node_id == atom.matched_node_id
        )

        if same_node and gap <= 2.0:
            current.append(atom)
        else:
            groups.append(current)
            current = [atom]

    groups.append(current)

    # Step 2: Build GenerationWindow per group
    for group in groups:
        window_counter += 1
        group_start = min(a.start_sec for a in group)
        group_end = max(a.end_sec for a in group)
        duration = group_end - group_start

        # Expand short windows
        if duration < min_duration_sec:
            deficit = min_duration_sec - duration
            # Expand right first (more natural for video), then left if needed
            right_room = video_duration - group_end
            right_expand = min(deficit, right_room)
            left_room = group_start
            left_expand = min(deficit - right_expand, left_room)
            group_end += right_expand
            group_start -= left_expand
            duration = group_end - group_start

        # Clamp to video bounds
        group_start = max(0.0, group_start)
        group_end = min(video_duration, group_end)

        # Determine match info from primary atom
        primary = group[0]
        matched_nid = primary.matched_node_id
        match_conf = primary.match_confidence

        # Get ref_images from canvas node
        ref_images: list[str] = []
        degradation_level = 0
        degradation_reason = ""

        if matched_nid and matched_nid in node_map:
            node = node_map[matched_nid]
            for ri in node.reference_images:
                url = ri if isinstance(ri, str) else ri.get("url", "") if isinstance(ri, dict) else ""
                if url:
                    ref_images.append(url)
            if not ref_images:
                degradation_level = max(degradation_level, 1)
                degradation_reason = "no_ref_images_in_node"
        else:
            degradation_level = 5
            degradation_reason = "unmatched_atom"
            matched_nid = None
            match_conf = None

        windows.append(GenerationWindow(
            window_id=f"window_{window_counter:03d}",
            start_sec=group_start,
            end_sec=group_end,
            atoms=group,
            matched_node_id=matched_nid,
            match_confidence=match_conf,
            ref_images=ref_images,
            degradation_level=degradation_level,
            degradation_reason=degradation_reason,
        ))

    logger.info("Window resolver: %d atoms → %d windows", len(atoms), len(windows))
    return windows
```

- [ ] **Step 4: Run tests**

```bash
python skills/timeline_plan/tests/test_generation_window_resolver.py
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/generation_window_resolver.py skills/timeline_plan/tests/test_generation_window_resolver.py
git commit -m "feat(window_resolver): resolve EditAtoms into >=4s GenerationWindows"
```

---

## Phase 5: Window Prompt Rewriter

### Task 7: Implement prompt_rewriter.py

**Files:**
- Create: `skills/timeline_plan/prompt_rewriter.py`
- Create: `skills/timeline_plan/tests/test_prompt_rewriter.py`

- [ ] **Step 1: Write the failing test**

Create `skills/timeline_plan/tests/test_prompt_rewriter.py`:

```python
"""Tests for prompt_rewriter.py — window-level prompt rewriting."""
import concurrent.futures
from unittest.mock import patch, MagicMock
from skills.timeline_plan.models import (
    EditAtom, AtomLine, GenerationWindow, CanvasNode,
)
from skills.timeline_plan.prompt_rewriter import (
    rewrite_prompts_for_windows,
    _make_rewrite_prompt,
    _check_rewritten_prompt,
)


def _make_atom(aid, lines, shot=1):
    return EditAtom(
        atom_id=aid, primary_shot_number=shot, start_sec=0.0, end_sec=3.0,
        scene_description="test", lines=lines, shot_numbers=[shot],
    )


def _make_line(lid, original, rewritten, speaker="Mia"):
    return AtomLine(line_id=lid, speaker=speaker, original=original,
                    rewritten=rewritten, start_sec=0.0, end_sec=1.0)


def _make_window(wid, atoms, node_id="n1"):
    return GenerationWindow(
        window_id=wid, start_sec=0.0, end_sec=5.0, atoms=atoms,
        matched_node_id=node_id,
    )


class TestMakeRewritePrompt:
    def test_includes_rewritten_lines(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "hello", "hi there"),
        ])])
        node = CanvasNode(node_id="n1", prompt="A scene: hello", video_url="")
        prompt = _make_rewrite_prompt(window, node, "B2")
        assert "hello" in prompt
        assert "hi there" in prompt
        assert "n1" not in prompt  # prompt is for generation, not matching

    def test_includes_level(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "a", "b"),
        ])])
        node = CanvasNode(node_id="n1", prompt="test", video_url="")
        prompt = _make_rewrite_prompt(window, node, "B2")
        assert "B2" in prompt

    def test_preserves_style_instruction(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "hello", "hi"),
        ])])
        node = CanvasNode(node_id="n1", prompt="cinematic 8k scene", video_url="")
        prompt = _make_rewrite_prompt(window, node, "B2")
        assert "preserve" in prompt.lower() or "keep" in prompt.lower()
        assert "environment" in prompt.lower() or "style" in prompt.lower()


class TestCheckRewrittenPrompt:
    def test_all_lines_present(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "hello", "hi"),
        ])])
        errors = _check_rewritten_prompt(window, "Someone says: hi")
        assert len(errors) == 0

    def test_missing_line_reported(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "hello", "uniquephrase123"),
        ])])
        errors = _check_rewritten_prompt(window, "Someone says: hi")
        assert len(errors) == 1
        assert "uniquephrase123" in errors[0]

    def test_original_line_not_required(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "hello", "hello"),  # unchanged
        ])])
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

        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "hello", "hi there"),
        ])])
        nodes = [CanvasNode(node_id="n1", prompt="A scene: hello", video_url="")]
        rewrite_prompts_for_windows([window], nodes, "B2")

        assert window.rewritten_prompt == "A rewritten scene: hi there"

    def test_no_client_skips(self):
        window = _make_window("W1", [_make_atom("A1", [
            _make_line("L1", "a", "b"),
        ])])
        # Without DEEPSEEK_API_KEY set, _get_client returns None
        rewrite_prompts_for_windows([window], [], "B2")
        assert window.rewritten_prompt is None  # not set

    def test_empty_windows_noop(self):
        rewrite_prompts_for_windows([], [], "B2")
        # should not raise


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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python skills/timeline_plan/tests/test_prompt_rewriter.py
```

Expected: FAIL because `rewrite_prompts_for_windows` is not defined.

- [ ] **Step 3: Implement prompt_rewriter.py**

Create `skills/timeline_plan/prompt_rewriter.py`:

```python
"""Window Prompt Rewriter: rewrite canvas node prompts per GenerationWindow.

Operates AFTER window resolution — each window gets one rewritten prompt
that replaces all rewritten dialogue lines while preserving visual style.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import time
from pathlib import Path

from skills.timeline_plan.models import GenerationWindow, CanvasNode

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = os.environ.get("LLM_PLANNER_MODEL", "deepseek-v4-pro")
_LOG_DIR = Path("runs/v4_plans/rewriter_logs")
_log_counter = 0


def _get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def _log_llm(window_id: str, prompt: str, resp: str, dur: float):
    global _log_counter
    _log_counter += 1
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _LOG_DIR / f"{_log_counter:03d}_rewrite_{window_id}_{time.strftime('%H%M%S')}.json"
    with open(fpath, "w") as f:
        json.dump({
            "window_id": window_id, "duration_sec": round(dur, 1),
            "prompt_chars": len(prompt), "response_chars": len(resp),
            "prompt": prompt, "response": resp,
        }, f, ensure_ascii=False, indent=2)


def _make_rewrite_prompt(
    window: GenerationWindow,
    node: CanvasNode,
    level: str,
) -> str:
    """Build the prompt rewrite instruction for one window."""
    changed_lines = []
    for atom in window.atoms:
        for line in atom.rewritten_lines:
            changed_lines.append({
                "line_id": line.line_id,
                "speaker": line.speaker,
                "original": line.original,
                "rewritten": line.rewritten,
            })

    changed_json = json.dumps(changed_lines, ensure_ascii=False, indent=2)

    return f"""## Role
Rewrite a video generation prompt for CEFR level {level}.
ONLY change the spoken dialogue listed under "Lines to Change".
All other text — environment, actions, camera, lighting, style keywords,
resolution — must be preserved verbatim.

## Original Prompt
```
{node.prompt}
```

## Lines to Change ({len(changed_lines)})
Replace each ORIGINAL dialogue with the REWRITTEN version in the prompt.
```json
{changed_json}
```

## Critical Rules
1. Do NOT change any visual description, camera direction, or style keywords.
2. The rewritten dialogue must appear verbatim in the output.
3. If original dialogue doesn't appear word-for-word in the prompt, insert
   the rewritten line in a semantically appropriate location.
4. Maintain the original prompt's structure, pacing, and tone.

Return ONLY the rewritten prompt text. No JSON, no markdown, no explanation."""


def _check_rewritten_prompt(
    window: GenerationWindow,
    rewritten_prompt: str,
) -> list[str]:
    """Check that all rewritten dialogue lines appear in the rewritten prompt."""
    errors = []
    for atom in window.atoms:
        for line in atom.rewritten_lines:
            rw = line.rewritten
            p_lower = rewritten_prompt.lower()
            rw_lower = rw.lower()
            rw_stripped = rw.rstrip(".!?,;:\"' ")
            if rw_lower in p_lower or rw_stripped.lower() in p_lower:
                continue
            errors.append(f"rewritten text '{rw[:60]}' not found in prompt")
    return errors


def rewrite_prompts_for_windows(
    windows: list[GenerationWindow],
    canvas_nodes: list[CanvasNode],
    level: str,
) -> None:
    """Rewrite prompts for all windows. Updates windows in-place.

    Each window gets its matched canvas node's prompt rewritten to
    use the rewritten dialogue lines.

    Args:
        windows: GenerationWindows to rewrite prompts for.
        canvas_nodes: All canvas nodes (indexed by node_id).
        level: CEFR level for prompt context.
    """
    if not windows:
        return

    client = _get_client()
    if not client:
        logger.warning("No LLM client available — skipping prompt rewrite")
        return

    node_map: dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)

    def rewrite_one(window: GenerationWindow) -> None:
        nid = window.matched_node_id
        if not nid or nid not in node_map:
            window.degradation_level = max(window.degradation_level, 3)
            window.degradation_reason = "no_canvas_node_for_rewrite"
            return

        node = node_map[nid]
        ri_prompt = _make_rewrite_prompt(window, node, level)

        # Retry up to 3 times, pick best
        best_text = ""
        best_errors = float("inf")

        for attempt in range(3):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": ri_prompt}],
                    temperature=0.3,
                    max_tokens=32768,
                    reasoning_effort="low",
                    extra_body={"thinking": {"type": "enabled"}},
                )
                text = (resp.choices[0].message.content or "").strip()
                # Strip markdown fences
                if text.startswith("```"):
                    ls = text.split("\n")
                    if ls[0].startswith("```"):
                        ls = ls[1:]
                    if ls and ls[-1].strip() in ("```", "```json"):
                        ls = ls[:-1]
                    text = "\n".join(ls).strip()
                _log_llm(window.window_id, ri_prompt, text, time.time() - t0)
            except Exception as e:
                logger.warning("Rewrite %s attempt %d failed: %s", window.window_id, attempt + 1, e)
                continue

            errors = _check_rewritten_prompt(window, text)
            if len(errors) < best_errors:
                best_text = text
                best_errors = len(errors)
            if best_errors == 0:
                break

        if best_text:
            window.rewritten_prompt = best_text
            if best_errors > 0:
                logger.warning("Rewrite %s: %d lines not verified in prompt", window.window_id, best_errors)
        else:
            window.degradation_level = max(window.degradation_level, 4)
            window.degradation_reason = "prompt_rewrite_failed"

    # Rewrite windows in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(windows), 5)) as pool:
        list(pool.map(rewrite_one, windows))

    ok = sum(1 for w in windows if w.rewritten_prompt)
    logger.info("Rewriter: %d/%d windows have rewritten prompts", ok, len(windows))
```

- [ ] **Step 4: Run tests**

```bash
python skills/timeline_plan/tests/test_prompt_rewriter.py
```

Expected: All PASS (note: `TestRewritePromptsForWindows.test_no_client_skips` will pass because no DEEPSEEK_API_KEY env var).

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/prompt_rewriter.py skills/timeline_plan/tests/test_prompt_rewriter.py
git commit -m "feat(prompt_rewriter): window-level prompt rewriting after window resolution"
```

---

## Phase 6: Plan Finalizer + Orchestration

### Task 8: Implement plan_finalizer.py

**Files:**
- Create: `skills/timeline_plan/plan_finalizer.py`
- Create: `skills/timeline_plan/tests/test_plan_finalizer.py`

- [ ] **Step 1: Write the failing test**

Create `skills/timeline_plan/tests/test_plan_finalizer.py`:

```python
"""Tests for plan_finalizer.py — GenerationWindow → TimelinePlan assembly."""
from skills.timeline_plan.models import (
    EditAtom, AtomLine, GenerationWindow, CanvasNode, TimelinePlan,
)
from skills.timeline_plan.plan_finalizer import finalize_timeline_plan


def _make_atom(aid, lines, shot=1, scene="test"):
    return EditAtom(
        atom_id=aid, primary_shot_number=shot, start_sec=0.0, end_sec=3.0,
        scene_description=scene, lines=lines, shot_numbers=[shot],
    )


def _make_line(lid, original, rewritten, speaker="Mia"):
    return AtomLine(line_id=lid, speaker=speaker,
                    original=original, rewritten=rewritten,
                    start_sec=0.0, end_sec=1.0)


def _make_window(wid, start, end, atoms, node_id="n1", prompt="test prompt",
                 ref_images=None, degradation=0):
    return GenerationWindow(
        window_id=wid, start_sec=start, end_sec=end, atoms=atoms,
        matched_node_id=node_id, match_confidence=0.9,
        rewritten_prompt=prompt, ref_images=ref_images or [],
        degradation_level=degradation, degradation_reason="",
    )


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
        plan = finalize_timeline_plan(
            windows=[window], shots=shots, video_duration=10.0,
            title="Test", level="B2",
        )
        # Should have: original [0-4], modified [4-8], original [8-10]
        assert plan.title == "Test"
        assert plan.level == "B2"
        modified = [i for i in plan.items if i.source == "modified"]
        original = [i for i in plan.items if i.source == "original"]
        assert len(modified) == 1
        assert modified[0].start_sec == 4.0
        assert modified[0].end_sec == 8.0
        assert modified[0].rewritten_prompt == "test prompt"
        assert modified[0].covered_line_ids == ["L1"]
        # Check full coverage
        assert plan.total_duration_sec == 10.0

    def test_modified_item_has_ref_images(self):
        window = _make_window("W1", start=2.0, end=6.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ], ref_images=["img1.jpg"])
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(
            windows=[window], shots=shots, video_duration=10.0,
            title="T", level="A2",
        )
        modified = [i for i in plan.items if i.source == "modified"]
        assert modified[0].ref_images == ["img1.jpg"]

    def test_modified_item_has_degradation(self):
        window = _make_window("W1", start=2.0, end=6.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ], degradation=1)
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(
            windows=[window], shots=shots, video_duration=10.0,
            title="T", level="B2",
        )
        modified = [i for i in plan.items if i.source == "modified"]
        assert modified[0].degradation_level == 1

    def test_no_overlap_in_output(self):
        window = _make_window("W1", start=3.0, end=7.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ])
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(
            windows=[window], shots=shots, video_duration=10.0,
            title="T", level="B2",
        )
        sorted_items = sorted(plan.items, key=lambda i: i.start_sec)
        for i in range(len(sorted_items) - 1):
            assert sorted_items[i].end_sec <= sorted_items[i + 1].start_sec + 0.1, \
                f"Overlap: {sorted_items[i].end_sec} > {sorted_items[i+1].start_sec}"

    def test_full_coverage_zero_to_duration(self):
        window = _make_window("W1", start=2.0, end=5.0, atoms=[
            _make_atom("A1", [_make_line("L1", "a", "b")]),
        ])
        shots = [FakeShot(1, 0.0, 10.0, "scene")]
        plan = finalize_timeline_plan(
            windows=[window], shots=shots, video_duration=10.0,
            title="T", level="B2",
        )
        sorted_items = sorted(plan.items, key=lambda i: i.start_sec)
        assert sorted_items[0].start_sec <= 0.1
        assert sorted_items[-1].end_sec >= 9.9

    def test_no_rewritten_lines_returns_all_original(self):
        shots = [FakeShot(1, 0.0, 5.0, "scene")]
        plan = finalize_timeline_plan(
            windows=[], shots=shots, video_duration=5.0,
            title="T", level="B2",
        )
        assert len(plan.items) == 1
        assert plan.items[0].source == "original"


if __name__ == "__main__":
    import sys
    failed = 0
    tests = TestFinalizeTimelinePlan()
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python skills/timeline_plan/tests/test_plan_finalizer.py
```

Expected: FAIL because `finalize_timeline_plan` is not defined.

- [ ] **Step 3: Implement plan_finalizer.py**

Create `skills/timeline_plan/plan_finalizer.py`:

```python
"""Plan Finalizer: GenerationWindows → executable TimelinePlan.

Performs pure geometry: carves modified windows out of the full timeline,
produces original segments for the remaining gaps, and validates the result.
Reuses v3's _carve_out() pattern — no semantic patching.
"""
from __future__ import annotations

import logging
from typing import Any

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, GenerationWindow, CutPoint,
    MIN_MODIFIED_DURATION,
)

logger = logging.getLogger(__name__)


def _carve_out(
    segments: list[tuple[float, float]],
    carve_start: float,
    carve_end: float,
) -> list[tuple[float, float]]:
    """Remove [carve_start, carve_end] from segment ranges. Pure geometry."""
    result: list[tuple[float, float]] = []
    for seg_start, seg_end in segments:
        if carve_start >= seg_end or carve_end <= seg_start:
            result.append((seg_start, seg_end))
        else:
            if seg_start < carve_start:
                result.append((seg_start, carve_start))
            if seg_end > carve_end:
                result.append((carve_end, seg_end))
    return result


def finalize_timeline_plan(
    windows: list[GenerationWindow],
    shots: list[Any],
    video_duration: float,
    title: str,
    level: str,
) -> TimelinePlan:
    """Convert GenerationWindows into a validated TimelinePlan.

    Steps:
      1. Convert each GenerationWindow to a modified TimelinePlanItem.
      2. Carve modified ranges out of the full timeline to produce original items.
      3. Fill gaps and sort chronologically.
      4. Validate coverage.

    Args:
        windows: Resolved GenerationWindows with rewritten prompts.
        shots: Stage 1 ScriptShots for scene descriptions.
        video_duration: Total video duration in seconds.
        title: Plan title.
        level: CEFR level.

    Returns:
        Validated TimelinePlan ready for Stage 4.
    """
    items: list[TimelinePlanItem] = []

    # Step 1: Modified items from windows
    for window in windows:
        primary_atom = window.atoms[0] if window.atoms else None
        items.append(TimelinePlanItem(
            shot_id=window.window_id,
            shot_number=primary_atom.primary_shot_number if primary_atom else 0,
            source="modified",
            start_sec=window.start_sec,
            end_sec=window.end_sec,
            scene_description=primary_atom.scene_description if primary_atom else "",
            ref_images=window.ref_images,
            rewritten_prompt=window.rewritten_prompt,
            matched_node_id=window.matched_node_id,
            match_confidence=window.match_confidence,
            original_duration=window.duration_sec,
            covered_line_ids=window.covered_line_ids,
            degradation_level=window.degradation_level,
            degradation_reason=window.degradation_reason,
        ))

    # Step 2: Carve modified ranges out of full [0, video_duration]
    original_segments = [(0.0, video_duration)]
    for window in windows:
        original_segments = _carve_out(
            original_segments, window.start_sec, window.end_sec,
        )

    for seg_start, seg_end in original_segments:
        if seg_end - seg_start > 0.1:
            items.append(TimelinePlanItem(
                shot_id=f"orig_{seg_start:.1f}",
                shot_number=0,
                source="original",
                start_sec=seg_start,
                end_sec=seg_end,
                scene_description="",
                original_duration=seg_end - seg_start,
            ))

    # Step 3: Sort chronologically
    items.sort(key=lambda i: i.start_sec)

    num_modified = sum(1 for i in items if i.source == "modified")
    num_original = sum(1 for i in items if i.source == "original")

    plan = TimelinePlan(
        title=title,
        level=level,
        total_duration_sec=video_duration,
        items=items,
        metadata={
            "num_items": len(items),
            "num_modified": num_modified,
            "num_original": num_original,
        },
    )

    # Step 4: Run validation
    from skills.timeline_plan.validator import validate_timeline_item, validate_timeline_items
    errors: list[str] = []
    for item in plan.items:
        errors.extend(validate_timeline_item(item))
    errors.extend(validate_timeline_items(plan.items, video_duration))

    blocking = [
        e for e in errors
        if any(kw in e.lower() for kw in (
            "overlap", "start_sec (", "missing start_sec", ">= end_sec",
            "empty rewritten_prompt", "covered by both",
            "gap at start", "gap at end", "empty shot_id", "invalid source",
        ))
    ]
    if blocking:
        raise ValueError(
            f"Validation FAILED with {len(blocking)} blocking errors:\n"
            + "\n".join(f"  - {e}" for e in blocking)
        )
    if errors:
        logger.warning("%d non-blocking validation warnings", len(errors))

    return plan
```

- [ ] **Step 4: Run tests**

```bash
python skills/timeline_plan/tests/test_plan_finalizer.py
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/plan_finalizer.py skills/timeline_plan/tests/test_plan_finalizer.py
git commit -m "feat(plan_finalizer): GenerationWindow → TimelinePlan assembly with pure geometry"
```

---

### Task 9: Update generate_plan.py for v4 orchestration

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py`

- [ ] **Step 1: Rewrite generate_plan.py with v4 pipeline**

Replace the entire body of `generate_timeline_plan()` and the imports. Keep `main()` and the `_SW`/`_S`/`_Sh`/`_L` helper classes unchanged.

```python
#!/usr/bin/env python3
"""Stage 3: Timeline plan generator — Segment-First v4 pipeline.

v4.0: EditAtom Builder → Segment Matcher → Window Resolver → Prompt Rewriter → Plan Finalizer.
Semantic matching at atom granularity. Execution windows resolved deterministically.
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, Stage3Input,
    AtomLine, EditAtom,
)
from skills.timeline_plan.edit_atom_builder import build_edit_atoms
from skills.timeline_plan.segment_matcher import match_atoms_to_nodes
from skills.timeline_plan.generation_window_resolver import resolve_generation_windows
from skills.timeline_plan.prompt_rewriter import rewrite_prompts_for_windows
from skills.timeline_plan.plan_finalizer import finalize_timeline_plan
from skills.timeline_plan.cut_fusion import determine_cut_points
from skills.timeline_plan.validator import validate_timeline_item, validate_timeline_items
import logging

logger = logging.getLogger(__name__)


def _collect_all_atom_lines(rewrite_lines_all: list[dict]) -> list[AtomLine]:
    """Collect all lines (changed + unchanged) as AtomLine objects for window resolution."""
    lines = []
    for rl in rewrite_lines_all:
        lines.append(AtomLine(
            line_id=str(rl.get("line_id", "")),
            speaker=str(rl.get("speaker", "")),
            original=str(rl.get("original", "")),
            rewritten=str(rl.get("rewritten", "")),
            start_sec=float(rl.get("start_seconds", 0.0)),
            end_sec=float(rl.get("end_seconds", 0.0)),
            shot_scene=str(rl.get("shot_scene", "")),
        ))
    return lines


def _resolve_video_duration(shots: list[Any], rewrite_lines: list[dict]) -> float:
    """Compute video duration from shot boundaries and ASR timing."""
    shot_end = max(
        (s.end_seconds for s in shots if hasattr(s, 'end_seconds')),
        default=60.0,
    )
    asr_end = max(
        (float(rl.get("end_seconds", 0.0)) for rl in rewrite_lines),
        default=0.0,
    )
    return max(shot_end, asr_end)


def _build_all_original_plan(
    shots: list[Any],
    scene_cuts: list[CutPoint],
    video_duration: float,
    title: str = "Untitled",
    level: str = "B2",
) -> TimelinePlan:
    """Fallback: build an all-original plan when no lines are rewritten."""
    cut_boundaries = determine_cut_points(shots, scene_cuts, video_duration)
    items = []
    for idx, shot in enumerate(shots):
        start_s, end_s = cut_boundaries[idx]
        items.append(TimelinePlanItem(
            shot_id=f"shot_{getattr(shot, 'shot_number', idx)}",
            shot_number=getattr(shot, "shot_number", idx),
            source="original",
            start_sec=start_s, end_sec=end_s,
            scene_description=getattr(shot, "scene_description", "") or "",
            original_duration=end_s - start_s,
        ))
    items.sort(key=lambda i: i.start_sec)
    return TimelinePlan(
        title=title, level=level,
        total_duration_sec=video_duration, items=items,
        metadata={"num_shots": len(shots), "num_items": len(items),
                   "num_modified": 0, "num_original": len(items)},
    )


def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    script_output = input_data.script_output
    shots = list(script_output.script.shots) if script_output else []
    rewrite_lines_all = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    scene_cuts = input_data.video_cut_points
    level = input_data.level

    video_duration = _resolve_video_duration(shots, rewrite_lines_all)
    title = getattr(script_output, "title", "Untitled") if script_output else "Untitled"

    # ── Stage 3A: Build EditAtoms ──
    logger.info("Building edit atoms...")
    atoms = build_edit_atoms(
        script_shots=shots,
        rewrite_lines=rewrite_lines_all,
        scene_cuts=scene_cuts,
        video_duration=video_duration,
    )
    logger.info("Built %d atoms from %d rewrite lines", len(atoms), len(rewrite_lines_all))

    target_atoms = [a for a in atoms if a.has_rewritten_lines]

    # Quick path: no rewritten lines → all original plan
    if not target_atoms:
        logger.info("No rewritten lines — all-original plan")
        return _build_all_original_plan(shots, scene_cuts, video_duration, title, level)

    logger.info("Target atoms: %d (with rewritten lines)", len(target_atoms))

    # ── Stage 3B: Segment Matcher ──
    logger.info("Matching atoms to canvas nodes...")
    match_atoms_to_nodes(target_atoms, canvas_nodes)
    matched = sum(1 for a in target_atoms if a.matched_node_id)
    logger.info("Matcher: %d/%d atoms matched", matched, len(target_atoms))

    # ── Stage 3C: Window Resolver ──
    logger.info("Resolving generation windows...")
    all_lines = _collect_all_atom_lines(rewrite_lines_all)
    windows = resolve_generation_windows(
        atoms=target_atoms,
        all_lines=all_lines,
        canvas_nodes=canvas_nodes,
        video_duration=video_duration,
    )
    logger.info("Windows: %d generation windows", len(windows))

    # ── Stage 3D: Prompt Rewriter ──
    logger.info("Rewriting prompts per window...")
    rewrite_prompts_for_windows(windows, canvas_nodes, level)
    ok = sum(1 for w in windows if w.rewritten_prompt)
    logger.info("Rewriter: %d/%d windows have prompts", ok, len(windows))

    # ── Stage 3E: Finalize ──
    logger.info("Finalizing timeline plan...")
    plan = finalize_timeline_plan(
        windows=windows,
        shots=shots,
        video_duration=video_duration,
        title=title,
        level=level,
    )

    logger.info("Timeline plan: %d items (%d modified, %d original)",
                len(plan.items),
                sum(1 for i in plan.items if i.source == "modified"),
                sum(1 for i in plan.items if i.source == "original"))

    return plan
```

The `main()` function stays unchanged from the existing file (lines 149-219). Only the function body of `generate_timeline_plan()` and the imports change.

- [ ] **Step 2: Verify the file is syntactically correct**

```bash
python -c "from skills.timeline_plan.generate_plan import generate_timeline_plan; print('OK')"
```

Expected: `OK` (import succeeds).

- [ ] **Step 3: Run existing validate/cut_fusion tests to check no regressions**

```bash
python skills/timeline_plan/tests/test_validator.py && python skills/timeline_plan/tests/test_cut_fusion.py
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add skills/timeline_plan/generate_plan.py
git commit -m "feat(generate_plan): v4 pipeline — EditAtom → SegmentMatcher → WindowResolver → PromptRewriter → Finalizer"
```

---

## Phase 7: Deprecation + Cleanup

### Task 10: Mark deprecated modules and update validator import

**Files:**
- Modify: `skills/timeline_plan/evidence_builder.py`
- Modify: `skills/timeline_plan/llm_planner.py`
- Modify: `skills/timeline_plan/timeline_normalizer.py`
- Modify: `skills/timeline_plan/cut_fusion.py`
- Modify: `skills/timeline_plan/planner_models.py`
- Modify: `skills/timeline_plan/planner_verifier.py`

- [ ] **Step 1: Add deprecation notices to deprecated modules**

Add this line to the top of each file (after the docstring):

**evidence_builder.py** (after line 10):
```python
# DEPRECATED in v4.0: replaced by edit_atom_builder.py + segment_matcher.py.
# Kept for reference; will be removed in a future version.
```

**llm_planner.py** (after line 7):
```python
# DEPRECATED in v4.0: replaced by segment_matcher.py + prompt_rewriter.py.
# Kept for reference; will be removed in a future version.
```

**timeline_normalizer.py** (after line 12):
```python
# DEPRECATED in v4.0: semantic patching replaced by window_resolver + prompt_rewriter;
# pure geometry (_carve_out, _finalize) retained in plan_finalizer.py.
# Kept for reference; will be removed in a future version.
```

**cut_fusion.py** (after line 8):
```python
# DEPRECATED in v4.0: boundary snapping logic migrated to edit_atom_builder.py.
# determine_cut_points() still used by generate_plan.py for all-original fallback.
# Will be fully replaced when the all-original path is refactored.
```

**planner_models.py** (after line 6):
```python
# DEPRECATED in v4.0: replaced by models.py EditAtom and GenerationWindow.
# Kept for reference; will be removed in a future version.
```

**planner_verifier.py** (after docstring, if any):
```python
# DEPRECATED in v4.0: validation covered by plan_finalizer.py + validator.py.
# Kept for reference; will be removed in a future version.
```

- [ ] **Step 2: Run full test suite to verify nothing is broken**

```bash
python skills/timeline_plan/tests/test_models.py && \
python skills/timeline_plan/tests/test_cut_fusion.py && \
python skills/timeline_plan/tests/test_evidence_builder.py && \
python skills/timeline_plan/tests/test_validator.py && \
python skills/timeline_plan/tests/test_edit_atom_builder.py && \
python skills/timeline_plan/tests/test_segment_matcher.py && \
python skills/timeline_plan/tests/test_generation_window_resolver.py && \
python skills/timeline_plan/tests/test_prompt_rewriter.py && \
python skills/timeline_plan/tests/test_plan_finalizer.py
```

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/evidence_builder.py skills/timeline_plan/llm_planner.py \
        skills/timeline_plan/timeline_normalizer.py skills/timeline_plan/cut_fusion.py \
        skills/timeline_plan/planner_models.py skills/timeline_plan/planner_verifier.py
git commit -m "chore: add deprecation notices to v3 modules replaced in v4"
```

---

## Integration Verification

### Task 11: End-to-end smoke test

**Files:**
- Create: `skills/timeline_plan/tests/test_v4_integration.py`

- [ ] **Step 1: Write integration test**

Create `skills/timeline_plan/tests/test_v4_integration.py`:

```python
"""Integration test: full v4 pipeline from input → TimelinePlan."""
import json
from dataclasses import asdict
from unittest.mock import patch, MagicMock
from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, Stage3Input,
)
from skills.timeline_plan.generate_plan import generate_timeline_plan


class FakeShot:
    def __init__(self, shot_number, start_sec, end_sec, scene_desc="test scene"):
        self.shot_number = shot_number
        self.start_seconds = start_sec
        self.end_seconds = end_sec
        self.scene_description = scene_desc


class FakeScript:
    def __init__(self, shots):
        self.shots = shots


class FakeScriptOutput:
    def __init__(self, shots, title="Test Episode"):
        self.script = FakeScript(shots)
        self.title = title


class TestV4Pipeline:
    @patch("skills.timeline_plan.segment_matcher._get_client")
    @patch("skills.timeline_plan.prompt_rewriter._get_client")
    def test_full_pipeline_single_rewritten_line(self, mock_rewrite_client, mock_match_client):
        """End-to-end: one line rewritten → matched → window → prompt → plan."""
        # Setup mocks
        mock_match = MagicMock()
        mock_match.chat.completions.create.return_value.choices = [MagicMock()]
        mock_match.chat.completions.create.return_value.choices[0].message.content = json.dumps({
            "matches": [{"atom_id": "atom_001", "node_id": "n1", "confidence": 0.9, "reasoning": "match"}],
            "unmatched": [],
        })
        mock_match_client.return_value = mock_match

        mock_rewrite = MagicMock()
        mock_rewrite.chat.completions.create.return_value.choices = [MagicMock()]
        mock_rewrite.chat.completions.create.return_value.choices[0].message.content = \
            "Rewritten prompt: hi there"
        mock_rewrite_client.return_value = mock_rewrite

        # Setup input
        script_output = FakeScriptOutput([
            FakeShot(1, 0.0, 10.0, "Mia in kitchen"),
        ])
        rewrite_json = {
            "lines": [
                {"line_id": "L1", "speaker": "Mia", "original": "hello", "rewritten": "hi",
                 "start_seconds": 2.0, "end_seconds": 4.0, "shot_number": 1,
                 "shot_scene": "Mia in kitchen"},
            ]
        }
        canvas_nodes = [
            CanvasNode(node_id="n1", prompt="Scene: Mia says hello in kitchen", video_url="",
                       reference_images=["img.jpg"]),
        ]

        inp = Stage3Input(
            script_output=script_output,
            rewrite_json=rewrite_json,
            canvas_nodes=canvas_nodes,
            level="B2",
        )

        plan = generate_timeline_plan(inp)

        # Assertions
        assert isinstance(plan, TimelinePlan)
        assert plan.title == "Test Episode"
        assert plan.level == "B2"
        assert len(plan.items) >= 2  # at least one modified + one original
        assert plan.total_duration_sec > 0

        modified = [i for i in plan.items if i.source == "modified"]
        assert len(modified) == 1
        assert modified[0].matched_node_id == "n1"
        assert modified[0].rewritten_prompt is not None
        assert "L1" in modified[0].covered_line_ids
        assert modified[0].ref_images == ["img.jpg"]

        # Check full coverage
        original = [i for i in plan.items if i.source == "original"]
        assert len(original) >= 1

        # No gaps
        sorted_items = sorted(plan.items, key=lambda i: i.start_sec)
        for i in range(len(sorted_items) - 1):
            assert sorted_items[i].end_sec <= sorted_items[i + 1].start_sec + 0.1

    def test_no_rewritten_lines_returns_all_original(self):
        """When no lines are rewritten, return all-original plan."""
        script_output = FakeScriptOutput([FakeShot(1, 0.0, 10.0, "scene")])
        rewrite_json = {
            "lines": [
                {"line_id": "L1", "speaker": "M", "original": "hello", "rewritten": "hello",
                 "start_seconds": 2.0, "end_seconds": 4.0, "shot_number": 1, "shot_scene": "s"},
            ]
        }
        inp = Stage3Input(script_output=script_output, rewrite_json=rewrite_json, level="B2")
        plan = generate_timeline_plan(inp)
        assert all(i.source == "original" for i in plan.items)


if __name__ == "__main__":
    import sys
    failed = 0
    tests = TestV4Pipeline()
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
```

- [ ] **Step 2: Run integration test**

```bash
python skills/timeline_plan/tests/test_v4_integration.py
```

Expected: All PASS.

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/tests/test_v4_integration.py
git commit -m "test(integration): end-to-end v4 pipeline smoke test"
```

---

## Summary

| Task | Component | Files Created | Files Modified |
|------|-----------|---------------|----------------|
| 1 | Models + CutPoint | — | `models.py`, `generate_plan.py`, `tests/test_models.py` |
| 2 | Scene Detection cleanup | — | `detect_scenes.py`, `tests/test_detect_scenes.py` |
| 3 | extract_script scene cuts | — | `extract_script.py` |
| 4 | Edit Atom Builder | `edit_atom_builder.py`, `tests/test_edit_atom_builder.py` | — |
| 5 | Segment Matcher | `segment_matcher.py`, `tests/test_segment_matcher.py` | — |
| 6 | Window Resolver | `generation_window_resolver.py`, `tests/test_generation_window_resolver.py` | — |
| 7 | Prompt Rewriter | `prompt_rewriter.py`, `tests/test_prompt_rewriter.py` | — |
| 8 | Plan Finalizer | `plan_finalizer.py`, `tests/test_plan_finalizer.py` | — |
| 9 | Orchestration | — | `generate_plan.py` |
| 10 | Deprecation | — | 6 deprecated modules |
| 11 | Integration test | `tests/test_v4_integration.py` | — |

**Execution order:** Tasks 1-3 are parallelizable (no dependencies between them). Tasks 4-9 are sequential. Task 10 can run anytime after Task 9. Task 11 runs last.

**Key interfaces between modules:**

```
build_edit_atoms(shots, rewrite_lines, cuts, duration) → list[EditAtom]
match_atoms_to_nodes(atoms, canvas_nodes) → None (in-place update)
resolve_generation_windows(atoms, all_lines, canvas_nodes, duration) → list[GenerationWindow]
rewrite_prompts_for_windows(windows, canvas_nodes, level) → None (in-place update)
finalize_timeline_plan(windows, shots, duration, title, level) → TimelinePlan
```
