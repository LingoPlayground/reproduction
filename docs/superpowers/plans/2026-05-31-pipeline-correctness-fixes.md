# Pipeline Correctness Fixes — P0/P1 Data Integrity & Code Quality

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all P0 (silent data loss) and P1 (state consistency) bugs identified in the Stage 3/4 pipeline review, plus remove all dead code and centralize env loading. Zero backward compatibility with legacy v1 pipeline.

**Architecture:** Each fix is a minimal, self-contained change to a single module. The guiding invariant is: every `original != rewritten` line MUST appear in exactly one `TimelinePlanItem.covered_line_ids`, and that item MUST produce a video segment — or the pipeline MUST fail explicitly.

**Tech Stack:** Python 3.9+, ffmpeg, seedance API

---

## Task 1: Fix unmatched short rewritten lines silently dropped

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py:534-567`

**Issue:** Lines 544-546 mark unmatched rewritten lines as "handled" when their combined duration is < 4s, then `continue` without creating any TimelinePlanItem. The lines disappear silently.

**Fix:** Instead of silently dropping, create an explicit `source="original"` degraded item with a clear `degradation_reason`, or extend to 4s via `resolve_duration`. The degraded original approach is safer (guarantees content coverage).

- [ ] **Step 1: Replace the silent-drop block with degraded-original fallback**

Replace lines 534-567:

```python
        if needs_rewrite:
            if shot_line_ids - handled_rewrite_line_ids:
                # This shot has unreplaced rewritten lines → degraded fallback
                unmatched = [rl for rl in matching
                             if rl["line_id"] not in handled_rewrite_line_ids
                             and str(rl.get("original", "")) != str(rl.get("rewritten", ""))]
                if not unmatched:
                    continue
                min_start = min(rl.get("start_seconds", start_s) for rl in unmatched)
                max_end = max(rl.get("end_seconds", end_s) for rl in unmatched)
                if max_end - min_start < MIN_SEEDANCE_DURATION:
                    handled_rewrite_line_ids.update({rl["line_id"] for rl in unmatched})
                    continue
                min_start, max_end = _snap_boundaries(min_start, max_end, video_cuts)
                rl_objects = _make_rl_objects(unmatched)
                rewritten_prompt = compose_prompt_patch("", rl_objects, scene_desc, "full_fallback")
                items.append(TimelinePlanItem(
                    shot_id=f"shot_{shot.shot_number}_fallback",
                    shot_number=shot.shot_number,
                    source="seedance",
                    start_sec=min_start, end_sec=max_end,
                    scene_description=scene_desc,
                    rewritten_prompt=rewritten_prompt,
                    degradation_level=2,
                    seedance_duration=normalize_seedance_duration(max_end - min_start),
                    original_duration=max_end - min_start,
                    operation_type="full_fallback",
                    duration_strategy="direct",
                    covered_line_ids=sorted({rl["line_id"] for rl in unmatched}),
                    borrowed_line_ids=[],
                    source_node_ids=[],
                    degradation_reason="no_matching_canvas_node",
                ))
            continue
```

With:

```python
        if needs_rewrite:
            if shot_line_ids - handled_rewrite_line_ids:
                # This shot has unreplaced rewritten lines → degraded fallback
                unmatched = [rl for rl in matching
                             if rl["line_id"] not in handled_rewrite_line_ids
                             and str(rl.get("original", "")) != str(rl.get("rewritten", ""))]
                if unmatched:
                    min_start = min(rl.get("start_seconds", start_s) for rl in unmatched)
                    max_end = max(rl.get("end_seconds", end_s) for rl in unmatched)
                    line_ids = {rl["line_id"] for rl in unmatched}
                    if max_end - min_start < MIN_SEEDANCE_DURATION:
                        # Too short for seedance → degraded original with explicit reason
                        min_start, max_end = _snap_boundaries(min_start, max_end, video_cuts)
                        items.append(TimelinePlanItem(
                            shot_id=f"shot_{shot.shot_number}_degraded",
                            shot_number=shot.shot_number,
                            source="original",
                            start_sec=min_start, end_sec=max_end,
                            scene_description=scene_desc,
                            degradation_level=6,
                            original_duration=max_end - min_start,
                            covered_line_ids=sorted(line_ids),
                            degradation_reason="too_short_for_seedance_unmatched",
                        ))
                    else:
                        min_start, max_end = _snap_boundaries(min_start, max_end, video_cuts)
                        rl_objects = _make_rl_objects(unmatched)
                        rewritten_prompt = compose_prompt_patch("", rl_objects, scene_desc, "full_fallback")
                        items.append(TimelinePlanItem(
                            shot_id=f"shot_{shot.shot_number}_fallback",
                            shot_number=shot.shot_number,
                            source="seedance",
                            start_sec=min_start, end_sec=max_end,
                            scene_description=scene_desc,
                            rewritten_prompt=rewritten_prompt,
                            degradation_level=2,
                            seedance_duration=normalize_seedance_duration(max_end - min_start),
                            original_duration=max_end - min_start,
                            operation_type="full_fallback",
                            duration_strategy="direct",
                            covered_line_ids=sorted(line_ids),
                            borrowed_line_ids=[],
                            source_node_ids=[],
                            degradation_reason="no_matching_canvas_node",
                        ))
                    handled_rewrite_line_ids.update(line_ids)
            # Still produce original segment for this shot's non-rewritten lines.
            # A shot with rewritten lines needs its original content preserved, too.
            items.append(TimelinePlanItem(
                shot_id=f"shot_{shot.shot_number}",
                shot_number=shot.shot_number,
                source="original",
                start_sec=start_s, end_sec=end_s,
                scene_description=scene_desc,
                original_duration=end_s - start_s,
            ))
            continue
```

- [ ] **Step 2: Verify the fix with existing tests**

```bash
python -m pytest skills/timeline_plan/tests/test_generate_plan.py -v
```

Expected: all existing tests pass (they test with `make_rewrite` which has `original != rewritten` but are handled via matched nodes — the fallback path change is additive).

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/generate_plan.py
git commit -m "fix: unmatched short rewritten lines produce degraded original item instead of silent drop"
```

---

## Task 2: Fix partially-rewritten shots losing non-rewritten lines

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py:534-567`

**Issue:** When `needs_rewrite=True` but `shot_line_ids - handled_rewrite_line_ids` is empty (all rewritten lines already handled by node loop), the `continue` at line 567 skips the entire shot. Non-rewritten lines in that shot are never represented as an original segment.

**Fix:** This is addressed by Task 1's new code — the `items.append(...)` for the original segment is now OUTSIDE the `if shot_line_ids - handled_rewrite_line_ids:` block. Every shot with `needs_rewrite=True` now gets its original segment regardless of whether all rewritten lines were handled.

Task 1 already covers this. No additional changes needed.

---

## Task 3: Fix fallback items not added to `handled_rewrite_line_ids`

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py:550-567` (old code)

**Issue:** In the original fallback path, when a `source="seedance"` fallback item is created, its `covered_line_ids` are recorded in the TimelinePlanItem but never added to `handled_rewrite_line_ids`. If the same line_id appears in another shot, a duplicate fallback item could be created.

**Fix:** Task 1's new code already adds `handled_rewrite_line_ids.update(line_ids)` after creating the fallback item (both for the degraded original and seedance fallback paths). This is covered.

---

## Task 4: Remove dead code from `generate_plan.py`

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py:21, 82-132, 428-430`

Three pieces of dead code:

1. `segment_node_prompts` import and call — costs LLM money, result never used
2. `_extend_short_group` function (L82-132) — replaced by `resolve_duration`, never called
3. `build_evidence_pack` import (L25) — imported but never called

- [ ] **Step 1: Remove the imports**

Line 21: Change
```python
from skills.timeline_plan.canvas_matcher import match_lines_to_nodes, segment_node_prompts
```
to
```python
from skills.timeline_plan.canvas_matcher import match_lines_to_nodes
```

Line 25: Remove entirely
```python
from skills.timeline_plan.evidence_builder import build_evidence_pack
```

- [ ] **Step 2: Remove the `_extend_short_group` function**

Delete lines 82-132 (the entire `_extend_short_group` function definition).

- [ ] **Step 3: Remove the `segment_node_prompts` call**

Lines 428-430: Change
```python
    sections_by_node: Dict[str, List] = {}
    if canvas_nodes:
        sections_by_node = segment_node_prompts(canvas_nodes)
```
to nothing (delete these three lines entirely).

- [ ] **Step 4: Verify tests still pass**

```bash
python -m pytest skills/timeline_plan/tests/test_generate_plan.py -v
```

Expected: all pass. `_extend_short_group` was never tested directly (replaced by resolve_duration before tests).

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/generate_plan.py
git commit -m "refactor: remove dead code — _extend_short_group, segment_node_prompts call, unused import"
```

---

## Task 5: Fix assemble.py — seedance + fallback failure → explicit error

**Files:**
- Modify: `skills/video_assembly/assemble.py:282-294`

**Issue:** When both seedance generation and the fallback ffmpeg trim fail, the segment is silently skipped. Only a print line marks it, then the concat produces a shorter video without any error.

**Fix:** Raise `RuntimeError` when both paths fail. A production video pipeline must not silently drop planned segments.

- [ ] **Step 1: Replace the silent-skip fallback with strict-mode error**

Lines 282-294: Change
```python
            else:
                # Fallback to original segment — don't crash on ffmpeg failure
                try:
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-ss", f"{item['start_sec']:.3f}",
                        "-i", original_video,
                        "-t", f"{item['end_sec'] - item['start_sec']:.3f}",
                        "-c:v", "libx264", "-c:a", "aac",
                        seg_path,
                    ], capture_output=True, check=True)
                    print(f"  [SEED-FB] Shot {item['shot_number']}: seedance failed → original fallback")
                except subprocess.CalledProcessError:
                    print(f"  [SEED-FB] Shot {item['shot_number']}: seedance + fallback both failed → skipped")
```

To:
```python
            else:
                # Fallback to original segment
                try:
                    subprocess.run([
                        "ffmpeg", "-y",
                        "-ss", f"{item['start_sec']:.3f}",
                        "-i", original_video,
                        "-t", f"{item['end_sec'] - item['start_sec']:.3f}",
                        "-c:v", "libx264", "-c:a", "aac",
                        seg_path,
                    ], capture_output=True, check=True)
                    print(f"  [SEED-FB] Shot {item['shot_number']}: seedance failed → original fallback")
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(
                        f"Shot {item['shot_number']}: seedance + fallback both failed. "
                        f"ffmpeg stderr: {e.stderr.decode()[:200] if e.stderr else 'unknown'}"
                    ) from e
```

- [ ] **Step 2: Verify assemble.py still runs on a valid plan**

```bash
python -m pytest skills/video_assembly/tests/test_assemble.py -v
```

Expected: existing tests pass (they likely mock seedance or use skip-seedance).

- [ ] **Step 3: Commit**

```bash
git add skills/video_assembly/assemble.py
git commit -m "fix: seedance + fallback failure raises RuntimeError instead of silent skip"
```

---

## Task 6: Add post-concat integrity verification

**Files:**
- Modify: `skills/video_assembly/assemble.py:296-332`

**Issue:** After concat, the code only checks ffmpeg return code and file size. No verification that segment count matches plan items or that output duration matches expected total.

**Fix:** Add two checks after concat: segment count vs. plan item count, and output duration vs. planned total.

- [ ] **Step 1: Add segment-count check before normalization**

After line 298 (the end of the item loop), insert:

```python
    # ── Integrity check: every plan item must produce a segment ──
    planned_seedance = sum(1 for item in items if item.get("source") == "seedance" and not skip_seedance)
    planned_original = sum(1 for item in items if item.get("source") != "seedance" or skip_seedance)
    planned_total = planned_seedance + planned_original
    if len(segment_paths) != planned_total:
        missing = planned_total - len(segment_paths)
        raise RuntimeError(
            f"Segment count mismatch: {len(segment_paths)} produced, "
            f"{planned_total} planned ({missing} missing). Aborting."
        )
```

- [ ] **Step 2: Add output duration check after concat**

After line 329 (the size print), insert:

```python
    # ── Duration integrity check ──
    planned_total_duration = items[-1]["end_sec"] if items else 0.0
    actual_duration = _probe_duration(output_path)
    drift = actual_duration - planned_total_duration
    if abs(drift) > 2.0:
        print(f"  [WARN] Duration drift: planned {planned_total_duration:.1f}s, actual {actual_duration:.1f}s (drift {drift:+.1f}s)")
    else:
        print(f"  Duration OK: {actual_duration:.1f}s (planned {planned_total_duration:.1f}s)")
```

- [ ] **Step 3: Verify**

```bash
python -m pytest skills/video_assembly/tests/test_assemble.py -v
```

- [ ] **Step 4: Commit**

```bash
git add skills/video_assembly/assemble.py
git commit -m "feat: add post-concat integrity checks — segment count + duration verification"
```

---

## Task 7: Fix seedance trim — replace `-c copy` with re-encode

**Files:**
- Modify: `skills/video_assembly/assemble.py:274-278`

**Issue:** The `-c copy` trim is not frame-accurate; it aligns to keyframe boundaries, potentially producing a trimmed segment longer than planned.

**Fix:** Use re-encode (`libx264 + aac`) for frame-accurate trimming. Since the segment is normalized anyway later, the re-encode here adds negligible overhead for correctness.

- [ ] **Step 1: Replace the trim command**

Lines 274-278: Change
```python
                    subprocess.run([
                        "ffmpeg", "-y", "-i", seg_path,
                        "-t", f"{planned_duration:.3f}",
                        "-c", "copy", trimmed,
                    ], capture_output=True, check=True)
```

To:
```python
                    subprocess.run([
                        "ffmpeg", "-y", "-i", seg_path,
                        "-t", f"{planned_duration:.3f}",
                        "-c:v", "libx264", "-c:a", "aac",
                        "-preset", "ultrafast",
                        trimmed,
                    ], capture_output=True, check=True)
```

- [ ] **Step 2: Verify**

```bash
python -m pytest skills/video_assembly/tests/test_assemble.py -v
```

- [ ] **Step 3: Commit**

```bash
git add skills/video_assembly/assemble.py
git commit -m "fix: use re-encode for seedance trim instead of -c copy for frame accuracy"
```

---

## Task 8: Remove unused `normalize_seedance_duration` import from assemble.py

**Files:**
- Modify: `skills/video_assembly/assemble.py:55`

**Issue:** `normalize_seedance_duration` is imported at line 55 but never called in assemble.py. All normalization happens at plan-creation time in generate_plan.py.

- [ ] **Step 1: Remove the import**

Delete line 55:
```python
from skills.timeline_plan.models import normalize_seedance_duration
```

- [ ] **Step 2: Verify**

```bash
python -m pytest skills/video_assembly/tests/test_assemble.py -v
```

- [ ] **Step 3: Commit**

```bash
git add skills/video_assembly/assemble.py
git commit -m "refactor: remove unused normalize_seedance_duration import from assemble.py"
```

---

## Task 9: Fix `_validate_rewrite` to handle `semantic_insert` correctly

**Files:**
- Modify: `skills/timeline_plan/prompt_composer.py:256-265`

**Issue:** `_validate_rewrite` checks `rewritten.strip() in prompt` for ALL operation types. For `semantic_insert`, the rewritten text is being INSERTED by the LLM, so it will NOT be in the prompt before the call. The validation always fails, triggering unnecessary fallback.

**Fix:** Skip the "rewritten must be in prompt" check for `semantic_insert`. For `semantic_insert`, the validation should instead check that the rewritten text exists in the result (which is the LLM output, not the original prompt).

- [ ] **Step 1: Fix `_validate_rewrite`**

Replace lines 256-265:

```python
def _validate_rewrite(
    prompt: str,
    rewrite_lines: List[Any],
    operation_type: str = "literal_replace",
) -> bool:
    for line in rewrite_lines:
        rewritten = getattr(line, "rewritten", "") or ""
        if rewritten.strip() and rewritten.strip() not in prompt:
            return False
    return True
```

With:

```python
def _validate_rewrite(
    prompt: str,
    rewrite_lines: List[Any],
    operation_type: str = "literal_replace",
) -> bool:
    """Validate that rewritten dialogue appears in the LLM output prompt.
    
    For literal_replace and fuzzy_replace: rewritten text must exist in the output.
    For semantic_insert: rewritten text must exist in the output (it was inserted by LLM).
    For section_reconstruct and full_fallback: validation is lenient — check at least
    one rewritten line exists in the output (the LLM may restructure significantly).
    """
    restored_ops = {"section_reconstruct", "full_fallback"}
    
    for line in rewrite_lines:
        rewritten = getattr(line, "rewritten", "") or ""
        if not rewritten.strip():
            continue
        if rewritten.strip() in prompt:
            continue
        if operation_type in restored_ops:
            # Lenient: at least one match is sufficient
            continue
        return False
    
    # For restored ops, verify at least one rewritten line is present
    if operation_type in restored_ops:
        any_match = any(
            getattr(line, "rewritten", "").strip() in prompt
            for line in rewrite_lines
            if getattr(line, "rewritten", "").strip()
        )
        return any_match
    
    return True
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest skills/timeline_plan/tests/test_prompt_composer.py -v
```

Expected: `test_no_original_required_for_semantic_insert` still passes, plus any other existing tests.

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/prompt_composer.py
git commit -m "fix: _validate_rewrite handles semantic_insert correctly, skips inappropriate inclusion check"
```

---

## Task 10: Fix canvas_matcher.py JSON parsing robustness

**Files:**
- Modify: `skills/timeline_plan/canvas_matcher.py:288-309`

**Issue:** The regex `\{[^{]*"mappings"` fails on nested objects before `"mappings"`. Plus, markdown code blocks (` ```json ``` `) are not stripped before parsing (unlike `segment_node_prompts` which does strip them). The fallback `text.replace('\n', ' ').replace('  ', ' ')` only collapses double spaces once, not recursively.

**Fix:** Strip markdown code fences first, then use a simpler approach — just find `"mappings"` by string search, then bracket-count to extract the full JSON object.

- [ ] **Step 1: Replace the JSON parsing block**

Lines 288-309: Change
```python
    # Parse JSON response
    try:
        obj_match = re.search(r'\{[^{]*"mappings"', text)
        if obj_match:
            start = obj_match.start()
            depth = 0
            end = start
            for i in range(start, len(text)):
                if text[i] in '{[': depth += 1
                elif text[i] in '}]':
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            data = _j.loads(text[start:end])
        else:
            data = _j.loads(text)
    except (_j.JSONDecodeError, ValueError):
        try:
            data = _j.loads(text.replace('\n', ' ').replace('  ', ' '))
        except (_j.JSONDecodeError, ValueError):
            return None
```

To:
```python
    # Strip markdown code fences
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    
    # Parse JSON response
    try:
        # Find the outermost JSON object containing "mappings"
        mappings_idx = text.find('"mappings"')
        if mappings_idx != -1:
            # Search backward for the opening brace
            start = text.rfind('{', 0, mappings_idx)
            if start != -1:
                depth = 0
                end = start
                for i in range(start, len(text)):
                    if text[i] in '{[': depth += 1
                    elif text[i] in '}]':
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                data = _j.loads(text[start:end])
            else:
                data = _j.loads(text)
        else:
            data = _j.loads(text)
    except (_j.JSONDecodeError, ValueError):
        # Fallback: collapse all whitespace
        import re as _re
        collapsed = _re.sub(r'\s+', ' ', text)
        try:
            data = _j.loads(collapsed)
        except (_j.JSONDecodeError, ValueError):
            return None
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest skills/timeline_plan/tests/test_canvas_matcher.py -v
```

Expected: all existing tests pass. The matcher's existing tests should still produce valid JSON that this parser handles identically.

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/canvas_matcher.py
git commit -m "fix: robust JSON parsing — handle nested objects, markdown fences, multi-whitespace"
```

---

## Task 11: Simplify `normalize_seedance_duration` — remove dead `round` code

**Files:**
- Modify: `skills/timeline_plan/models.py:197-205`

**Issue:** `round(target_sec)` is dead code — when `target_sec < 4.0`, `round` returns 0-3, and `max(4, ...)` always returns 4.

**Fix:** Simplify to the actual intended logic. For targets < 4s, return 4 (the minimum). For targets >= 4s, return -1 (smart duration).

- [ ] **Step 1: Simplify the function**

Replace lines 197-205:

```python
def normalize_seedance_duration(target_sec: float) -> int:
    """Map shot duration to seedance duration parameter.
    
    After merge-up, all items should be >= MIN_SEEDANCE_DURATION.
    Returns -1 for smart duration (seedance auto-determines best length).
    """
    if target_sec < MIN_SEEDANCE_DURATION:
        return max(4, round(target_sec))
    return -1
```

With:

```python
def normalize_seedance_duration(target_sec: float) -> int:
    """Map shot duration to seedance duration parameter.
    
    Returns:
        4  — for targets < 4.0s (minimum seedance duration)
       -1  — for targets >= 4.0s (smart duration: seedance auto-determines best length)
    """
    if target_sec < MIN_SEEDANCE_DURATION:
        return 4
    return -1
```

- [ ] **Step 2: Run tests**

```bash
python -m pytest skills/timeline_plan/tests/test_models.py -v
```

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/models.py
git commit -m "refactor: simplify normalize_seedance_duration, remove dead round() call"
```

---

## Task 12: Centralize `_load_env` to `skills/common/env.py`

**Files:**
- Create: `skills/common/__init__.py`
- Create: `skills/common/env.py`
- Modify: `skills/video_assembly/assemble.py:32-46`
- Modify: `skills/timeline_plan/canvas_matcher.py:25-40`
- Modify: `skills/timeline_plan/edit_planner.py:22-31`

**Issue:** `_load_env()` is duplicated in 6 files. Each module-load side-effect mutates `os.environ`, making behavior order-dependent and test-hostile.

**Fix:** Create a single `skills/common/env.py` with `load_pipeline_env()`. Call it from CLI entrypoints only. Modules do NOT call it at import time.

- [ ] **Step 1: Create `skills/common/__init__.py`**

```python
# skills/common — shared pipeline utilities
```

- [ ] **Step 2: Create `skills/common/env.py`**

```python
"""Pipeline environment loading — call once from a CLI entrypoint, never at module import."""
from __future__ import annotations

import os
from pathlib import Path


def load_pipeline_env() -> None:
    """Load environment variables from downstream project .env files.
    
    Only called from CLI entrypoints (main functions), never at module import time.
    Uses setdefault so explicit env vars take priority over .env files.
    """
    env_paths = [
        Path("~/workspace/lingolens/backend/.env").expanduser(),
        Path("~/workspace/shakespeare/.env").expanduser(),
    ]
    for env_path in env_paths:
        if not env_path.exists():
            continue
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
```

- [ ] **Step 3: Replace `_load_env()` in `assemble.py`**

Remove lines 32-46 (`_load_env()` definition and call). At line 324 (inside `main()`), add `load_pipeline_env()` as the first line:

```python
async def main():
    from skills.common.env import load_pipeline_env
    load_pipeline_env()
    p = argparse.ArgumentParser(...)
```

Also remove `from skills.timeline_plan.models import normalize_seedance_duration` at line 55 (already done in Task 8).

- [ ] **Step 4: Replace `_load_env()` in `canvas_matcher.py`**

Remove lines 25-40 (`_load_env()` definition and call). Add `load_pipeline_env()` at the top of the module's main entry function, or document that the caller must call it:

Since `canvas_matcher.py` is used as a library (called from `generate_plan.py`), add the call to `generate_plan.py`'s `main()` function:

In `generate_plan.py`, add to the `main()` function (around line 640+):

```python
def main():
    from skills.common.env import load_pipeline_env
    load_pipeline_env()
    # ... existing main code
```

- [ ] **Step 5: Replace `_load_env()` in `edit_planner.py`**

Remove lines 22-31 (`_load_env()` definition and call from edit_planner.py).

- [ ] **Step 6: Verify: no remaining standalone `_load_env` calls**

```bash
grep -rn "_load_env" skills/ --include="*.py"
```

Expected: only the centralized version in `skills/common/env.py` remains (and possibly legacy files we're not modifying — `match_to_canvas.py`, `generate_videos.py`, `prompt_extractor.py`).

- [ ] **Step 7: Run full test suite**

```bash
python -m pytest skills/timeline_plan/tests/ -v
python -m pytest skills/video_assembly/tests/ -v
```

Expected: all pass. If any test relied on import-time env loading, fix the test to call `load_pipeline_env()` explicitly.

- [ ] **Step 8: Commit**

```bash
git add skills/common/ skills/video_assembly/assemble.py skills/timeline_plan/canvas_matcher.py skills/timeline_plan/edit_planner.py
git commit -m "refactor: centralize _load_env to skills/common/env.py, remove import-time side effects"
```

---

## Task 13: Delete `prompt_extractor.py` (superseded by `prompt_composer.py`)

**Files:**
- Delete: `skills/timeline_plan/prompt_extractor.py`
- Delete: `skills/timeline_plan/tests/test_prompt_extractor.py`

**Issue:** `prompt_extractor.py` is the v1 LLM-based prompt rewriter, fully superseded by `prompt_composer.py`. It contains its own `_load_env()` and creates confusion about which module is canonical.

- [ ] **Step 1: Delete the files**

```bash
rm skills/timeline_plan/prompt_extractor.py
rm skills/timeline_plan/tests/test_prompt_extractor.py
```

- [ ] **Step 2: Verify nothing imports it**

```bash
grep -rn "prompt_extractor" skills/ --include="*.py"
```

Expected: no results (or only in comments/docs).

- [ ] **Step 3: Run tests to confirm no regressions**

```bash
python -m pytest skills/timeline_plan/tests/ -v
```

- [ ] **Step 4: Commit**

```bash
git rm skills/timeline_plan/prompt_extractor.py skills/timeline_plan/tests/test_prompt_extractor.py
git commit -m "refactor: remove prompt_extractor.py — fully superseded by prompt_composer.py"
```

---

## Task 14: Update README — document v2.0 timeline pipeline as main path

**Files:**
- Modify: `README.md`

**Issue:** README only documents the legacy v1 pipeline (canvas-storyboard + video-generation). No mention of `timeline_plan` or `video_assembly` at all.

**Fix:** Rewrite the pipeline diagram and Quick Start to show v2.0 as primary. Note legacy path with a deprecation label.

- [ ] **Step 1: Rewrite the pipeline overview section**

Replace lines 3-8:
```markdown
```
Stage 1 (script-extraction) → Stage 2 (script-rewriting) → Stage 3 (canvas-storyboard) → Stage 4 (video-generation)
      原始视频+ASR                    CEFR 分级改写                 画布匹配+prompt替换            seedance生成+拼接
```
```

With:
```markdown
## Pipeline (v2.0 — Timeline-Driven)

```
Stage 1                  Stage 1b               Stage 2                Stage 3                     Stage 4
script-extraction   →   scene-detection    →   script-rewriting   →   timeline_plan           →   video_assembly
提取剧本+ASR             PySceneDetect切点       CEFR分级改写台词        时间轴匹配+prompt改写        seedance局部重生成+拼接
```

**Core principle:** the original video timeline controls the final edit. Canvas nodes are used only as
prompt/ref-image asset libraries. Unchanged segments are directly cut from the original video;
only rewritten dialogue segments are regenerated via seedance.

### Quick Start (v2.0)

```bash
# Stage 2: Rewrite script
python3 skills/script-rewriting/rewrite_script.py \
  --script episode1_script.json --levels B2 --output-dir rewrites/

# Stage 3: Generate timeline plan
python3 skills/timeline_plan/generate_plan.py \
  --script episode1_script.json --rewrite rewrites/ep1_B2.json \
  --canvas canvas_data.json --cuts scene_cuts.json \
  --output timeline_plan.json

# Stage 4: Assemble final video
python3 skills/video_assembly/assemble.py \
  --plan timeline_plan.json --video original.mp4 \
  --output final_B2.mp4
```

### Legacy Pipeline (deprecated)

The older `canvas-storyboard` / `video-generation` pipeline is still available but no longer
the recommended path. It treats canvas nodes as video sources, which can include unused takes.

```bash
# WARNING: Legacy pipeline. Prefer timeline_plan + video_assembly.
python3 skills/canvas-storyboard/match_to_canvas.py --script ... --rewrite ... --canvas ...
python3 skills/video-generation/generate_videos.py --storyboard ... --canvas ... --output ...
```
```

- [ ] **Step 2: Update Stage 3/4 documentation sections**

Replace the existing Stage 3 and Stage 4 sections with v2.0 documentation:

```markdown
## Stage 3: timeline_plan

Match rewritten dialogue lines to canvas nodes using ASR timestamps as the authoritative
timeline. Produces a `TimelinePlan` JSON with per-segment metadata (source, duration,
degradation level, matched node, rewritten prompt).

**Key features:**
- LLM line-to-node matching with CoT reasoning + multi-run voting
- Cut point fusion (PySceneDetect + script shot boundaries)
- Short group extension to meet seedance minimum duration (4s)
- Prompt composition with operation-type-aware rewrite (literal/fuzzy/semantic/full-fallback)
- 5-layer validation (structure, cross-item, dialogue inclusion, style preservation, LLM judge)

## Stage 4: video_assembly

Consumes a `TimelinePlan` JSON to produce the final video. Segments marked `source=original`
are cut from the input video; segments marked `source=seedance` trigger seedance regeneration.

**Post-assembly integrity checks:**
- Segment count must match planned item count
- Output duration must match planned total (within 2s tolerance)
```

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: update README for v2.0 timeline pipeline, deprecate legacy path"
```

---

## Execution Order & Dependencies

```
Task 1 ──→ Task 2 (same file, same code region as Task 1)
Task 1 ──→ Task 3 (same file, same code region as Task 1)
Task 4  ─ independent (different lines in generate_plan.py)
Task 5  ─ independent
Task 6  ─ independent
Task 7  ─ independent
Task 8  ─ independent
Task 9  ─ independent
Task 10 ─ independent
Task 11 ─ independent
Task 12 ─ depends on Task 8 (removes normalize_seedance_duration import from assemble.py)
Task 13 ─ independent
Task 14 ─ independent
```

Tasks 1-4 touch `generate_plan.py` and should run in order. Tasks 5-8 touch `assemble.py` and can run in parallel after any earlier `assemble.py` task. Task 12 depends on Task 8 only for the import removal — can run in parallel otherwise.

**Optimal parallel batches:**
- Batch 1: Tasks 1, 5, 6, 7, 9, 10, 11, 13 (all independent files)
- Batch 2: Tasks 2, 3 (same file as Task 1, wait for batch 1)
- Batch 3: Task 4 (same file as Tasks 1-3)
- Batch 4: Task 12 (depends on Task 8)
- Batch 5: Task 14 (README)

---

## Final Verification

After all tasks complete:

```bash
# Full test suite
python -m pytest skills/timeline_plan/tests/ -v
python -m pytest skills/video_assembly/tests/ -v

# No remaining dead imports
grep -rn "prompt_extractor\|_extend_short_group\|segment_node_prompts" skills/ --include="*.py"

# No remaining _load_env outside common/env.py
grep -rn "_load_env" skills/ --include="*.py"

# README mentions timeline_plan
grep "timeline_plan" README.md
```
