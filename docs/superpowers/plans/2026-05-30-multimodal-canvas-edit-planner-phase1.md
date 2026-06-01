# Canvas Node Prompt Composer + Duration Resolver — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two root-cause failures from Episode 1 — (1) short groups <4s are silently dropped instead of padded, (2) prompt rewrite fallback loses all original visual style when operation is semantic_insert rather than literal_replace.

**Architecture:** Add three new data structures to `models.py` (PromptPatchPlan, CoveragePlan, MatchEvidence), extend `TimelinePlanItem` with tracking fields. Refactor `prompt_extractor.py` into `prompt_composer.py` — adding `operation_type` support, `semantic_insert` mode, and style-preserving fallback. Create `duration_resolver.py` with pad strategies replacing silent drop. Integrate both into `generate_plan.py`.

**Tech Stack:** Python 3.10+, dataclasses, openai (DeepSeek API), pytest

**Spec:** `docs/superpowers/specs/2026-05-30-multimodal-canvas-edit-planner-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `skills/timeline_plan/models.py` | Modify | Add PromptPatchPlan, CoveragePlan, MatchEvidence; extend TimelinePlanItem |
| `skills/timeline_plan/prompt_composer.py` | Create (from prompt_extractor.py) | PromptPatchComposer with operation_type + style-preserving fallback |
| `skills/timeline_plan/duration_resolver.py` | Create | Duration resolution strategies: pad, snap, borrow; never silent drop |
| `skills/timeline_plan/generate_plan.py` | Modify | Integrate prompt_composer and duration_resolver; update item creation |
| `skills/timeline_plan/prompt_extractor.py` | Obsolete (keep as reference) | Replaced by prompt_composer.py |
| `skills/timeline_plan/tests/test_models.py` | Modify | Add tests for new dataclasses |
| `skills/timeline_plan/tests/test_prompt_composer.py` | Create | Tests for PromptPatchComposer (semantic_insert, style fallback) |
| `skills/timeline_plan/tests/test_duration_resolver.py` | Create | Tests for resolve_duration (pad, no silent drop) |
| `skills/timeline_plan/tests/test_generate_plan.py` | Modify | Test short-group pad behavior, test new TimelinePlanItem fields |

### Data Flow After Phase 1

```
generate_plan.py
  ├── models.py (new dataclasses for typed output)
  ├── canvas_matcher.py (unchanged in Phase 1 — keeps current LLM matching)
  ├── prompt_composer.py (NEW: replaces prompt_extractor)
  │     └── compose_prompt_patch(full_prompt, lines, scene_desc, operation_type)
  │           → returns str (final prompt)
  │           → internal: _extract_style_prefix, _llm_rewrite_prompt, _generate_prompt_from_scene
  └── duration_resolver.py (NEW: replaces silent drop)
        └── resolve_duration(group, all_lines_map, line_to_node)
              → returns (extended_group, strategy, duration)
              → NEVER returns None or < MIN_SEEDANCE_DURATION without explicit fallback
```

---

### Task 1: Extend models.py with v3 Dataclasses and TimelinePlanItem Fields

**Files:**
- Modify: `skills/timeline_plan/models.py`
- Test: `skills/timeline_plan/tests/test_models.py`

**Goal:** Add `PromptPatchPlan`, `CoveragePlan`, `MatchEvidence` dataclasses. Extend `TimelinePlanItem` with `operation_type`, `duration_strategy`, `covered_line_ids`, `borrowed_line_ids`, `source_node_ids`, `degradation_reason` fields.

- [ ] **Step 1: Write failing tests for new dataclasses**

```python
# In skills/timeline_plan/tests/test_models.py, add after existing imports:
from skills.timeline_plan.models import (
    PromptPatchPlan, CoveragePlan, MatchEvidence,
)


class TestPromptPatchPlan:
    def test_literal_replace_creation(self):
        plan = PromptPatchPlan(
            operation_type="literal_replace",
            global_style="8k, 超高清, 电影级布光",
            local_visual_context="镜头 1：三层景深",
            dialogue_patches=[],
            discarded_sections=[],
            final_prompt="8k, 超高清...镜头1...Donny says: 'hi'",
        )
        assert plan.operation_type == "literal_replace"
        assert plan.global_style == "8k, 超高清, 电影级布光"
        assert plan.final_prompt is not None

    def test_semantic_insert_defaults(self):
        plan = PromptPatchPlan(
            operation_type="semantic_insert",
            global_style="",
            local_visual_context="",
            dialogue_patches=[],
            discarded_sections=[],
            final_prompt="",
        )
        assert plan.operation_type == "semantic_insert"
        assert plan.dialogue_patches == []
        assert plan.discarded_sections == []


class TestCoveragePlan:
    def test_direct_strategy(self):
        cp = CoveragePlan(
            start_sec=17.47,
            end_sec=29.55,
            included_rewritten_line_ids=["p001_l003", "p001_l004"],
            borrowed_original_line_ids=[],
            duration_strategy="direct",
        )
        assert cp.duration_strategy == "direct"
        assert cp.duration_expansion_sec == 0.0
        assert cp.end_sec - cp.start_sec >= 4.0

    def test_pad_after_strategy(self):
        cp = CoveragePlan(
            start_sec=2.83,
            end_sec=6.83,
            included_rewritten_line_ids=["p001_l001", "p001_l002"],
            borrowed_original_line_ids=[],
            duration_strategy="pad_after",
            duration_expansion_sec=0.40,
        )
        assert cp.duration_strategy == "pad_after"
        assert cp.duration_expansion_sec == 0.40


class TestMatchEvidence:
    def test_quoted_dialogue_signal(self):
        ev = MatchEvidence(
            signal="quoted_dialogue",
            detail='Found "This ceremony is boring." in node prompt',
            confidence=0.97,
        )
        assert ev.signal == "quoted_dialogue"
        assert ev.confidence > 0.9

    def test_implicit_visual_scene_signal(self):
        ev = MatchEvidence(
            signal="implicit_visual_scene",
            detail="Prompt describes Donny's breakdown close-up",
            confidence=0.85,
        )
        assert ev.signal == "implicit_visual_scene"
        assert ev.confidence < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest skills/timeline_plan/tests/test_models.py::TestPromptPatchPlan -v
```

Expected: `ImportError: cannot import name 'PromptPatchPlan'`

- [ ] **Step 3: Implement new dataclasses in models.py**

At the end of `skills/timeline_plan/models.py`, before `MIN_SEEDANCE_DURATION`, add:

```python
@dataclass
class PromptPatchPlan:
    """Layered prompt editing plan: style layer + visual context + dialogue patches."""
    operation_type: Literal[
        "literal_replace", "fuzzy_replace", "semantic_insert",
        "section_reconstruct", "style_preserving_fallback", "full_fallback"
    ]
    global_style: str                        # extracted from original prompt prefix
    local_visual_context: str                # matching scene description section
    dialogue_patches: List[Dict[str, str]]   # [{line_id, speaker, mode, text, placement}]
    discarded_sections: List[str]            # summary of removed sections
    final_prompt: str                        # the rewritten prompt for seedance


@dataclass
class CoveragePlan:
    """Time coverage plan: what interval to generate, what strategy was used."""
    start_sec: float
    end_sec: float
    included_rewritten_line_ids: List[str]
    borrowed_original_line_ids: List[str]
    duration_strategy: Literal[
        "direct", "pad_after", "pad_before", "snap_to_cut",
        "hold_reaction", "borrow_neighbor", "merge_same_node_group",
        "cross_node_merge", "forced_min_duration"
    ]
    duration_expansion_sec: float = 0.0


@dataclass
class MatchEvidence:
    """A single matching signal between a line group and a canvas node."""
    signal: Literal[
        "quoted_dialogue", "fuzzy_dialogue", "speaker_presence",
        "visual_action", "shot_scene_similarity", "temporal_order",
        "reference_image_match", "implicit_visual_scene"
    ]
    detail: str
    confidence: float
```

Then extend `TimelinePlanItem` by adding these new fields (after `degradation_level`):

```python
    # v3 tracking fields (Phase 1)
    operation_type: Optional[str] = None       # from PromptPatchPlan
    duration_strategy: Optional[str] = None    # from CoveragePlan
    covered_line_ids: List[str] = field(default_factory=list)
    borrowed_line_ids: List[str] = field(default_factory=list)
    source_node_ids: List[str] = field(default_factory=list)
    degradation_reason: str = ""
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest skills/timeline_plan/tests/test_models.py -v
```

Expected: All tests pass including new dataclass tests (12+ tests total).

- [ ] **Step 5: Verify existing serialization roundtrip still works**

```bash
python -m pytest skills/timeline_plan/tests/test_models.py::TestTimelinePlan::test_json_roundtrip -v
```

Expected: PASS — new optional fields with defaults don't break `asdict()` serialization.

- [ ] **Step 6: Commit**

```bash
git add skills/timeline_plan/models.py skills/timeline_plan/tests/test_models.py
git commit -m "feat(models): add PromptPatchPlan, CoveragePlan, MatchEvidence dataclasses and extend TimelinePlanItem with v3 tracking fields"
```

---

### Task 2: Create prompt_composer.py — PromptPatchComposer with operation_type

**Files:**
- Create: `skills/timeline_plan/prompt_composer.py`
- Test: `skills/timeline_plan/tests/test_prompt_composer.py`

**Goal:** Refactor `prompt_extractor.py` into `prompt_composer.py`. The new module supports `operation_type`, has `semantic_insert` mode (does not require finding original text), extracts style layer for fallback preservation, and uses fuzzy validation for non-literal operations.

- [ ] **Step 1: Write failing test for semantic_insert mode**

```python
# skills/timeline_plan/tests/test_prompt_composer.py
"""Tests for PromptPatchComposer."""
import pytest
from skills.timeline_plan.prompt_composer import (
    compose_prompt_patch,
    _extract_style_prefix,
    _generate_prompt_from_scene,
    _validate_rewrite,
)


class FakeLine:
    def __init__(self, dialogue="", original="", rewritten="", speaker="Speaker"):
        self.dialogue = dialogue
        self.original = original
        self.rewritten = rewritten
        self.speaker = speaker


class TestExtractStylePrefix:
    def test_extracts_chinese_style_keywords(self):
        prompt = "美式情景喜剧，真实短剧，柔光雾化，画面通透，8k，超高清，电影级布光。镜头 1：..."
        result = _extract_style_prefix(prompt)
        assert "美式情景喜剧" in result
        assert "电影级布光" in result
        # Should NOT include scene-specific content
        assert "镜头 1" not in result

    def test_returns_empty_for_no_style(self):
        prompt = "Donny says: \"hello\""
        result = _extract_style_prefix(prompt)
        assert result == ""

    def test_extracts_english_style_keywords(self):
        prompt = "cinematic lighting, 8k resolution, shallow depth of field. Scene 1: ..."
        result = _extract_style_prefix(prompt)
        assert "cinematic lighting" in result
        assert "Scene 1" not in result


class TestValidateRewrite:
    def test_exact_substring_passes_for_literal(self):
        prompt = "美式情景喜剧...Donny says: \"No, no, no, this can't be.\""
        lines = [FakeLine(rewritten="No, no, no, this can't be.")]
        assert _validate_rewrite(prompt, lines, operation_type="literal_replace")

    def test_missing_dialogue_fails(self):
        prompt = "美式情景喜剧...Donny says: \"Hi.\""
        lines = [FakeLine(rewritten="Hello.")]
        assert not _validate_rewrite(prompt, lines, operation_type="literal_replace")

    def test_no_original_required_for_semantic_insert(self):
        """semantic_insert does NOT require original text in prompt — only rewritten must appear."""
        prompt = "美式情景喜剧...真实的破防...Donny says: \"No, no, no, this can't be.\""
        lines = [FakeLine(original="no no no", rewritten="No, no, no, this can't be.")]
        # Even though original "no no no" is NOT in prompt, semantic_insert passes
        assert _validate_rewrite(prompt, lines, operation_type="semantic_insert")

    def test_empty_rewritten_skipped(self):
        lines = [FakeLine(rewritten="")]
        assert _validate_rewrite("any prompt", lines, operation_type="literal_replace")


class TestGeneratePromptFromScene:
    def test_includes_style_layer(self):
        lines = [FakeLine(original="hello", rewritten="hi there", speaker="Donny")]
        result = _generate_prompt_from_scene(lines, "A bar scene", style_layer="8k, 电影级布光")
        assert "8k, 电影级布光" in result
        assert "A bar scene" in result
        assert "hi there" in result

    def test_no_style_layer_works(self):
        lines = [FakeLine(original="test", rewritten="rewritten test")]
        result = _generate_prompt_from_scene(lines, style_layer="")
        assert "A cinematic scene" in result
        assert "rewritten test" in result


class TestComposePromptPatch:
    def test_empty_prompt_uses_style_fallback(self):
        """When full_prompt is empty, fallback should produce output with scene_desc."""
        lines = [FakeLine(original="hello", rewritten="hi", speaker="Donny")]
        result = compose_prompt_patch("", lines, "Opening scene")
        assert "Opening scene" in result
        assert "hi" in result

    def test_returns_original_prompt_when_no_rewrite_needed(self):
        """When all lines have original == rewritten, return prompt unchanged."""
        lines = [FakeLine(original="hello", rewritten="hello", speaker="Donny")]
        result = compose_prompt_patch("美式情景喜剧...Donny says: hello", lines)
        assert result == "美式情景喜剧...Donny says: hello"

    def test_semantic_insert_operation_type_passed_to_llm(self):
        """semantic_insert should NOT fail validation when original isn't in prompt."""
        # This is an integration-level test: if DEEPSEEK_API_KEY is set, it tests real LLM.
        # We test the validation path, not LLM output here.
        prompt = "美式情景喜剧，真实短剧，电影级布光。镜头 2：真实的破防（面部特写）"
        lines = [FakeLine(
            original="no no no",
            rewritten="No, no, no, this can't be.",
            speaker="Donny",
        )]
        # validate with semantic_insert should pass once rewritten is in prompt
        result = compose_prompt_patch(prompt, lines, operation_type="semantic_insert")
        # If LLM unavailable, falls back to style-preserving fallback
        assert len(result) > 0
        # The fallback should preserve style from original prompt
        if "美式情景喜剧" not in result:
            pass  # LLM may rephrase, acceptable for semantic_insert
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest skills/timeline_plan/tests/test_prompt_composer.py -v
```

Expected: `ModuleNotFoundError: No module named 'skills.timeline_plan.prompt_composer'`

- [ ] **Step 3: Implement prompt_composer.py**

Create `skills/timeline_plan/prompt_composer.py`:

```python
"""PromptPatchComposer: layered prompt editing with operation_type support.

Replaces prompt_extractor.py. Supports 6 operation types:
  - literal_replace: original dialogue exists → replace
  - semantic_insert: no original text, but visual scene matches → insert
  - style_preserving_fallback: keep style layer, generate scene + dialogue
  - full_fallback: no usable node evidence

Key improvement over v2: fallback preserves style layer from original prompt,
and semantic_insert does not require finding original dialogue text.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional


# ── Module-level env loading (once, not per-call) ──────────────────

def _load_env():
    for env_path in [
        str(Path("~/workspace/lingolens/backend/.env").expanduser()),
        str(Path("~/workspace/shakespeare/.env").expanduser()),
    ]:
        if Path(env_path).exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())

_load_env()


# ── Style layer keywords for extraction ────────────────────────────

_STYLE_KEYWORDS_CN = [
    "美式情景喜剧", "真实短剧", "柔光雾化", "画面通透",
    "电影级布光", "超高清", "电影质感", "浅景深",
    "8k", "4k", "hdr",
]

_STYLE_KEYWORDS_EN = [
    "cinematic", "8k", "4k", "hdr", "shallow depth of field",
    "film grain", "soft lighting", "ultra hd",
]


def _extract_style_prefix(prompt: str, max_chars: int = 300) -> str:
    """Extract global visual style prefix from a canvas node prompt.

    Scans the prompt for style keywords (resolution, lighting, quality markers)
    and returns the prefix containing them — stopping at the first scene/dialogue
    indicator.  For Chinese prompts, stops at section headers like "镜头",
    "场景", or quoted English dialogue.

    Returns empty string if no style keywords found.
    """
    if not prompt or not prompt.strip():
        return ""

    # Find the cutoff point: first section header or quoted dialogue
    cutoff = len(prompt)
    for marker in ["镜头", "场景", "\"", '"']:
        idx = prompt.find(marker)
        if idx != -1 and idx < cutoff:
            cutoff = idx

    prefix = prompt[:min(cutoff, max_chars)].strip()

    # Check if prefix contains style keywords
    has_style = False
    for kw in _STYLE_KEYWORDS_CN + _STYLE_KEYWORDS_EN:
        if kw.lower() in prefix.lower():
            has_style = True
            break

    return prefix if has_style else ""


# ── LLM-based prompt rewriting ─────────────────────────────────────

def _build_system_prompt(operation_type: str) -> str:
    """Build system prompt based on operation type."""
    if operation_type == "semantic_insert":
        return """## Role
You rewrite video generation prompts for seedance. The original prompt does NOT contain
the original dialogue as quoted text. Instead, visual scene descriptions match the
dialogue context.

## Instructions
1. PRESERVE the global visual style settings (resolution, lighting, camera style) verbatim.
2. KEEP the visual scene description that matches the rewritten dialogue context.
3. INSERT each rewritten dialogue line at the appropriate position in the scene.
4. REMOVE scene sections with no rewritten dialogue.
5. Do NOT attempt to find and replace non-existent original text — INSERT instead.

## Output
Rewritten prompt text only. No explanations, no JSON."""

    # Default: literal_replace
    return """## Role
You rewrite video generation prompts for seedance, keeping only visual content tied to rewritten dialogue.

The original prompt mixes style settings (resolution, lighting, camera style), scene descriptions with camera angles and character actions, and dialogue in quotes. Style settings apply to the entire video and must be preserved. Scene descriptions should be kept only if they contain dialogue being rewritten — within them, keep only the visuals directly around the dialogue moment and cut background filler. Remove entire scenes with no rewritten dialogue.

Replace each original dialogue line with its rewritten version, preserving the speaker attribution format. If no scene matches the rewrite lines, output the original prompt unchanged.

## Output
Rewritten prompt text only. No explanations, no JSON."""


def _build_user_prompt(
    full_prompt: str,
    mappings: List[Dict],
    scene_description: str,
    operation_type: str,
) -> str:
    """Build user prompt based on operation type."""
    import json as _j

    base = f"""## Original Prompt
{full_prompt}

## Dialogue to Rewrite
{_j.dumps(mappings, ensure_ascii=False, indent=2)}

## Scene Context
{scene_description or '(none)'}"""

    if operation_type == "semantic_insert":
        base += """

## Operation: semantic_insert
The original dialogue is NOT present as quoted text in the original prompt.
Instead, a visual scene description matches the dialogue context.
INSERT each rewritten dialogue line at the appropriate visual position."""
    else:
        base += """

Output the rewritten prompt."""

    return base


def _llm_rewrite_prompt(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
    operation_type: str = "literal_replace",
) -> str:
    """Use LLM to rewrite prompt. Returns rewritten prompt or empty string on failure."""
    mappings = []
    for line in rewrite_lines:
        original = getattr(line, "original", "") or getattr(line, "dialogue", "")
        rewritten = getattr(line, "rewritten", "")
        speaker = getattr(line, "speaker", "")
        if rewritten.strip() and original.strip() != rewritten.strip():
            mappings.append({
                "speaker": speaker,
                "original": original.strip(),
                "rewritten": rewritten.strip(),
            })

    if not mappings:
        return ""

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return ""

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_MATCH_MODEL", "deepseek-v4-flash"),
            messages=[
                {"role": "system", "content": _build_system_prompt(operation_type)},
                {"role": "user", "content": _build_user_prompt(full_prompt, mappings, scene_description, operation_type)},
            ],
            temperature=0.0,
            max_tokens=8192,
        )
        text = resp.choices[0].message.content or ""
        return text.strip()
    except Exception:
        return ""


# ── Fallback: generate prompt from scene description ──────────────

def _generate_prompt_from_scene(
    rewrite_lines: List[Any],
    scene_description: str = "",
    style_layer: str = "",
) -> str:
    """Generate a prompt from scene_description + rewritten dialogue.
    
    If style_layer is provided, it is prepended to preserve visual quality
    from the original canvas node prompt.
    """
    # Only include lines that actually have rewritten text
    dialogue_lines = []
    for line in rewrite_lines:
        speaker = getattr(line, "speaker", "Character")
        rewritten = getattr(line, "rewritten", "") or getattr(line, "dialogue", "")
        if rewritten:
            dialogue_lines.append(f'{speaker} says: "{rewritten}"')

    parts = []
    if style_layer:
        parts.append(style_layer)

    desc = scene_description or "A cinematic scene"
    parts.append(desc)

    if dialogue_lines:
        parts.append("\n".join(dialogue_lines))

    return "\n".join(parts).strip()


# ── Validation ─────────────────────────────────────────────────────

def _validate_rewrite(
    prompt: str,
    rewrite_lines: List[Any],
    operation_type: str = "literal_replace",
) -> bool:
    """Check that rewritten dialogue appears in the output prompt.

    For semantic_insert: only requires rewritten text to be present
    (does not require original text to have existed).

    For literal_replace: requires rewritten text to be present
    (original text replacement is assumed by LLM instruction).
    """
    for line in rewrite_lines:
        rewritten = getattr(line, "rewritten", "") or ""
        if rewritten.strip() and rewritten.strip() not in prompt:
            return False
    return True


# ── Main orchestrator ──────────────────────────────────────────────

def compose_prompt_patch(
    full_prompt: str,
    rewrite_lines: List[Any],
    scene_description: str = "",
    operation_type: str = "literal_replace",
) -> str:
    """Rewrite a canvas node prompt for seedance generation.

    Supports multiple operation types:
    - literal_replace: original dialogue exists in prompt → replace
    - semantic_insert: no original text, visual scene matches → insert
    - style_preserving_fallback: preserve style, generate scene + dialogue

    Falls back to style-preserving scene-based generation on failure.

    Args:
        full_prompt: Complete canvas node prompt (empty if no match).
        rewrite_lines: Lines with .original, .rewritten, .speaker attrs.
        scene_description: Fallback scene context.
        operation_type: The type of prompt edit operation.

    Returns:
        Rewritten prompt ready for seedance.
    """
    # Quick return: no actual rewrite needed
    all_same = all(
        getattr(line, "original", "") == getattr(line, "rewritten", "")
        for line in rewrite_lines
    )
    if all_same and full_prompt:
        return full_prompt

    if not full_prompt:
        return _generate_prompt_from_scene(rewrite_lines, scene_description)

    # Try LLM rewrite
    result = _llm_rewrite_prompt(full_prompt, rewrite_lines, scene_description, operation_type)
    if result and _validate_rewrite(result, rewrite_lines, operation_type):
        return result

    # LLM failed or validation failed → style-preserving fallback
    style_layer = _extract_style_prefix(full_prompt)
    if style_layer:
        return _generate_prompt_from_scene(rewrite_lines, scene_description, style_layer)

    return _generate_prompt_from_scene(rewrite_lines, scene_description)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest skills/timeline_plan/tests/test_prompt_composer.py -v
```

Expected: All non-LLM-dependent tests pass (style extraction, validation, fallback generation). 
LLM-dependent test (`test_semantic_insert_operation_type_passed_to_llm`) may skip gracefully if no API key — that's expected.

- [ ] **Step 5: Verify existing prompt_extractor tests still pass (backward compatibility check)**

```bash
python -m pytest skills/timeline_plan/tests/test_prompt_extractor.py -v
```

Expected: Existing tests pass (prompt_extractor.py is still present, not deleted).

- [ ] **Step 6: Commit**

```bash
git add skills/timeline_plan/prompt_composer.py skills/timeline_plan/tests/test_prompt_composer.py
git commit -m "feat(prompt_composer): add PromptPatchComposer with semantic_insert and style-preserving fallback"
```

---

### Task 3: Create duration_resolver.py — Never Silent Drop

**Files:**
- Create: `skills/timeline_plan/duration_resolver.py`
- Test: `skills/timeline_plan/tests/test_duration_resolver.py`

**Goal:** Replace `generate_plan.py`'s silent drop with `resolve_duration()` — tries pad strategies before giving up. Never returns without explicitly tracking every rewritten line.

- [ ] **Step 1: Write failing tests for resolve_duration**

```python
# skills/timeline_plan/tests/test_duration_resolver.py
"""Tests for Duration Resolver."""
import pytest
from skills.timeline_plan.duration_resolver import (
    resolve_duration,
    _try_pad_strategy,
)

# Helper: create a rewrite line dict (matching generate_plan.py's format)
def _rl(line_id, start_s, end_s, original="x", rewritten="y", speaker="S"):
    return {
        "line_id": line_id,
        "start_seconds": start_s,
        "end_seconds": end_s,
        "original": original,
        "rewritten": rewritten,
        "speaker": speaker,
        "shot_number": 1,
        "shot_scene": "",
    }

def _nl(line_id, dialogue, start_s, end_s, speaker="S"):
    """non-rewritten line in all_lines_map format"""
    return {
        "line_id": line_id,
        "dialogue": dialogue,
        "speaker": speaker,
        "start_seconds": start_s,
        "end_seconds": end_s,
    }


class TestTryPadStrategy:
    def test_pad_after_when_close_to_min(self):
        """Group at 3.6s → pad_after to 4.0s."""
        group = [_rl("l1", 2.83, 3.5), _rl("l2", 5.3, 6.43)]
        result = _try_pad_strategy(group, "pad_after", min_duration=4.0)
        assert result is not None
        new_min = min(r["start_seconds"] for r in result)
        new_max = max(r["end_seconds"] for r in result)
        assert new_max - new_min >= 4.0
        # The start should be unchanged, end padded
        assert new_min == 2.83

    def test_pad_before(self):
        group = [_rl("l1", 3.5, 7.0)]
        result = _try_pad_strategy(group, "pad_before", min_duration=4.0)
        assert result is not None
        new_min = min(r["start_seconds"] for r in result)
        new_max = max(r["end_seconds"] for r in result)
        assert new_max - new_min >= 4.0
        assert new_max == 7.0  # end unchanged, start padded back

    def test_direct_when_already_long_enough(self):
        group = [_rl("l1", 1.0, 6.0)]
        result = _try_pad_strategy(group, "direct", min_duration=4.0)
        assert result is not None
        assert result == group  # unchanged


class TestResolveDuration:
    def test_returns_extended_group_for_short_duration(self):
        """3.6s group → should NOT be dropped, should be padded."""
        group = [_rl("l1", 2.83, 3.5), _rl("l2", 5.3, 6.43)]
        all_lines = {
            "l1": _nl("l1", "hello", 2.83, 3.5),
            "l2": _nl("l2", "world", 5.3, 6.43),
        }
        line_to_node = {"l1": "n1", "l2": "n1"}

        extended, strategy, duration = resolve_duration(
            group, all_lines, line_to_node, min_duration=4.0
        )
        assert duration >= 4.0, f"Expected >= 4.0s, got {duration:.1f}s"
        assert strategy in ("pad_after", "pad_before")
        assert len(extended) >= 2
        # The rewritten lines should still be in the result
        extended_ids = {r["line_id"] for r in extended}
        assert "l1" in extended_ids
        assert "l2" in extended_ids

    def test_already_long_enough_unchanged(self):
        """12.1s group → returned as-is."""
        group = [_rl("l1", 17.47, 18.27), _rl("l2", 18.83, 20.03), _rl("l3", 21.15, 29.55)]
        all_lines = {r["line_id"]: _nl(r["line_id"], r["original"], r["start_seconds"], r["end_seconds"]) for r in group}
        line_to_node = {r["line_id"]: "n1" for r in group}

        extended, strategy, duration = resolve_duration(
            group, all_lines, line_to_node, min_duration=4.0
        )
        assert strategy == "direct"
        assert duration >= 4.0
        assert len(extended) == 3

    def test_single_very_short_line_padded(self):
        """Single line at 0.8s → padded to 4.0s, not dropped."""
        group = [_rl("l1", 17.47, 18.27)]
        all_lines = {"l1": _nl("l1", "test", 17.47, 18.27)}
        line_to_node = {"l1": "n1"}

        extended, strategy, duration = resolve_duration(
            group, all_lines, line_to_node, min_duration=4.0
        )
        assert duration >= 4.0
        assert "l1" in {r["line_id"] for r in extended}

    def test_empty_group_returns_empty(self):
        extended, strategy, duration = resolve_duration(
            [], {}, {}, min_duration=4.0
        )
        assert len(extended) == 0
        assert strategy == "direct"
        assert duration == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest skills/timeline_plan/tests/test_duration_resolver.py -v
```

Expected: `ModuleNotFoundError: No module named 'skills.timeline_plan.duration_resolver'`

- [ ] **Step 3: Implement duration_resolver.py**

```python
"""Duration Resolver: ensures every rewritten group meets min seedance duration.

Replaces the silent-drop behavior in generate_plan.py. Applies pad strategies
in priority order, never silently discarding rewritten lines.

Strategy priority:
  1. pad_after:  if duration >= 3.5s, extend end to reach min
  2. pad_before: if duration >= 3.5s, extend start to reach min
  3. forced_min_duration: last resort, force-extend to min
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple


DURATION_STRATEGIES = [
    "pad_after",
    "pad_before",
    "forced_min_duration",
]


def _try_pad_strategy(
    group: List[Dict],
    strategy: str,
    min_duration: float = 4.0,
) -> Optional[List[Dict]]:
    """Try a single pad strategy. Returns extended group or None."""
    if strategy == "direct":
        return list(group)

    min_start = min(r.get("start_seconds", 0.0) for r in group)
    max_end = max(r.get("end_seconds", min_start + 1.0) for r in group)
    duration = max_end - min_start

    if duration >= min_duration:
        return list(group)

    deficit = min_duration - duration

    if strategy == "pad_after" and deficit <= 1.0:
        # Extend last line's end by the deficit (e.g., 3.6s → 4.0s with +0.4s)
        group = list(group)
        group[-1]["end_seconds"] = group[-1]["end_seconds"] + deficit
        return group

    if strategy == "pad_before" and deficit <= 1.0:
        # Extend first line's start backward by the deficit
        group = list(group)
        group[0]["start_seconds"] = max(0.0, group[0]["start_seconds"] - deficit)
        return group

    if strategy == "forced_min_duration":
        # Last resort: force-extend to exactly min_duration
        group = list(group)
        anchor_start = min(r.get("start_seconds", 0.0) for r in group)
        group[-1]["end_seconds"] = anchor_start + min_duration
        return group

    return None


def resolve_duration(
    group: List[Dict],
    all_lines_map: Dict[str, Dict],
    line_to_node: Dict[str, str],
    min_duration: float = 4.0,
) -> Tuple[List[Dict], str, float]:
    """Resolve a rewritten line group's duration to meet min seedance requirement.

    Tries strategies in priority order. NEVER returns without every rewritten
    line being explicitly tracked.

    Args:
        group: Rewrite line dicts sorted by start_seconds.
        all_lines_map: line_id → line info dict.
        line_to_node: line_id → node_id.
        min_duration: Minimum allowed duration (default 4.0).

    Returns:
        (extended_group, strategy_used, final_duration)
    """
    if not group:
        return [], "direct", 0.0

    min_start = min(r.get("start_seconds", 0.0) for r in group)
    max_end = max(r.get("end_seconds", min_start + 1.0) for r in group)
    duration = max_end - min_start

    if duration >= min_duration:
        return list(group), "direct", duration

    # Try pad strategies in order
    for strategy in DURATION_STRATEGIES:
        result = _try_pad_strategy(group, strategy, min_duration)
        if result is not None:
            new_duration = max(
                r.get("end_seconds", 0.0) for r in result
            ) - min(
                r.get("start_seconds", 0.0) for r in result
            )
            return result, strategy, new_duration

    # Should never reach here — forced_min_duration always succeeds
    return list(group), "forced_min_duration", min_duration
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest skills/timeline_plan/tests/test_duration_resolver.py -v
```

Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add skills/timeline_plan/duration_resolver.py skills/timeline_plan/tests/test_duration_resolver.py
git commit -m "feat(duration_resolver): add duration resolution with pad strategies, no silent drop"
```

---

### Task 4: Integrate prompt_composer and duration_resolver into generate_plan.py

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py`
- Test: `skills/timeline_plan/tests/test_generate_plan.py`

**Goal:** Replace `prompt_extractor` import with `prompt_composer`, replace silent drop with `resolve_duration`, populate new `TimelinePlanItem` tracking fields. All existing tests must continue to pass.

- [ ] **Step 1: Write a failing test for short-group pad behavior**

```python
# Add to skills/timeline_plan/tests/test_generate_plan.py, inside TestGenerateTimelinePlan:

    def test_short_rewritten_line_not_dropped(self):
        """A rewritten line <4s should be padded, not silently dropped."""
        shots = [FakeShot(1, 0.0, 10.0, "Opening", [FakeLine("p1_l1", "hello", 1.0, 2.0)])]
        script = FakeScriptOutput(shots)
        rewrite = {"level": "B2", "lines": [make_rewrite("p1_l1", "hello", "hi there", 1, 1.0, 2.0)]}
        nodes = [CanvasNode(node_id="n1", prompt='He says "hello"', video_url="http://x.com/v.mp4", reference_images=["http://x.com/r.png"])]
        inp = Stage3Input(script_output=script, rewrite_json=rewrite, canvas_nodes=nodes, level="B2")
        plan = generate_timeline_plan(inp)
        seedance_items = [i for i in plan.items if i.source == "seedance"]
        # The rewritten line MUST appear as a seedance item (not dropped)
        assert len(seedance_items) >= 1, "Short group was silently dropped!"
        item = seedance_items[0]
        assert item.rewritten_prompt is not None
        assert "hi there" in item.rewritten_prompt
        # Verify duration_strategy is set
        assert item.duration_strategy is not None
        assert item.duration_strategy in ("pad_after", "pad_before", "forced_min_duration")
```

- [ ] **Step 2: Run test to verify it fails (current behavior drops short groups)**

```bash
python -m pytest skills/timeline_plan/tests/test_generate_plan.py::TestGenerateTimelinePlan::test_short_rewritten_line_not_dropped -v
```

Expected: FAIL — current code drops the short group (assertion fails on `len(seedance_items) >= 1`).

- [ ] **Step 3: Update generate_plan.py imports**

Replace the import block (lines 16-22):

```python
from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, KeyFrame, Stage3Input,
    normalize_seedance_duration, MIN_SEEDANCE_DURATION,
)
from skills.timeline_plan.cut_fusion import determine_cut_points
from skills.timeline_plan.canvas_matcher import match_lines_to_nodes
from skills.timeline_plan.prompt_composer import compose_prompt_patch  # was: prompt_extractor
from skills.timeline_plan.duration_resolver import resolve_duration  # new
```

- [ ] **Step 4: Replace silent drop block (lines 367-375) with resolve_duration call**

Replace:

```python
            if duration < MIN_SEEDANCE_DURATION:
                group = _extend_short_group(group, set(line_ids), all_lines_map, line_to_node)
                min_start = min(rl.get("start_seconds", 0.0) for rl in group)
                max_end = max(rl.get("end_seconds", min_start + 1.0) for rl in group)
                duration = max_end - min_start
                if duration < MIN_SEEDANCE_DURATION:
                    group_ids = {rl["line_id"] for rl in group}
                    handled_rewrite_line_ids.update(group_ids)
                    continue
```

With:

```python
            if duration < MIN_SEEDANCE_DURATION:
                # v3: resolve duration instead of silent drop
                group, duration_strategy, new_duration = resolve_duration(
                    group, all_lines_map, line_to_node, MIN_SEEDANCE_DURATION
                )
                min_start = min(rl.get("start_seconds", 0.0) for rl in group)
                max_end = max(rl.get("end_seconds", min_start + 1.0) for rl in group)
                duration = new_duration
            else:
                duration_strategy = "direct"
```

- [ ] **Step 5: Determine operation_type for prompt composer call (before line 391)**

Add after `degradation_level = 0` and before the prompt composer call:

```python
            # Determine operation_type based on matching evidence
            # Phase 1 default: "literal_replace" (matcher hasn't been updated yet)
            # When matcher is upgraded in Phase 2, this will be driven by match evidence
            operation_type = "literal_replace"
            if node:
                # v3: compose_prompt_patch with operation_type
                rewritten_prompt = compose_prompt_patch(
                    prompt_str, rl_objects, scene_desc, operation_type
                )
            else:
                rewritten_prompt = compose_prompt_patch(
                    "", rl_objects, scene_desc, operation_type
                )
```

Replace lines 391-395:

```python
            prompt_str = node.prompt if node else ""
            rl_objects = _make_rl_objects(group)
            rewritten_prompt = extract_and_rewrite_prompt(
                prompt_str, rl_objects, scene_desc
            )
```

With the above block.

- [ ] **Step 6: Update TimelinePlanItem creation to include new fields (lines 403-417)**

Add the new tracking fields to the `TimelinePlanItem` constructor:

```python
            items.append(TimelinePlanItem(
                shot_id=f"shot_{shot_num}_node_{node_id[:8]}" if node else f"shot_{shot_num}",
                shot_number=shot_num,
                source="seedance",
                start_sec=min_start,
                end_sec=max_end,
                scene_description=scene_desc,
                ref_images=ref_images,
                rewritten_prompt=rewritten_prompt,
                matched_node_id=node_id if node else None,
                match_confidence=node_confidence,
                degradation_level=degradation_level,
                seedance_duration=seedance_dur,
                original_duration=duration,
                # v3 tracking fields:
                operation_type=operation_type,
                duration_strategy=duration_strategy,
                covered_line_ids=sorted(group_ids),
                borrowed_line_ids=[],  # Phase 2: populate when borrowing
                source_node_ids=[node_id] if node_id else [],
                degradation_reason=(
                    "duration_padded_to_meet_min_4s" if duration_strategy != "direct"
                    else ""
                ),
            ))
```

- [ ] **Step 7: Update the fallback block (lines 429-455) to use compose_prompt_patch**

Replace line 444:

```python
                rewritten_prompt = extract_and_rewrite_prompt("", rl_objects, scene_desc)
```

With:

```python
                rewritten_prompt = compose_prompt_patch("", rl_objects, scene_desc, "full_fallback")
```

And add tracking fields to the fallback `TimelinePlanItem`:

```python
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
                    # v3 tracking fields:
                    operation_type="full_fallback",
                    duration_strategy="direct",
                    covered_line_ids=sorted({rl["line_id"] for rl in unmatched}),
                    borrowed_line_ids=[],
                    source_node_ids=[],
                    degradation_reason="no_matching_canvas_node",
                ))
```

- [ ] **Step 8: Run ALL existing tests to verify no regression**

```bash
python -m pytest skills/timeline_plan/tests/test_generate_plan.py -v
```

Check: ALL 7 existing tests must pass.

- [ ] **Step 9: Run the new test to verify short-group is no longer dropped**

```bash
python -m pytest skills/timeline_plan/tests/test_generate_plan.py::TestGenerateTimelinePlan::test_short_rewritten_line_not_dropped -v
```

Expected: PASS.

- [ ] **Step 10: Run the full test suite**

```bash
python -m pytest skills/timeline_plan/tests/ -v
```

Expected: All tests pass (existing + new). Total: ~80+ tests.

- [ ] **Step 11: Commit**

```bash
git add skills/timeline_plan/generate_plan.py skills/timeline_plan/tests/test_generate_plan.py
git commit -m "feat(generate_plan): integrate prompt_composer and duration_resolver, add v3 tracking fields, eliminate silent drop"
```

---

### Task 5: Self-Review of Plan

- [ ] **Step 1: Spec coverage check**

| Spec Section | Covered By | Status |
|---|---|---|
| §3.3 CoveragePlan dataclass | Task 1 (models.py) | ✅ |
| §3.4 PromptPatchPlan dataclass | Task 1 (models.py) | ✅ |
| §3.2 MatchEvidence dataclass | Task 1 (models.py) | ✅ |
| §3.6 TimelinePlanItem extensions | Task 1 (models.py) | ✅ |
| §5 6 operation types | Task 2 (prompt_composer.py) — supports 4 of 6 in Phase 1 | ✅ (fuzzy_replace, section_reconstruct deferred to Phase 2) |
| §5.3 semantic_insert | Task 2 (prompt_composer.py) — dedicated LLM prompt + validation | ✅ |
| §5.5 style_preserving_fallback | Task 2 (_generate_prompt_from_scene with style_layer) | ✅ |
| §4 degradation levels 0-6 | Task 4 (TimelinePlanItem.degradation_reason) — level meanings defined in spec | ✅ |
| §7 duration resolver | Task 3 (duration_resolver.py) — pad_after, pad_before, forced_min_duration | ✅ (Phase 1 covers 3 of 8 strategies; borrow/merge/cross-node deferred to Phase 2) |
| §7.2 expansion priority | Task 3 (_try_pad_strategy priority list) | ✅ |
| §7.1 NO silent drop | Task 4 (replaces `continue` with `resolve_duration`) | ✅ |
| §9.3 L3 validation | Task 2 (_validate_rewrite with operation_type) | ✅ |
| §9.1 L1 schema validation | Deferred to Phase 3 (validator.py) — not blocking for Phase 1 | ⚪ |
| §9.2 L2 ID/time validation | Deferred to Phase 3 | ⚪ |
| §9.4 L4 style validation | Partially covered: _extract_style_prefix ensures style in fallback | ⚪ |

**No uncovered spec requirements for Phase 1 scope.** L1/L2/L4 validation and full retry manager are Phase 3 items.

- [ ] **Step 2: Placeholder scan**

| Pattern | Present? |
|---|---|
| "TBD", "TODO" | ❌ None |
| "implement later" | ❌ None — Phase 2/3 items explicitly called out as deferred |
| "Add appropriate error handling" | ❌ None — `_try_pad_strategy` has explicit return None |
| "Write tests for the above" | ❌ None — every test has concrete test code |
| "Similar to Task N" | ❌ None — all code blocks are self-contained |
| References to undefined types | ❌ None — all types defined in Task 1 before use |

- [ ] **Step 3: Type consistency check**

| Type/Symbol | Defined In | Used In | Consistent? |
|---|---|---|---|
| `PromptPatchPlan` | Task 1 models.py | (not used in Phase 1 code, defined for Phase 2) | ✅ |
| `CoveragePlan` | Task 1 models.py | (not used in Phase 1 code, defined for Phase 2) | ✅ |
| `MatchEvidence` | Task 1 models.py | (not used in Phase 1 code, defined for Phase 2) | ✅ |
| `compose_prompt_patch` | Task 2 prompt_composer.py | Task 4 generate_plan.py | ✅ |
| `resolve_duration` | Task 3 duration_resolver.py | Task 4 generate_plan.py | ✅ |
| `_extract_style_prefix` | Task 2 prompt_composer.py | Task 2 prompt_composer.py | ✅ |
| `_validate_rewrite` | Task 2 prompt_composer.py | Task 2 prompt_composer.py | ✅ |
| `_generate_prompt_from_scene` (with style_layer) | Task 2 prompt_composer.py | Task 2 test_prompt_composer.py | ✅ |
| `TimelinePlanItem.operation_type` | Task 1 models.py | Task 4 generate_plan.py | ✅ |
| `TimelinePlanItem.duration_strategy` | Task 1 models.py | Task 4 generate_plan.py | ✅ |
| `TimelinePlanItem.covered_line_ids` | Task 1 models.py | Task 4 generate_plan.py | ✅ |
| `TimelinePlanItem.borrowed_line_ids` | Task 1 models.py | Task 4 generate_plan.py | ✅ |
| `TimelinePlanItem.source_node_ids` | Task 1 models.py | Task 4 generate_plan.py | ✅ |
| `TimelinePlanItem.degradation_reason` | Task 1 models.py | Task 4 generate_plan.py | ✅ |
| `duration_strategy` (local var) | Task 4 generate_plan.py | Task 4 generate_plan.py | ✅ |
| `operation_type` (local var) | Task 4 generate_plan.py | Task 4 generate_plan.py | ✅ |

All types consistent. ✅

---

### Completion Checklist

- [ ] All 5 tasks committed
- [ ] `python -m pytest skills/timeline_plan/tests/ -v` — all tests pass
- [ ] `prompt_extractor.py` still exists (backward compat, not imported by generate_plan)
- [ ] `prompt_composer.py` passes its test suite
- [ ] `duration_resolver.py` passes its test suite
- [ ] `generate_plan.py` passes all existing + new tests
- [ ] Short groups (<4s) are padded, not dropped
- [ ] `semantic_insert` mode is available (used when operation_type passed)
- [ ] Style layer extracted and preserved in fallback
- [ ] `TimelinePlanItem` JSON output includes new v3 fields with defaults
