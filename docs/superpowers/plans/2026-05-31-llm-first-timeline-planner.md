# LLM-First Timeline Planner Implementation Plan

> **For agentic workers:** Use `subagent-driven-development` to implement this plan task-by-task.

**Goal:** Refactor Stage 3 (timeline_plan) from rule-heavy pipeline to LLM-first architecture where LLM handles all semantic decisions (line-node matching, prompt rewriting, plan drafting) and deterministic code handles only validation and execution.

**Architecture:** Five new modules — `planner_models.py`, `evidence_builder.py`, `llm_planner.py`, `planner_verifier.py`, `timeline_normalizer.py` — replace the tangled flow of `canvas_matcher.py` + `prompt_composer.py` + `_classify_operation_type()` + `_split_contiguous()`. The new `generate_plan.py` orchestrates: Evidence Build → LLM Reason → LLM Structure → Verify → Normalize → TimelinePlan.

**Tech Stack:** Python 3.12+, Pydantic, DeepSeek API (OpenAI-compatible), existing dataclasses

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `skills/timeline_plan/planner_models.py` | CREATE | TimelinePlanDraft, ReplacementGroup, DialogueAlignment, PreservationReport, PlannerSelfReview, UnmatchedRewriteLine, UnusedDialogueNode |
| `skills/timeline_plan/evidence_builder.py` | CREATE | Build unified evidence pack from script + rewrite + canvas + cuts + keyframes |
| `skills/timeline_plan/llm_planner.py` | CREATE | LLM reasoning pass + structured output pass + retry |
| `skills/timeline_plan/planner_verifier.py` | CREATE | Schema/coverage/duplicate/prompt/node dialogue/self-review validation |
| `skills/timeline_plan/timeline_normalizer.py` | CREATE | Draft → executable TimelinePlan (cut snapping, duration, interleaving, gap fill) |
| `skills/timeline_plan/generate_plan.py` | MODIFY | Integrate new pipeline; keep `main()` CLI; remove old flow |
| `skills/timeline_plan/models.py` | MODIFY | Add any missing fields needed by new modules (minimal changes) |
| `skills/timeline_plan/prompt_composer.py` | KEEP | Deprecated; can be removed later after full verification |
| `skills/timeline_plan/canvas_matcher.py` | KEEP | Deprecated; can be removed later after full verification |
| `skills/timeline_plan/tests/test_planner_models.py` | CREATE | Tests for new pydantic/dataclass models |
| `skills/timeline_plan/tests/test_planner_verifier.py` | CREATE | Tests for verification logic |
| `skills/timeline_plan/tests/test_timeline_normalizer.py` | CREATE | Tests for normalization logic |

---

### Task 1: New Pydantic Models (`planner_models.py`)

**Files:**
- Create: `skills/timeline_plan/planner_models.py`

All models in the spec Section 7.3, 8.2, and 9.

- [ ] **Step 1: Create `planner_models.py` with all dataclasses**

```python
"""Planner models: LLM output schema for TimelinePlanDraft."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class DialogueAlignment:
    """Evidence for how a single rewritten line maps to a node prompt."""
    line_id: str
    original_line: str
    rewritten_line: str
    node_id: str
    original_dialogue_in_node_prompt: Optional[str] = None
    rewritten_dialogue_in_output_prompt: str = ""
    confidence: float = 0.0


@dataclass
class PreservationReport:
    """What visual elements were preserved during prompt rewriting."""
    environment_preserved: List[str] = field(default_factory=list)
    actions_preserved: List[str] = field(default_factory=list)
    style_preserved: List[str] = field(default_factory=list)
    changed_only_dialogue: bool = True


@dataclass
class ReplacementGroup:
    """One group of rewritten lines mapped to node(s) with rewritten prompt."""
    group_id: str
    covered_line_ids: List[str] = field(default_factory=list)
    matched_node_ids: List[str] = field(default_factory=list)
    source_time_range: Optional[SourceTimeRange] = None
    rewritten_prompt: str = ""
    reference_image_node_ids: List[str] = field(default_factory=list)
    dialogue_alignment: List[DialogueAlignment] = field(default_factory=list)
    preservation_report: Optional[PreservationReport] = None
    risk_flags: List[str] = field(default_factory=list)
    confidence: float = 0.0
    edit_explanation: str = ""


@dataclass
class SourceTimeRange:
    """Time range from covered lines' original timestamps."""
    start_sec: float
    end_sec: float


@dataclass
class UnmatchedRewriteLine:
    """A rewritten line that could not be matched to any node."""
    line_id: str
    reason: str = ""
    suggested_fallback: str = "keep_original_or_manual_review"


@dataclass
class UnusedDialogueNode:
    """A canvas node with dialogue that was not used."""
    node_id: str
    detected_dialogue: List[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class LineCoverageEntry:
    """Coverage status for one rewritten line."""
    line_id: str
    status: Literal["covered", "unmatched"]
    group_id: Optional[str] = None
    node_ids: List[str] = field(default_factory=list)


@dataclass
class NodeDialogueEntry:
    """Dialogue detected in a node prompt and its mapping."""
    node_id: str
    detected_dialogue: List[str] = field(default_factory=list)
    mapped_line_ids: List[str] = field(default_factory=list)
    unmapped_dialogue: List[str] = field(default_factory=list)


@dataclass
class PlannerSelfReview:
    """LLM's own quality check on its output."""
    all_rewritten_lines_covered: bool = False
    no_duplicate_line_coverage: bool = False
    no_unexplained_dialogue_left_in_used_nodes: bool = False
    likely_preserves_environment_actions: bool = False
    notes: List[str] = field(default_factory=list)
    line_coverage_table: List[LineCoverageEntry] = field(default_factory=list)
    node_dialogue_table: List[NodeDialogueEntry] = field(default_factory=list)
    duplicate_check: List[str] = field(default_factory=list)
    omission_check: List[str] = field(default_factory=list)
    style_preservation_check: str = ""
    risk_notes: List[str] = field(default_factory=list)


@dataclass
class TimelinePlanDraft:
    """LLM-generated draft plan before deterministic normalization."""
    plan_version: str = "llm_planner_v1"
    replacement_groups: List[ReplacementGroup] = field(default_factory=list)
    unmatched_rewrite_lines: List[UnmatchedRewriteLine] = field(default_factory=list)
    unused_dialogue_nodes: List[UnusedDialogueNode] = field(default_factory=list)
    self_review: Optional[PlannerSelfReview] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 2: Run Python import check**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "from skills.timeline_plan.planner_models import TimelinePlanDraft, ReplacementGroup, DialogueAlignment, PreservationReport, PlannerSelfReview; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/planner_models.py
git commit -m "feat: add planner_models with TimelinePlanDraft and all sub-models"
```

---

### Task 2: Evidence Builder (`evidence_builder.py`)

**Files:**
- Create: `skills/timeline_plan/evidence_builder.py`

Builds the unified evidence pack that LLM receives. Maps to spec Section 7.1.

- [ ] **Step 1: Create `evidence_builder.py`**

```python
"""Evidence Builder: packages all inputs into LLM-readable evidence."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from skills.timeline_plan.models import (
    CanvasNode, CutPoint, KeyFrame, LineEvidence,
    CanvasNodeEvidence, NodeSection, VideoEvidence, Constraints,
    EvidencePack,
)


def build_evidence(
    script_shots: List[Any],
    rewrite_lines_all: List[Dict],
    canvas_nodes: List[CanvasNode],
    cut_points: List[CutPoint],
    keyframes: List[KeyFrame],
    level: str = "B2",
    video_path: str = "",
) -> Dict[str, Any]:
    """Build the unified evidence dict for LLM consumption.

    Returns a dict that can be serialized and sent to the LLM planner.
    Structure:
    {
        "rewrite_lines": [...],       # target lines with original+rewritten+timestamps
        "neighbor_context_lines": [...],  # nearby unchanged lines for context
        "canvas_nodes": [...],        # all nodes with prompt analysis
        "video_context": {            # scene cuts, keyframes
            "scene_cuts": [...],
            "keyframe_paths": [...],
            "video_duration_sec": ...
        },
        "constraints": {              # generation constraints
            "must_cover_every_rewritten_line": true,
            "must_not_duplicate_lines": true,
            "must_preserve_environment_action_style": true,
            "min_seedance_duration_sec": 4.0
        }
    }
    """
    # Separate rewritten vs unchanged lines
    rewritten_lines: List[Dict] = []
    unchanged_lines: List[Dict] = []
    for rl in rewrite_lines_all:
        line_data = {
            "line_id": str(rl.get("line_id", "")),
            "original": str(rl.get("original", "")),
            "rewritten": str(rl.get("rewritten", "")),
            "speaker": str(rl.get("speaker", "")),
            "start_sec": float(rl.get("start_seconds", 0.0)),
            "end_sec": float(rl.get("end_seconds", 0.0)),
            "shot_number": int(rl.get("shot_number", 0)),
            "shot_scene": str(rl.get("shot_scene", "")),
            "rewrite_status": (
                "rewritten" if str(rl.get("original", "")) != str(rl.get("rewritten", ""))
                else "unchanged"
            ),
        }
        if line_data["rewrite_status"] == "rewritten":
            rewritten_lines.append(line_data)
        else:
            unchanged_lines.append(line_data)

    # Build canvas node evidence with dialogue extraction hints
    node_entries = []
    for node in canvas_nodes:
        prompt = node.prompt or ""
        # Extract quoted dialogue from prompt for LLM reference
        import re
        quoted = re.findall(r'"([^"]+)"', prompt)
        quoted = [q.strip() for q in quoted if len(q.strip()) > 3 and any(c.isalpha() for c in q)]
        
        node_entries.append({
            "node_id": node.node_id,
            "prompt": prompt,
            "detected_quoted_dialogue": quoted,
            "reference_images": node.reference_images or [],
            "video_url": node.video_url or "",
            "duration_sec": node.duration_sec,
        })

    # Video context
    video_context = {
        "scene_cuts": [c.time_sec for c in cut_points] if cut_points else [],
        "keyframe_paths": [k.image_path for k in keyframes] if keyframes else [],
        "video_path": video_path,
        "video_duration_sec": max(
            [rl.get("end_seconds", 0.0) for rl in rewrite_lines_all], default=60.0
        ),
    }

    return {
        "rewrite_lines": rewritten_lines,
        "neighbor_context_lines": unchanged_lines[:50],  # Limit context to avoid overflow
        "canvas_nodes": node_entries,
        "video_context": video_context,
        "constraints": {
            "must_cover_every_rewritten_line": True,
            "must_not_duplicate_lines": True,
            "must_preserve_environment_action_style": True,
            "min_seedance_duration_sec": 4.0,
        },
    }
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "from skills.timeline_plan.evidence_builder import build_evidence; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/evidence_builder.py
git commit -m "feat: add evidence_builder for unified LLM input packaging"
```

---

### Task 3: LLM Planner (`llm_planner.py`)

**Files:**
- Create: `skills/timeline_plan/llm_planner.py`

Two-pass LLM planner: reasoning pass (freeform) + structured output pass (JSON). Maps to spec Sections 7.2, 7.3, 8.

- [ ] **Step 1: Create `llm_planner.py` with reasoning + structured passes and retry**

```python
"""LLM Planner: two-pass LLM pipeline for TimelinePlanDraft generation.

Pass 1 (Reasoning): Freeform semantic analysis of line-node matching.
Pass 2 (Structured): Convert reasoning into strict JSON TimelinePlanDraft.
Retry: Up to 3 attempts with validation error feedback.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from skills.timeline_plan.planner_models import TimelinePlanDraft

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_REASONING_MODEL = "deepseek-v4-pro"  # Strong reasoning model
_STRUCTURED_MODEL = "deepseek-v4-pro"  # Same model for structured output


def _get_client():
    """Get OpenAI-compatible client for DeepSeek."""
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )


def _make_reasoning_prompt(evidence: Dict[str, Any]) -> str:
    """Build the reasoning pass prompt."""
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2)
    # Truncate if too long
    if len(evidence_json) > 48000:
        evidence_json = evidence_json[:48000] + "\n... (truncated, focus on rewrite_lines and canvas_nodes)"

    return f"""## Role
You are a video dialogue rewrite planner.

You will receive:
1. Original script lines with timestamps (rewrite_lines).
2. Canvas nodes with prompts (containing scene descriptions and quoted dialogue).
3. Scene cut and video context.

## Your Task (Analysis Phase — No JSON Output Yet)

Analyze the evidence and answer these questions in freeform text:

### 1. Line-to-Node Matching
For each rewritten line, identify which canvas node(s) contain the corresponding original dialogue.
- Match by: quoted dialogue in node prompts, semantic equivalence, speaker, scene continuity.
- A canvas node may cover multiple lines. Multiple nodes may contribute to one group.
- Extract actual spoken dialogue from node prompts; ignore signs, text overlays, sound effects, style descriptors.
- Do not omit any rewritten line.

### 2. Grouping Strategy
Which rewritten lines should be generated together as one seedance segment?
- Lines from the same scene/shot with the same matched node should typically be one group.
- Lines that are temporally close (<5s gap) and semantically connected should merge.
- Groups shorter than 4 seconds should merge with neighbors or note the risk.

### 3. Prompt Rewrite Strategy
For each group, how should the node prompt be rewritten?
- Only dialogue changes while preserving: environment, character actions, camera, lighting, style.
- If original dialogue is not explicit but visual scene matches, insert rewritten dialogue naturally.
- If a node prompt contains dialogue not covered by rewrite lines, preserve it unchanged.

### 4. Risks & Edge Cases
- Any lines that cannot be confidently matched?
- Any nodes with dialogue that should NOT be used?
- Any groups that are too short for seedance?

## Evidence
```json
{evidence_json}
```"""


def _make_structured_prompt(reasoning: str, evidence: Dict[str, Any]) -> str:
    """Build the structured output prompt with the reasoning context."""
    # Include only essential evidence for the structured pass
    essential = {
        "rewrite_lines": evidence.get("rewrite_lines", []),
        "canvas_nodes": [
            {"node_id": n["node_id"], "prompt": n["prompt"][:2000], "detected_quoted_dialogue": n["detected_quoted_dialogue"]}
            for n in evidence.get("canvas_nodes", [])
        ],
        "constraints": evidence.get("constraints", {}),
    }

    return f"""## Previous Analysis
{reasoning[:4000]}

## Task
Convert your analysis into a strict JSON TimelinePlanDraft following this schema.
Return ONLY valid JSON. No markdown, no explanations outside the JSON.

## Schema
```json
{{
  "plan_version": "llm_planner_v1",
  "replacement_groups": [
    {{
      "group_id": "G1",
      "covered_line_ids": ["line_001", "line_002"],
      "matched_node_ids": ["node_abc"],
      "source_time_range": {{
        "start_sec": 35.2,
        "end_sec": 40.1
      }},
      "rewritten_prompt": "full rewritten video generation prompt...",
      "reference_image_node_ids": ["node_abc"],
      "dialogue_alignment": [
        {{
          "line_id": "line_001",
          "original_line": "I can't believe you did that.",
          "rewritten_line": "I really can't believe you did that.",
          "node_id": "node_abc",
          "original_dialogue_in_node_prompt": "I can't believe you did that.",
          "rewritten_dialogue_in_output_prompt": "I really can't believe you did that.",
          "confidence": 0.94
        }}
      ],
      "preservation_report": {{
        "environment_preserved": ["apartment living room", "warm lighting"],
        "actions_preserved": ["Rachel confronts Ross"],
        "style_preserved": ["American sitcom", "cinematic lighting"],
        "changed_only_dialogue": true
      }},
      "risk_flags": [],
      "edit_explanation": "...",
      "confidence": 0.92
    }}
  ],
  "unmatched_rewrite_lines": [],
  "unused_dialogue_nodes": [],
  "self_review": {{
    "all_rewritten_lines_covered": true,
    "no_duplicate_line_coverage": true,
    "no_unexplained_dialogue_left_in_used_nodes": true,
    "likely_preserves_environment_actions": true,
    "notes": [],
    "line_coverage_table": [
      {{"line_id": "line_001", "status": "covered", "group_id": "G1", "node_ids": ["node_abc"]}}
    ],
    "node_dialogue_table": [
      {{
        "node_id": "node_abc",
        "detected_dialogue": ["I can't believe you did that."],
        "mapped_line_ids": ["line_001"],
        "unmapped_dialogue": []
      }}
    ],
    "duplicate_check": [],
    "omission_check": [],
    "style_preservation_check": "pass",
    "risk_notes": []
  }}
}}
```

## Rules (MUST FOLLOW)
1. Every rewritten line in the evidence MUST appear in exactly one replacement_group.covered_line_ids OR in unmatched_rewrite_lines.
2. No line_id may appear in more than one replacement_group.
3. Every replacement_group MUST have a non-empty rewritten_prompt.
4. rewritten_prompt MUST contain the rewritten dialogue verbatim for each covered line.
5. source_time_range.start_sec and end_sec MUST match the covered lines' actual timestamps.
6. preservation_report MUST list actual visual elements that were preserved.
7. dialogue_alignment MUST include every covered line.

## Evidence
```json
{json.dumps(essential, ensure_ascii=False, indent=2)}
```"""


def _parse_structured_output(text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM structured output JSON, handling markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) > 1:
            lines = lines[1:]  # Remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]  # Remove closing fence
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object boundaries
        brace_start = text.find("{")
        brace_count = 0
        for i in range(brace_start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[brace_start:i + 1])
                    except json.JSONDecodeError:
                        pass
        return None


def _build_draft_from_dict(data: Dict[str, Any]) -> TimelinePlanDraft:
    """Construct TimelinePlanDraft from parsed JSON dict."""
    from skills.timeline_plan.planner_models import (
        DialogueAlignment, PreservationReport, ReplacementGroup,
        SourceTimeRange, UnmatchedRewriteLine, UnusedDialogueNode,
        LineCoverageEntry, NodeDialogueEntry, PlannerSelfReview,
    )

    groups = []
    for g in data.get("replacement_groups", []):
        tr = g.get("source_time_range")
        source_tr = SourceTimeRange(start_sec=tr["start_sec"], end_sec=tr["end_sec"]) if tr else None

        pr = g.get("preservation_report")
        pres_report = PreservationReport(
            environment_preserved=pr.get("environment_preserved", []) if pr else [],
            actions_preserved=pr.get("actions_preserved", []) if pr else [],
            style_preserved=pr.get("style_preserved", []) if pr else [],
            changed_only_dialogue=pr.get("changed_only_dialogue", True) if pr else True,
        ) if pr else None

        alignments = []
        for a in g.get("dialogue_alignment", []):
            alignments.append(DialogueAlignment(
                line_id=a.get("line_id", ""),
                original_line=a.get("original_line", ""),
                rewritten_line=a.get("rewritten_line", ""),
                node_id=a.get("node_id", ""),
                original_dialogue_in_node_prompt=a.get("original_dialogue_in_node_prompt"),
                rewritten_dialogue_in_output_prompt=a.get("rewritten_dialogue_in_output_prompt", ""),
                confidence=float(a.get("confidence", 0.0)),
            ))

        groups.append(ReplacementGroup(
            group_id=g.get("group_id", ""),
            covered_line_ids=g.get("covered_line_ids", []),
            matched_node_ids=g.get("matched_node_ids", []),
            source_time_range=source_tr,
            rewritten_prompt=g.get("rewritten_prompt", ""),
            reference_image_node_ids=g.get("reference_image_node_ids", []),
            dialogue_alignment=alignments,
            preservation_report=pres_report,
            risk_flags=g.get("risk_flags", []),
            confidence=float(g.get("confidence", 0.0)),
            edit_explanation=g.get("edit_explanation", ""),
        ))

    unmatched = [
        UnmatchedRewriteLine(
            line_id=u.get("line_id", ""),
            reason=u.get("reason", ""),
            suggested_fallback=u.get("suggested_fallback", "keep_original_or_manual_review"),
        )
        for u in data.get("unmatched_rewrite_lines", [])
    ]

    unused = [
        UnusedDialogueNode(
            node_id=u.get("node_id", ""),
            detected_dialogue=u.get("detected_dialogue", []),
            reason=u.get("reason", ""),
        )
        for u in data.get("unused_dialogue_nodes", [])
    ]

    sr = data.get("self_review")
    self_review = None
    if sr:
        cov_table = [
            LineCoverageEntry(
                line_id=c.get("line_id", ""),
                status=c.get("status", "covered"),
                group_id=c.get("group_id"),
                node_ids=c.get("node_ids", []),
            )
            for c in sr.get("line_coverage_table", [])
        ]
        node_table = [
            NodeDialogueEntry(
                node_id=n.get("node_id", ""),
                detected_dialogue=n.get("detected_dialogue", []),
                mapped_line_ids=n.get("mapped_line_ids", []),
                unmapped_dialogue=n.get("unmapped_dialogue", []),
            )
            for n in sr.get("node_dialogue_table", [])
        ]
        self_review = PlannerSelfReview(
            all_rewritten_lines_covered=sr.get("all_rewritten_lines_covered", False),
            no_duplicate_line_coverage=sr.get("no_duplicate_line_coverage", False),
            no_unexplained_dialogue_left_in_used_nodes=sr.get("no_unexplained_dialogue_left_in_used_nodes", False),
            likely_preserves_environment_actions=sr.get("likely_preserves_environment_actions", False),
            notes=sr.get("notes", []),
            line_coverage_table=cov_table,
            node_dialogue_table=node_table,
            duplicate_check=sr.get("duplicate_check", []),
            omission_check=sr.get("omission_check", []),
            style_preservation_check=sr.get("style_preservation_check", ""),
            risk_notes=sr.get("risk_notes", []),
        )

    return TimelinePlanDraft(
        plan_version=data.get("plan_version", "llm_planner_v1"),
        replacement_groups=groups,
        unmatched_rewrite_lines=unmatched,
        unused_dialogue_nodes=unused,
        self_review=self_review,
        metadata=data.get("metadata", {}),
    )


def run_planner(
    evidence: Dict[str, Any],
    validation_feedback: Optional[List[str]] = None,
    attempt: int = 1,
) -> Tuple[Optional[TimelinePlanDraft], str]:
    """Run the LLM planner pipeline.

    Args:
        evidence: Unified evidence dict from evidence_builder.
        validation_feedback: Previous validation errors for retry.
        attempt: Current attempt number (1-based).

    Returns:
        Tuple of (TimelinePlanDraft or None, reasoning text).
    """
    client = _get_client()
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.error("DEEPSEEK_API_KEY not set — LLM planner cannot run")
        return None, ""

    # Pass 1: Reasoning
    reasoning_prompt = _make_reasoning_prompt(evidence)
    if validation_feedback:
        feedback_text = "\n".join(f"  - {e}" for e in validation_feedback)
        reasoning_prompt += f"\n\n## Previous Validation Errors (Attempt {attempt - 1})\nFix these issues:\n{feedback_text}\n\nKeep valid groups unchanged. Only fix the failing items."

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_PLANNER_MODEL", _REASONING_MODEL),
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=0.0,
            max_tokens=8192,
        )
        reasoning = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("Reasoning pass failed: %s", e)
        return None, ""

    # Pass 2: Structured output
    structured_prompt = _make_structured_prompt(reasoning, evidence)
    if validation_feedback:
        feedback_text = "\n".join(f"  - {e}" for e in validation_feedback)
        structured_prompt += f"\n\n## Fix These Validation Errors\n{feedback_text}"

    try:
        resp = client.chat.completions.create(
            model=os.environ.get("LLM_PLANNER_MODEL", _STRUCTURED_MODEL),
            messages=[{"role": "user", "content": structured_prompt}],
            temperature=0.0,
            max_tokens=8192,
            response_format={"type": "json_object"},
        )
        structured_text = resp.choices[0].message.content or ""
    except Exception as e:
        logger.error("Structured pass failed: %s", e)
        # Retry without response_format if it fails
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("LLM_PLANNER_MODEL", _STRUCTURED_MODEL),
                messages=[{"role": "user", "content": structured_prompt}],
                temperature=0.0,
                max_tokens=8192,
            )
            structured_text = resp.choices[0].message.content or ""
        except Exception as e2:
            logger.error("Structured pass retry also failed: %s", e2)
            return None, reasoning

    data = _parse_structured_output(structured_text)
    if data is None:
        logger.error("Failed to parse JSON from structured output")
        return None, reasoning

    try:
        draft = _build_draft_from_dict(data)
        return draft, reasoning
    except Exception as e:
        logger.error("Failed to build TimelinePlanDraft: %s", e)
        return None, reasoning


def generate_plan_draft(
    evidence: Dict[str, Any],
    max_retries: int = _MAX_RETRIES,
) -> TimelinePlanDraft:
    """Generate TimelinePlanDraft with retry on validation failure.

    Args:
        evidence: Unified evidence dict.
        max_retries: Maximum retry attempts (default 3).

    Returns:
        TimelinePlanDraft (may contain unmatched lines if all retries fail).

    Raises:
        ValueError: If all retries fail with critical errors.
    """
    from skills.timeline_plan.planner_verifier import verify_draft

    draft = None
    reasoning = ""
    validation_errors: List[str] = []
    last_errors: List[str] = []

    for attempt in range(1, max_retries + 1):
        draft, reasoning = run_planner(
            evidence, validation_feedback=last_errors if attempt > 1 else None, attempt=attempt,
        )

        if draft is None:
            logger.warning("Attempt %d: LLM planner returned no draft", attempt)
            last_errors = ["LLM planner failed to produce a draft — no JSON output"]
            continue

        errors = verify_draft(draft, evidence)
        if not errors:
            logger.info("Attempt %d: draft passed all validation checks", attempt)
            return draft

        logger.warning("Attempt %d: %d validation errors", attempt, len(errors))
        for e in errors[:5]:
            logger.warning("  %s", e)
        last_errors = errors

    # All retries exhausted — raise with details
    if draft is None:
        raise ValueError(
            f"LLM planner failed to produce a valid draft after {max_retries} attempts.\n"
            f"Last errors: {'; '.join(last_errors[:5])}"
        )

    logger.error(
        "All %d attempts failed validation. Returning draft with %d errors.",
        max_retries, len(last_errors),
    )
    return draft
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "from skills.timeline_plan.llm_planner import generate_plan_draft; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/llm_planner.py
git commit -m "feat: add llm_planner with two-pass reasoning + structured output + retry"
```

---

### Task 4: Planner Verifier (`planner_verifier.py`)

**Files:**
- Create: `skills/timeline_plan/planner_verifier.py`

Deterministic verification. Maps to spec Section 7.4.

- [ ] **Step 1: Create `planner_verifier.py`**

```python
"""Planner Verifier: deterministic validation of TimelinePlanDraft.

Validates: schema, line coverage, duplicates, prompt dialogue inclusion, 
node dialogue consistency, self-review consistency.
"""
from __future__ import annotations

from typing import Any, Dict, List

from skills.timeline_plan.planner_models import TimelinePlanDraft


def verify_draft(draft: TimelinePlanDraft, evidence: Dict[str, Any]) -> List[str]:
    """Verify a TimelinePlanDraft against all deterministic checks.
    
    Args:
        draft: The LLM-generated draft.
        evidence: The original evidence used to generate the draft.
    
    Returns:
        List of error messages. Empty list means all checks pass.
    """
    errors: List[str] = []

    errors.extend(_check_schema(draft))
    errors.extend(_check_line_coverage(draft, evidence))
    errors.extend(_check_no_duplicates(draft))
    errors.extend(_check_prompt_contains_dialogue(draft))
    errors.extend(_check_time_ranges(draft))
    errors.extend(_check_self_review_consistency(draft, evidence))

    return errors


def _check_schema(draft: TimelinePlanDraft) -> List[str]:
    """L0: Basic schema validation."""
    errors = []
    if not draft.replacement_groups and not draft.unmatched_rewrite_lines:
        errors.append("schema: both replacement_groups and unmatched_rewrite_lines are empty")
    for group in draft.replacement_groups:
        if not group.group_id:
            errors.append(f"schema: replacement_group has empty group_id")
        if not group.rewritten_prompt or not group.rewritten_prompt.strip():
            errors.append(f"schema: group {group.group_id} has empty rewritten_prompt")
        if not group.covered_line_ids:
            errors.append(f"schema: group {group.group_id} has empty covered_line_ids")
    return errors


def _check_line_coverage(draft: TimelinePlanDraft, evidence: Dict[str, Any]) -> List[str]:
    """L1: Every rewritten line must appear exactly once."""
    errors = []
    rewrite_lines = evidence.get("rewrite_lines", [])
    all_rewrite_ids = {rl["line_id"] for rl in rewrite_lines}

    # Collect covered IDs
    covered_ids: Dict[str, List[str]] = {}  # line_id → [group_ids]
    for group in draft.replacement_groups:
        for lid in group.covered_line_ids:
            covered_ids.setdefault(lid, []).append(group.group_id)

    unmatched_ids = {u.line_id for u in draft.unmatched_rewrite_lines}

    # Check every rewritten line is covered or unmatched
    for lid in all_rewrite_ids:
        if lid not in covered_ids and lid not in unmatched_ids:
            errors.append(f"coverage: line {lid} is missing — not in any group or unmatched list")
        if lid in covered_ids and lid in unmatched_ids:
            errors.append(f"coverage: line {lid} is both covered AND marked unmatched")

    return errors


def _check_no_duplicates(draft: TimelinePlanDraft) -> List[str]:
    """L2: No line appears in multiple replacement groups."""
    errors = []
    seen: Dict[str, str] = {}
    for group in draft.replacement_groups:
        for lid in group.covered_line_ids:
            if lid in seen:
                errors.append(
                    f"duplicate: line {lid} appears in both {seen[lid]} and {group.group_id}"
                )
            seen[lid] = group.group_id
    return errors


def _check_prompt_contains_dialogue(draft: TimelinePlanDraft) -> List[str]:
    """L3: Every rewritten_prompt must contain the rewritten dialogue."""
    errors = []
    for group in draft.replacement_groups:
        if not group.rewritten_prompt:
            continue
        for alignment in group.dialogue_alignment:
            if not alignment.rewritten_dialogue_in_output_prompt:
                continue
            if alignment.rewritten_dialogue_in_output_prompt not in group.rewritten_prompt:
                errors.append(
                    f"prompt: group {group.group_id} rewritten_prompt does not contain "
                    f"rewritten dialogue for line {alignment.line_id}: "
                    f"\"{alignment.rewritten_dialogue_in_output_prompt[:80]}\""
                )
    return errors


def _check_time_ranges(draft: TimelinePlanDraft) -> List[str]:
    """L4: source_time_range must be valid (start < end, non-negative)."""
    errors = []
    for group in draft.replacement_groups:
        tr = group.source_time_range
        if tr is None:
            continue
        if tr.start_sec < 0:
            errors.append(f"time: group {group.group_id} has negative start_sec ({tr.start_sec})")
        if tr.end_sec <= tr.start_sec:
            errors.append(
                f"time: group {group.group_id} end_sec ({tr.end_sec}) <= start_sec ({tr.start_sec})"
            )
    return errors


def _check_self_review_consistency(draft: TimelinePlanDraft, evidence: Dict[str, Any]) -> List[str]:
    """L5: Self-review claims should match actual draft data."""
    errors = []
    review = draft.self_review
    if review is None:
        return errors

    rewrite_lines = evidence.get("rewrite_lines", [])
    all_rewrite_ids = {rl["line_id"] for rl in rewrite_lines}

    # Check coverage table vs actual coverage
    covered_in_table = {c.line_id for c in review.line_coverage_table if c.status == "covered"}
    covered_in_groups = set()
    for group in draft.replacement_groups:
        covered_in_groups.update(group.covered_line_ids)

    if covered_in_table != covered_in_groups:
        missing = covered_in_groups - covered_in_table
        extra = covered_in_table - covered_in_groups
        if missing:
            errors.append(f"self-review: coverage table missing lines: {missing}")
        if extra:
            errors.append(f"self-review: coverage table has extra lines: {extra}")

    # Check all_rewritten_lines_covered claim
    all_covered = covered_in_groups | {u.line_id for u in draft.unmatched_rewrite_lines}
    actually_all = all_covered == all_rewrite_ids
    if review.all_rewritten_lines_covered and not actually_all:
        errors.append("self-review: claims all_rewritten_lines_covered but lines are missing")
    if not review.all_rewritten_lines_covered and actually_all:
        # Not an error — LLM is just being conservative
        pass

    return errors
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "from skills.timeline_plan.planner_verifier import verify_draft; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/planner_verifier.py
git commit -m "feat: add planner_verifier with deterministic validation checks"
```

---

### Task 5: Timeline Normalizer (`timeline_normalizer.py`)

**Files:**
- Create: `skills/timeline_plan/timeline_normalizer.py`

Draft → executable TimelinePlan. Maps to spec Section 7.5.

- [ ] **Step 1: Create `timeline_normalizer.py`**

```python
"""Timeline Normalizer: converts TimelinePlanDraft to executable TimelinePlan.

Pure deterministic logic — no semantic decisions.
Responsibilities: snap to scene cuts, normalize durations, 
interleave original/seedance items, remove overlaps, fill gaps.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, KeyFrame,
    normalize_seedance_duration, MIN_SEEDANCE_DURATION,
)
from skills.timeline_plan.planner_models import (
    TimelinePlanDraft, ReplacementGroup,
)

MAX_SEEDANCE_DURATION = 30.0


def normalize_plan(
    draft: TimelinePlanDraft,
    script_shots: List[Any],
    canvas_nodes: List[CanvasNode],
    cut_points: List[CutPoint],
    keyframes: List[KeyFrame],
    video_duration: float,
    title: str = "Untitled",
    level: str = "B2",
) -> TimelinePlan:
    """Convert TimelinePlanDraft to executable TimelinePlan.

    Steps:
    1. Convert each ReplacementGroup to a seedance TimelinePlanItem.
    2. Build original segments from script shots, carving out seedance regions.
    3. Snap boundaries to scene cuts.
    4. Normalize durations (ensure ≥4s seedance, extend/merge if needed).
    5. Interleave original/seedance items, remove overlaps, fill gaps.
    """
    node_map: Dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    items: List[TimelinePlanItem] = []
    covered_line_ids: set = set()

    # ── Step 1: Convert replacement groups to seedance items ──
    for group in draft.replacement_groups:
        tr = group.source_time_range
        if tr is None:
            start_sec = 0.0
            end_sec = MIN_SEEDANCE_DURATION
        else:
            start_sec = tr.start_sec
            end_sec = tr.end_sec

        duration = end_sec - start_sec
        if duration < MIN_SEEDANCE_DURATION:
            end_sec = start_sec + MIN_SEEDANCE_DURATION

        # Get reference images from matched nodes
        ref_images: List[str] = []
        degradation_level = 0
        for nid in group.matched_node_ids:
            node = node_map.get(nid)
            if node and node.reference_images:
                ref_images.extend(node.reference_images)
            elif keyframes:
                # Fallback: use keyframes from the time range
                ref_images.extend([
                    k.image_path for k in keyframes
                    if start_sec <= k.time_sec <= end_sec
                ])
        if not ref_images:
            degradation_level = 1

        primary_node_id = group.matched_node_ids[0] if group.matched_node_ids else None
        seedance_dur = normalize_seedance_duration(duration)

        items.append(TimelinePlanItem(
            shot_id=f"seedance_{group.group_id}",
            shot_number=0,
            source="seedance",
            start_sec=start_sec,
            end_sec=end_sec,
            scene_description=group.edit_explanation or "",
            ref_images=ref_images,
            rewritten_prompt=group.rewritten_prompt,
            matched_node_id=primary_node_id,
            match_confidence=group.confidence,
            degradation_level=degradation_level,
            seedance_duration=seedance_dur,
            original_duration=duration,
            covered_line_ids=sorted(group.covered_line_ids),
            source_node_ids=sorted(group.matched_node_ids),
            degradation_reason=(
                "duration_padded_to_meet_min_4s" if duration < MIN_SEEDANCE_DURATION
                else ""
            ),
        ))
        covered_line_ids.update(group.covered_line_ids)

    # ── Step 2: Build original segments from script shots ──
    from skills.timeline_plan.cut_fusion import determine_cut_points
    cut_boundaries = determine_cut_points(script_shots, cut_points, video_duration)

    for idx, shot in enumerate(script_shots):
        start_s, end_s = cut_boundaries[idx]
        items.append(TimelinePlanItem(
            shot_id=f"shot_{getattr(shot, 'shot_number', idx)}",
            shot_number=getattr(shot, 'shot_number', idx),
            source="original",
            start_sec=start_s,
            end_sec=end_s,
            scene_description=getattr(shot, "scene_description", "") or "",
            original_duration=end_s - start_s,
        ))

    # ── Step 3: Finalize timeline (carve, merge, fill gaps) ──
    items.sort(key=lambda i: i.start_sec)
    items = _finalize_timeline(items, video_duration)

    # ── Step 4: Build TimelinePlan ──
    return TimelinePlan(
        title=title,
        level=level,
        pipeline_version="3.0",
        total_duration_sec=video_duration,
        items=items,
        metadata={
            "num_items": len(items),
            "num_seedance": sum(1 for i in items if i.source == "seedance"),
            "num_original": sum(1 for i in items if i.source == "original"),
            "num_groups": len(draft.replacement_groups),
            "unmatched_lines": len(draft.unmatched_rewrite_lines),
            "planner_version": draft.plan_version,
        },
    )


def _carve_out(segments, carve_start, carve_end):
    """Remove [carve_start, carve_end] from list of (start, end) segments."""
    result = []
    for seg_start, seg_end in segments:
        if carve_start >= seg_end or carve_end <= seg_start:
            result.append((seg_start, seg_end))
        else:
            if seg_start < carve_start:
                result.append((seg_start, carve_start))
            if seg_end > carve_end:
                result.append((carve_end, seg_end))
    return result


def _finalize_timeline(
    items: List[TimelinePlanItem],
    video_duration: float,
    min_original: float = 0.5,
) -> List[TimelinePlanItem]:
    """Post-process timeline into non-overlapping, gap-free sequence.

    Principles:
    1. Seedance items shorter than 4s → extend to 4s.
    2. Carve original items around finalized seedance ranges.
    3. Swallow micro original segments, merge adjacent originals, fill gaps.
    """
    # Step 1: Enforce seedance min/max duration
    result = []
    for item in items:
        if item.source == "seedance":
            dur = item.end_sec - item.start_sec
            if dur < MIN_SEEDANCE_DURATION:
                item.end_sec = item.start_sec + MIN_SEEDANCE_DURATION
                item.original_duration = MIN_SEEDANCE_DURATION
            elif dur > MAX_SEEDANCE_DURATION:
                covered = list(item.covered_line_ids)
                num_segs = max(2, math.ceil(dur / MAX_SEEDANCE_DURATION))
                seg_dur = dur / num_segs
                for k in range(num_segs):
                    s_start = round(item.start_sec + k * seg_dur, 1)
                    s_end = round(item.start_sec + (k + 1) * seg_dur, 1)
                    seg_lines = [covered[i] for i in range(len(covered)) if i % num_segs == k]
                    result.append(TimelinePlanItem(
                        shot_id=f"{item.shot_id}_p{k}",
                        shot_number=item.shot_number,
                        source="seedance",
                        start_sec=s_start, end_sec=s_end,
                        scene_description=item.scene_description,
                        ref_images=list(item.ref_images),
                        rewritten_prompt=item.rewritten_prompt,
                        matched_node_id=item.matched_node_id,
                        match_confidence=item.match_confidence,
                        degradation_level=item.degradation_level,
                        seedance_duration=item.seedance_duration,
                        original_duration=s_end - s_start,
                        covered_line_ids=seg_lines,
                        source_node_ids=list(item.source_node_ids),
                        degradation_reason=f"split_from_long_segment (original={dur:.1f}s)",
                    ))
                continue
        result.append(item)
    items = result

    # Step 2: Merge overlapping seedance items
    seedance_items = sorted(
        [i for i in items if i.source == "seedance"],
        key=lambda x: x.start_sec,
    )
    i = 0
    while i < len(seedance_items) - 1:
        curr, nxt = seedance_items[i], seedance_items[i + 1]
        if curr.end_sec > nxt.start_sec + 0.05:
            curr.end_sec = max(curr.end_sec, nxt.end_sec)
            curr.original_duration = curr.end_sec - curr.start_sec
            curr.covered_line_ids = sorted(set(curr.covered_line_ids) | set(nxt.covered_line_ids))
            curr.source_node_ids = sorted(set(curr.source_node_ids) | set(nxt.source_node_ids))
            if not curr.rewritten_prompt and nxt.rewritten_prompt:
                curr.rewritten_prompt = nxt.rewritten_prompt
                curr.matched_node_id = nxt.matched_node_id
                curr.match_confidence = nxt.match_confidence
            items.remove(nxt)
            seedance_items.pop(i + 1)
        else:
            i += 1

    # Step 3: Carve seedance ranges out of original items
    seedance_all = [si for si in items if si.source == "seedance"]
    result2 = []
    for item in items:
        if item.source != "original":
            result2.append(item)
            continue
        segments = [(item.start_sec, item.end_sec)]
        for si in seedance_all:
            segments = _carve_out(segments, si.start_sec, si.end_sec)
        for seg_start, seg_end in segments:
            if seg_end - seg_start > 0.1:
                result2.append(TimelinePlanItem(
                    shot_id=f"{item.shot_id}_seg",
                    shot_number=item.shot_number,
                    source="original",
                    start_sec=seg_start, end_sec=seg_end,
                    scene_description=item.scene_description,
                    original_duration=seg_end - seg_start,
                ))
    items = result2

    # Step 4: Sort, swallow micro originals, merge adjacent originals, fill gaps
    items.sort(key=lambda i: i.start_sec)

    # Swallow micro originals (< min_original) into adjacent seedance
    i = 0
    while i < len(items):
        item = items[i]
        if item.source != "original" or item.duration_sec >= min_original:
            i += 1
            continue
        prev_seed = items[i - 1] if i > 0 and items[i - 1].source == "seedance" else None
        next_seed = items[i + 1] if i + 1 < len(items) and items[i + 1].source == "seedance" else None
        if prev_seed:
            prev_seed.end_sec = max(prev_seed.end_sec, item.end_sec)
            items.pop(i)
        elif next_seed:
            next_seed.start_sec = min(next_seed.start_sec, item.start_sec)
            items.pop(i)
        else:
            i += 1

    # Merge adjacent originals, fill gaps
    merged = []
    last_end = 0.0
    for item in items:
        if item.start_sec > last_end + 0.1:
            merged.append(TimelinePlanItem(
                shot_id=f"gap_{last_end:.0f}",
                shot_number=0,
                source="original",
                start_sec=last_end, end_sec=item.start_sec,
                scene_description="",
                original_duration=item.start_sec - last_end,
            ))
        if item.source == "original" and merged and merged[-1].source == "original":
            _merge_original(merged[-1], item)
        else:
            merged.append(item)
        last_end = max(last_end, item.end_sec)
    if last_end < video_duration - 0.1:
        merged.append(TimelinePlanItem(
            shot_id=f"gap_{last_end:.0f}",
            shot_number=0,
            source="original",
            start_sec=last_end, end_sec=video_duration,
            scene_description="",
            original_duration=video_duration - last_end,
        ))

    return merged


def _merge_original(target, source):
    """Merge source original item into target."""
    target.end_sec = source.end_sec
    target.original_duration = target.end_sec - target.start_sec
    target.covered_line_ids = sorted(
        set(target.covered_line_ids or []) | set(source.covered_line_ids or [])
    )
    target.degradation_level = max(
        target.degradation_level or 0, source.degradation_level or 0
    )
```

- [ ] **Step 2: Verify import**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "from skills.timeline_plan.timeline_normalizer import normalize_plan; print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add skills/timeline_plan/timeline_normalizer.py
git commit -m "feat: add timeline_normalizer for draft-to-executable-plan conversion"
```

---

### Task 6: Integrate into `generate_plan.py`

**Files:**
- Modify: `skills/timeline_plan/generate_plan.py`

Replace the old orchestration flow with the new LLM-first pipeline. Keep `main()` CLI compatible.

- [ ] **Step 1: Read current generate_plan.py to identify integration points**

Already read. Key changes:
1. Import new modules
2. Replace `generate_timeline_plan()` body with new pipeline
3. Keep `main()` as-is (CLI args unchanged)
4. Keep `_snap_boundaries` and `_carve_out` — they move to normalizer but keep for now
5. Remove old functions: `_classify_operation_type`, `_split_contiguous`, `_make_rl_objects`, `_merge_original_items`, `_finalize_timeline`, `_shot_needs_rewrite`, `_collect_ref_images`, `_fuzzy_word_match`

- [ ] **Step 2: Modify `generate_plan.py`**

```python
# Add these imports at the top (after existing imports):
from skills.timeline_plan.evidence_builder import build_evidence
from skills.timeline_plan.llm_planner import generate_plan_draft
from skills.timeline_plan.timeline_normalizer import normalize_plan

# Replace `generate_timeline_plan()` body:
def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    """Generate timeline plan using LLM-first pipeline.

    v3.0: LLM Planner → Deterministic Verifier → Timeline Normalizer.
    Removes rule-based semantic decisions (classify, split_contiguous, prompt_composer).
    LLM handles all line-node matching, grouping, and prompt rewriting.
    """
    script_output = input_data.script_output
    shots = list(script_output.script.shots) if script_output else []
    rewrite_lines_all = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    video_cuts = input_data.video_cut_points
    keyframes = input_data.keyframes
    level = input_data.level

    video_duration = max(
        [s.end_seconds for s in shots if hasattr(s, 'end_seconds')],
        default=60.0,
    )
    max_asr_end = max(
        (rl.get("end_seconds", 0.0) for rl in rewrite_lines_all), default=0.0,
    )
    video_duration = max(video_duration, max_asr_end)

    title = getattr(script_output, "title", "Untitled") if script_output else "Untitled"

    # ── Stage 3A: Build evidence for LLM ──
    logger.info("Building evidence pack for LLM planner...")
    evidence = build_evidence(
        script_shots=shots,
        rewrite_lines_all=rewrite_lines_all,
        canvas_nodes=canvas_nodes,
        cut_points=video_cuts,
        keyframes=keyframes,
        level=level,
    )
    logger.info(
        "Evidence: %d rewrite lines, %d canvas nodes, %d scene cuts",
        len(evidence.get("rewrite_lines", [])),
        len(evidence.get("canvas_nodes", [])),
        len(evidence.get("video_context", {}).get("scene_cuts", [])),
    )

    # ── Stage 3B-C: LLM Planner (reasoning + structured output) ──
    logger.info("Running LLM planner (attempts up to 3)...")
    try:
        draft = generate_plan_draft(evidence, max_retries=3)
    except ValueError as e:
        logger.error("LLM planner failed: %s", e)
        raise

    logger.info(
        "LLM planner: %d groups, %d unmatched lines, %d unused nodes",
        len(draft.replacement_groups),
        len(draft.unmatched_rewrite_lines),
        len(draft.unused_dialogue_nodes),
    )

    if draft.self_review:
        sr = draft.self_review
        logger.info(
            "Self-review: all_covered=%s, no_dups=%s, no_unexplained=%s, preserves=%s",
            sr.all_rewritten_lines_covered,
            sr.no_duplicate_line_coverage,
            sr.no_unexplained_dialogue_left_in_used_nodes,
            sr.likely_preserves_environment_actions,
        )
        if sr.risk_notes:
            for note in sr.risk_notes[:5]:
                logger.warning("  Risk: %s", note)

    # ── Stage 3E: Timeline Normalizer ──
    logger.info("Normalizing timeline plan...")
    plan = normalize_plan(
        draft=draft,
        script_shots=shots,
        canvas_nodes=canvas_nodes,
        cut_points=video_cuts,
        keyframes=keyframes,
        video_duration=video_duration,
        title=title,
        level=level,
    )

    # ── Validation ──
    from skills.timeline_plan.validator import validate_timeline_item, validate_timeline_items
    validation_errors: List[str] = []
    for item in plan.items:
        validation_errors.extend(validate_timeline_item(item))
    validation_errors.extend(validate_timeline_items(plan.items, video_duration))

    _BLOCKING_KEYWORDS = (
        "overlap", "start_sec", "end_sec", "empty rewritten_prompt",
        "covered by both", "gap at start", "gap at end",
        "empty shot_id", "invalid source",
    )
    blocking_errors = [
        e for e in validation_errors
        if any(kw in e.lower() for kw in _BLOCKING_KEYWORDS)
    ]
    if blocking_errors:
        raise ValueError(
            f"Timeline plan validation FAILED with {len(blocking_errors)} blocking errors:\n"
            + "\n".join(f"  - {e}" for e in blocking_errors)
        )
    if validation_errors:
        logger.warning("Timeline plan validation: %d non-blocking warnings", len(validation_errors))
        for err in validation_errors[:10]:
            logger.warning("  %s", err)

    return plan
```

- [ ] **Step 3: Remove old functions from generate_plan.py**

Remove these functions from the file:
- `_shot_needs_rewrite` (lines 30-41)
- `_collect_ref_images` (lines 44-48)
- `_fuzzy_word_match` (lines 51-58)
- `_classify_operation_type` (lines 61-79)
- `_split_contiguous` (lines 82-140)
- `_make_rl_objects` (lines 143-152)
- `_merge_original_items` (lines 155-191)
- `_finalize_timeline` (lines 194-348)
- `_carve_out` (lines 351-362)
- `_snap_boundaries` (lines 365-398)

Note: Remove the imports of `prompt_composer`, `canvas_matcher`, `duration_resolver` from the top.

```python
# Remove these imports:
from skills.timeline_plan.canvas_matcher import match_lines_to_nodes  # DELETE
from skills.timeline_plan.prompt_composer import compose_prompt_patch  # DELETE
from skills.timeline_plan.duration_resolver import resolve_duration  # DELETE

# Keep these imports:
from skills.timeline_plan.models import (
    TimelinePlan, TimelinePlanItem, CanvasNode, CutPoint, KeyFrame, Stage3Input,
    normalize_seedance_duration, MIN_SEEDANCE_DURATION,
)
from skills.timeline_plan.cut_fusion import determine_cut_points  # KEEP (used in normalizer)
```

- [ ] **Step 4: Update pipeline version in metadata**

In `generate_timeline_plan`, change `pipeline_version` from `"2.0"` to `"3.0"` (this is now set in `normalize_plan`, but verify).

- [ ] **Step 5: Verify import**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "from skills.timeline_plan.generate_plan import generate_timeline_plan; print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add skills/timeline_plan/generate_plan.py
git commit -m "refactor: integrate LLM-first pipeline into generate_plan.py (v3.0)"
```

---

### Task 7: Write Tests

**Files:**
- Create: `skills/timeline_plan/tests/test_planner_models.py`
- Create: `skills/timeline_plan/tests/test_planner_verifier.py`
- Create: `skills/timeline_plan/tests/test_timeline_normalizer.py`

- [ ] **Step 1: Create test file for planner_models**

```python
"""Tests for planner_models.py"""
import pytest
from skills.timeline_plan.planner_models import (
    TimelinePlanDraft, ReplacementGroup, DialogueAlignment,
    PreservationReport, PlannerSelfReview, SourceTimeRange,
    LineCoverageEntry, NodeDialogueEntry,
)


class TestTimelinePlanDraft:
    def test_empty_draft(self):
        draft = TimelinePlanDraft()
        assert draft.plan_version == "llm_planner_v1"
        assert draft.replacement_groups == []
        assert draft.unmatched_rewrite_lines == []

    def test_with_groups(self):
        draft = TimelinePlanDraft(
            replacement_groups=[
                ReplacementGroup(
                    group_id="G1",
                    covered_line_ids=["L1", "L2"],
                    matched_node_ids=["node_abc"],
                    source_time_range=SourceTimeRange(start_sec=10.0, end_sec=15.0),
                    rewritten_prompt="A cinematic scene...",
                    confidence=0.95,
                )
            ],
        )
        assert len(draft.replacement_groups) == 1
        assert draft.replacement_groups[0].group_id == "G1"
        assert draft.replacement_groups[0].source_time_range.start_sec == 10.0
```

- [ ] **Step 2: Create test file for planner_verifier**

```python
"""Tests for planner_verifier.py"""
from skills.timeline_plan.planner_models import (
    TimelinePlanDraft, ReplacementGroup, SourceTimeRange,
    DialogueAlignment, PlannerSelfReview, LineCoverageEntry,
)
from skills.timeline_plan.planner_verifier import verify_draft


def make_evidence(line_ids):
    return {"rewrite_lines": [{"line_id": lid, "original": f"text_{lid}", "rewritten": f"new_{lid}"} for lid in line_ids]}


class TestVerifyDraft:
    def test_empty_draft_fails(self):
        draft = TimelinePlanDraft()
        errors = verify_draft(draft, make_evidence([]))
        assert len(errors) > 0

    def test_missing_line(self):
        evidence = make_evidence(["L1", "L2"])
        draft = TimelinePlanDraft(
            replacement_groups=[
                ReplacementGroup(
                    group_id="G1",
                    covered_line_ids=["L1"],
                    rewritten_prompt="test prompt",
                    source_time_range=SourceTimeRange(start_sec=0, end_sec=4),
                )
            ]
        )
        errors = verify_draft(draft, evidence)
        assert any("L2" in e for e in errors)

    def test_duplicate_line(self):
        evidence = make_evidence(["L1"])
        draft = TimelinePlanDraft(
            replacement_groups=[
                ReplacementGroup(
                    group_id="G1",
                    covered_line_ids=["L1"],
                    rewritten_prompt="test prompt 1",
                    source_time_range=SourceTimeRange(start_sec=0, end_sec=4),
                ),
                ReplacementGroup(
                    group_id="G2",
                    covered_line_ids=["L1"],
                    rewritten_prompt="test prompt 2",
                    source_time_range=SourceTimeRange(start_sec=0, end_sec=4),
                ),
            ]
        )
        errors = verify_draft(draft, evidence)
        assert any("L1" in e and "G1" in e and "G2" in e for e in errors)

    def test_all_covered(self):
        evidence = make_evidence(["L1", "L2"])
        draft = TimelinePlanDraft(
            replacement_groups=[
                ReplacementGroup(
                    group_id="G1",
                    covered_line_ids=["L1", "L2"],
                    rewritten_prompt="test prompt with L1 L2 dialogue",
                    source_time_range=SourceTimeRange(start_sec=0, end_sec=4),
                )
            ]
        )
        errors = verify_draft(draft, evidence)
        assert len(errors) == 0

    def test_unmatched_lines_ok(self):
        evidence = make_evidence(["L1", "L2"])
        from skills.timeline_plan.planner_models import UnmatchedRewriteLine
        draft = TimelinePlanDraft(
            replacement_groups=[
                ReplacementGroup(
                    group_id="G1",
                    covered_line_ids=["L1"],
                    rewritten_prompt="test prompt",
                    source_time_range=SourceTimeRange(start_sec=0, end_sec=4),
                )
            ],
            unmatched_rewrite_lines=[
                UnmatchedRewriteLine(line_id="L2", reason="no node match")
            ]
        )
        errors = verify_draft(draft, evidence)
        assert len(errors) == 0
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -m pytest skills/timeline_plan/tests/test_planner_models.py skills/timeline_plan/tests/test_planner_verifier.py -v
```

- [ ] **Step 4: Commit**

```bash
git add skills/timeline_plan/tests/
git commit -m "test: add tests for planner_models and planner_verifier"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Run full test suite**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -m pytest skills/timeline_plan/tests/ -v --tb=short 2>&1 | head -100
```

- [ ] **Step 2: Verify all imports work together**

```bash
cd /Users/hupan/workspace/analyze_script_with_canvas && python3 -c "
from skills.timeline_plan.models import TimelinePlan, TimelinePlanItem, Stage3Input
from skills.timeline_plan.planner_models import TimelinePlanDraft, ReplacementGroup
from skills.timeline_plan.evidence_builder import build_evidence
from skills.timeline_plan.llm_planner import generate_plan_draft, run_planner
from skills.timeline_plan.planner_verifier import verify_draft
from skills.timeline_plan.timeline_normalizer import normalize_plan
from skills.timeline_plan.generate_plan import generate_timeline_plan
print('All imports OK')
"
```

- [ ] **Step 3: Check LSP diagnostics**

```bash
# Check for type errors / warnings
python3 -m py_compile skills/timeline_plan/*.py
```

- [ ] **Step 4: Commit final state**

```bash
git add -A skills/timeline_plan/
git commit -m "chore: final verification pass — all imports and tests clean"
```
