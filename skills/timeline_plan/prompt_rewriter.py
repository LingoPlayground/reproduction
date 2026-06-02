"""Window Prompt Rewriter: rewrite canvas node prompts per GenerationWindow."""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import threading
import time
from pathlib import Path

from skills.timeline_plan.models import GenerationWindow, CanvasNode
from skills.timeline_plan._llm_utils import get_llm_client, _DEFAULT_MODEL, strip_markdown_fence

logger = logging.getLogger(__name__)

_LOG_DIR = Path("runs/v4_plans/rewriter_logs")
_log_counter = 0
_log_lock = threading.Lock()


def _log_llm(window_id: str, prompt: str, resp: str, dur: float):
    global _log_counter
    with _log_lock:
        _log_counter += 1
        counter = _log_counter
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    fpath = _LOG_DIR / f"{counter:03d}_rewrite_{window_id}_{time.strftime('%H%M%S')}.json"
    with open(fpath, "w") as f:
        json.dump({"window_id": window_id, "duration_sec": round(dur, 1),
                    "prompt_chars": len(prompt), "response_chars": len(resp),
                    "prompt": prompt, "response": resp}, f, ensure_ascii=False, indent=2)


def _make_rewrite_prompt(window: GenerationWindow, node: CanvasNode, level: str) -> str:
    changed_lines = []
    for atom in window.atoms:
        for line in atom.rewritten_lines:
            changed_lines.append({"line_id": line.line_id, "speaker": line.speaker,
                                   "original": line.original, "rewritten": line.rewritten})
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


def _check_rewritten_prompt(window: GenerationWindow, rewritten_prompt: str) -> list[str]:
    errors = []
    p_lower = rewritten_prompt.lower()
    for atom in window.atoms:
        for line in atom.rewritten_lines:
            rw = line.rewritten
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
    if not windows:
        return
    client = get_llm_client()
    if not client:
        logger.warning("No LLM client available — skipping prompt rewrite")
        return
    node_map: dict[str, CanvasNode] = {n.node_id: n for n in canvas_nodes}
    model = os.environ.get("LLM_PLANNER_MODEL", _DEFAULT_MODEL)

    def rewrite_one(window: GenerationWindow) -> None:
        if window.degradation_level >= 5:
            return
        nid = window.matched_node_id
        if not nid or nid not in node_map:
            window.degradation_level = max(window.degradation_level, 3)
            window.degradation_reason = "no_canvas_node_for_rewrite"
            return
        node = node_map[nid]
        ri_prompt = _make_rewrite_prompt(window, node, level)
        best_text = ""
        best_errors = float("inf")
        for attempt in range(3):
            t0 = time.time()
            try:
                resp = client.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": ri_prompt}],
                    temperature=0.3, max_tokens=32768,
                    reasoning_effort="low", extra_body={"thinking": {"type": "enabled"}},
                )
                text = strip_markdown_fence(
                    (resp.choices[0].message.content or "") if resp.choices else ""
                )
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(windows), 5)) as pool:
        list(pool.map(rewrite_one, windows))

    ok = sum(1 for w in windows if w.rewritten_prompt)
    logger.info("Rewriter: %d/%d windows have rewritten prompts", ok, len(windows))
