#!/usr/bin/env python3
"""
Stage 2: CEFR-graded Script Rewriting

Wraps shakespeare FullRewriter to rewrite dialogue at CEFR levels.
Outputs one JSON file per level with backfilled shot context.

Usage:
  python3 skills/script-rewriting/rewrite_script.py \\
    --script ep1_script.json \\
    --levels A2,B1,B2,C1 \\
    --output-dir rewrites/ \\
    --output-prefix ep1
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# ── shakespeare import ─────────────────────────────────────────────────────
SHAKESPEARE_SRC = str(Path("~/workspace/shakespeare/src").expanduser().resolve())
sys.path.insert(0, SHAKESPEARE_SRC)

try:
    from shakespeare.models.script import ScriptInput
    from shakespeare.models.cefr import CEFRLevel
    from shakespeare.llm import LLMClient
    from shakespeare.vocab import CEFRVocabIndex
    from shakespeare.engine import FullRewriter
    from shakespeare.verifier import QualityVerifier
except ImportError as e:
    print(f"❌ Cannot import shakespeare module: {e}")
    print(f"   Ensure ~/workspace/shakespeare/src exists and is importable")
    print(f"   PYTHONPATH={SHAKESPEARE_SRC}")
    sys.exit(1)


# ── CLI ────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 2: CEFR-graded script rewriting"
    )
    parser.add_argument("--script", required=True, help="Stage 1 ScriptInput JSON")
    parser.add_argument(
        "--levels", default="A2,B1,B2,C1",
        help="Target CEFR levels, comma-separated (default: A2,B1,B2,C1)"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--output-prefix", default="rewrite", help="Output filename prefix")
    parser.add_argument("--temperature", type=float, default=0.3, help="LLM temperature")
    return parser.parse_args()


def parse_levels(levels_str: str) -> list[CEFRLevel]:
    """Parse comma-separated level string to CEFRLevel list."""
    valid = {l.value for l in CEFRLevel}
    levels = []
    for s in levels_str.split(","):
        s = s.strip().upper()
        if s not in valid:
            print(f"⚠️  Invalid level '{s}', skipping. Valid values: {sorted(valid)}")
            continue
        levels.append(CEFRLevel(s))
    if not levels:
        sys.exit("❌ No valid CEFR levels")
    return levels


# ── Shot context backfill ──────────────────────────────────────────────────


def build_line_context(script: ScriptInput) -> dict[str, dict]:
    """Build a map from line_id to shot context for output backfill."""
    ctx = {}
    for shot in script.script.shots:
        for line in shot.lines:
            ctx[line.line_id] = {
                "shot_number": shot.shot_number,
                "shot_scene": shot.scene_description,
                "start_seconds": line.start_seconds,
                "end_seconds": line.end_seconds,
                "speaker": line.speaker,
                "original": line.dialogue,
            }
    return ctx


def build_output_json(
    script: ScriptInput,
    level: CEFRLevel,
    rewrites: list,
    quality_report,
    line_context: dict[str, dict],
) -> dict:
    """Build per-level output JSON with backfilled shot context."""
    lines = []
    for r in rewrites:
        ctx = line_context.get(r.line_id, {})
        lines.append({
            "line_id": r.line_id,
            "shot_number": ctx.get("shot_number"),
            "speaker": ctx.get("speaker", ""),
            "original": ctx.get("original", r.original_dialogue or ""),
            "rewritten": r.rewritten_dialogue,
            "start_seconds": ctx.get("start_seconds"),
            "end_seconds": ctx.get("end_seconds"),
            "shot_scene": ctx.get("shot_scene"),
        })

    return {
        "title": script.script.title,
        "level": level.value,
        "lines": lines,
        "quality": {
            "cefr_precision": quality_report.cefr_precision,
            "cefr_recall": quality_report.cefr_recall,
            "matched_tokens": len(quality_report.matched_tokens),
            "total_words": quality_report.total_text_words,
        },
    }


async def main() -> None:
    args = parse_args()

    # 1. Load input script
    script_path = Path(args.script).resolve()
    if not script_path.exists():
        sys.exit(f"❌ Script file not found: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        script = ScriptInput(**json.load(f))

    dialogue_shots = [s for s in script.script.shots if s.lines]
    total_lines = sum(len(s.lines) for s in dialogue_shots)
    print(f"📖 Loaded script: {script.script.title} — {total_lines} lines")

    # 2. Build line context for backfill
    line_context = build_line_context(script)
    # Also enrich with speaker info
    for shot in script.script.shots:
        for line in shot.lines:
            if line.line_id in line_context:
                line_context[line.line_id]["speaker"] = line.speaker

    # 3. Parse target levels
    levels = parse_levels(args.levels)
    print(f"🎯 Target levels: {', '.join(l.value for l in levels)}")

    # 4. Initialize shakespeare components
    print("🔧 Building CEFR vocabulary index...")
    t0 = time.time()
    vocab = CEFRVocabIndex().build()
    print(f"   Done ({time.time() - t0:.1f}s)")

    llm = LLMClient()
    rewriter = FullRewriter(llm, vocab)
    verifier = QualityVerifier(vocab)

    # 5. Create output directory
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 6. Rewrite per level
    prefix = args.output_prefix
    for target in levels:
        print(f"\n{'─'*50}")
        print(f"🔄 Rewriting for {target.value}...")
        t_start = time.time()

        # Rewrite
        rewrites = await rewriter.rewrite(script, target, temperature=args.temperature)

        # Verify quality
        original_dialogues = [l.dialogue for s in dialogue_shots for l in s.lines]
        rewritten_dialogues = [r.rewritten_dialogue for r in rewrites]
        report = verifier.verify(original_dialogues, rewritten_dialogues, target)

        # Build output with backfilled context
        output = build_output_json(script, target, rewrites, report, line_context)

        # Write
        out_path = output_dir / f"{prefix}_{target.value}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        elapsed = time.time() - t_start
        print(f"✅ {target.value}: precision={report.cefr_precision:.3f} "
              f"recall={report.cefr_recall:.3f} ({elapsed:.0f}s)")
        print(f"   → {out_path}")

    print(f"\n{'='*50}")
    print(f"🎉 All rewrites complete! Output directory: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
