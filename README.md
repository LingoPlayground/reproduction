# reproduction

Four-stage pipeline for video script reproduction with CEFR-graded dialogue.

## Pipeline (v3.0 — LLM-First)

```
Stage 1                  Stage 1b               Stage 2                Stage 3                     Stage 4
script-extraction   →   scene-detection    →   script-rewriting   →   timeline_plan           →   video_assembly
提取剧本+ASR             PySceneDetect切点       CEFR分级改写台词        时间轴匹配+prompt改写        seedance局部重生成+拼接
```

**Core principle:** the original video timeline controls the final edit. Canvas nodes are used only as
prompt/ref-image asset libraries. Unchanged segments are directly cut from the original video;
only rewritten dialogue segments are regenerated via seedance.

### Quick Start (v3.0)

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

## Stage 1: script-extraction

Extract structured screenplay from AI-generated video + ASR transcription.
Uses multimodal LLM (Doubao Seed) to analyze video frames and map ASR utterances to shots/lines/characters.

**Requires**: [`lingolens`](https://github.com/LingoPlayground/lingolens) — `backend/agents/script_extraction/VideoScriptExtractor`

## Stage 2: script-rewriting

Rewrite dialogue at CEFR proficiency levels (A2, B1, B2, C1) using LLM-powered refinement.
One LLM call per CEFR level, with CEFR vocabulary index for quality guidance.

**Requires**: [`shakespeare`](https://github.com/LingoPlayground/shakespeare) — `FullRewriter`, `CEFRVocabIndex`, vocab data

## Stage 3: timeline_plan

LLM-first pipeline: a single unified LLM call handles all semantic decisions —
line-to-node matching, grouping, prompt rewriting, and risk assessment.
Deterministic code handles only validation (schema, coverage, timeline geometry)
and execution (seedance/ffmpeg).

**Key features:**
- LLM Planner (two-pass: reasoning + structured output with JSON schema)
- Deterministic Verifier (hard gate: known IDs, coverage, duplicates, dialogue alignment, time ranges)
- Timeline Normalizer (draft → executable plan with duration enforcement and gap fill)
- Post-normalization coverage check (catches split/merge bugs)
- Retry loop with validation feedback (max 3 attempts, fail fast on unrecoverable errors)

## Stage 4: video_assembly

Consumes a `TimelinePlan` JSON to produce the final video. Segments marked `source=original`
are cut from the input video; segments marked `source=seedance` trigger seedance regeneration.

**Post-assembly integrity checks:**
- Segment count must match planned item count
- Output duration must match planned total (within 2s tolerance)

**Requires**: `AQINFO_SEEDANCE_API_KEY` (from lingolens `.env`)

## Project Dependencies

| Project | Required for | Setup |
|---------|-------------|-------|
| [`lingolens`](https://github.com/LingoPlayground/lingolens) | Stage 1, Stage 4 | `git clone` to `~/workspace/lingolens` |
| [`shakespeare`](https://github.com/LingoPlayground/shakespeare) | Stage 2 | `git clone` to `~/workspace/shakespeare` |

Additional Python deps: `pydantic openai httpx python-dotenv chromadb nltk tqdm click sentence-transformers spacy openpyxl numpy`

spaCy model: `python -m spacy download en_core_web_md`
