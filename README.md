# reproduction

Four-stage pipeline for video script reproduction with CEFR-graded dialogue.

```
Stage 1 (script-extraction) → Stage 2 (script-rewriting) → Stage 3 (canvas-storyboard) → Stage 4 (video-generation)
      原始视频+ASR                    CEFR 分级改写                 画布匹配+prompt替换            seedance生成+拼接
```

## Stage 1: script-extraction

Extract structured screenplay from AI-generated video + ASR transcription.
Uses multimodal LLM (Doubao Seed) to analyze video frames and map ASR utterances to shots/lines/characters.

**Requires**: [`lingolens`](https://github.com/LingoPlayground/lingolens) — `backend/agents/script_extraction/VideoScriptExtractor`

## Stage 2: script-rewriting

Rewrite dialogue at CEFR proficiency levels (A2, B1, B2, C1) using LLM-powered refinement.
One LLM call per CEFR level, with CEFR vocabulary index for quality guidance.

**Requires**: [`shakespeare`](https://github.com/LingoPlayground/shakespeare) — `FullRewriter`, `CEFRVocabIndex`, vocab data

## Stage 3: canvas-storyboard

Match script lines to LibLib Canvas video nodes, replace original dialogue in prompts with rewritten text.
Two-stage matching: fuzzy_match (~89%, free) + LLM semantic (deepseek-chat, completes to 100%).

## Stage 4: video-generation

Generate new videos via seedance 2.0 fast for nodes with replaced prompts, download unchanged originals,
concatenate into final episode.

**Requires**: `AQINFO_SEEDANCE_API_KEY` (from lingolens `.env`)

## Quick Start

```bash
# Stage 2: Rewrite a script
python3 skills/script-rewriting/rewrite_script.py \
  --script episode1_script.json --levels A2,B2,C1 --output-dir rewrites/

# Stage 3: Generate rewrite storyboard
python3 skills/canvas-storyboard/match_to_canvas.py \
  --script episode1_script.json --rewrite rewrites/ep1_B2.json \
  --canvas m2VuuIZfI --output storyboards/storyboard_ep1_B2.md --llm

# Stage 4: Generate final video
python3 skills/video-generation/generate_videos.py \
  --storyboard storyboards/storyboard_ep1_B2.md \
  --canvas canvas_data.json --script episode1_script.json \
  --output generated/ep1_B2.mp4
```

## Project Dependencies

| Project | Required for | Setup |
|---------|-------------|-------|
| [`lingolens`](https://github.com/LingoPlayground/lingolens) | Stage 1, Stage 4 | `git clone` to `~/workspace/lingolens` |
| [`shakespeare`](https://github.com/LingoPlayground/shakespeare) | Stage 2 | `git clone` to `~/workspace/shakespeare` |

Additional Python deps: `pydantic openai httpx python-dotenv chromadb nltk tqdm click sentence-transformers spacy openpyxl numpy`

spaCy model: `python -m spacy download en_core_web_md`
