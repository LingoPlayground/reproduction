---
name: analyze-script-with-canvas
description: "三阶段视频剧本管线：剧本提取 → CEFR分级改写 → 画布匹配分镜。Use when: (1) extracting structured screenplay from AI-generated videos with ASR, (2) rewriting dialogue at A2/B1/B2/C1 CEFR proficiency levels, (3) matching rewritten lines to LibLib Canvas video generation nodes and producing multi-level storyboards. Triggers: 'extract script', 'rewrite script', 'CEFR rewrite', 'canvas storyboard', '分镜故事板', '剧本提取', '分级改写', '画布匹配'."
metadata:
  requires:
    bins: ["python3"]
    skills: ["script-extraction", "script-rewriting", "canvas-storyboard", "video-generation"]
---

# analyze-script-with-canvas — 视频剧本管线套件

## 概述

一套三阶段的视频剧本处理管线，将 AI 生成的影视视频转化为多级 CEFR 分级分镜故事板：

```
Stage 1: script-extraction         Stage 2: script-rewriting        Stage 3: canvas-storyboard      Stage 4: video-generation
┌──────────────────────┐         ┌──────────────────────────┐      ┌──────────────────────────┐      ┌──────────────────────────┐
│ 原始视频 + ASR 转录    │         │ 剧本 JSON (ScriptInput)    │      │ 单等级改写 JSON             │      │ storyboard + 原视频 URL     │
│     ↓                 │         │     ↓                     │      │ + LibLib Canvas 数据        │      │     ↓                      │
│ VideoScriptExtractor  │ ──────→ │ FullRewriter × 4 levels   │ ───→ │ fuzzy_match + LLM          │ ───→ │ seedance 生成 + 下载 + 拼接  │
│     ↓                 │         │     ↓                     │      │ + prompt 台词替换            │      │     ↓                      │
│ ep1_script.json      │         │ ep1_A2/B2/C1.json         │      │ storyboard_ep1_A2.md       │      │ ep1_A2.mp4                │
└──────────────────────┘         └──────────────────────────┘      └──────────────────────────┘      └──────────────────────────┘
```

**核心设计原则**：
- 每阶段产出**独立文件**，可单独使用或串联
- Stage 2 每个 CEFR 等级产出独立 JSON，Stage 3 对每个等级独立匹配
- 改写（LLM）和匹配（算法）完全解耦

## 快速开始（完整管线）

```bash
# Stage 1: 视频 → 剧本 JSON
python3 skills/script-extraction/extract_script.py \
  --video /path/to/video.mp4 \
  --utterances /path/to/asr.json \
  --output ep1_script.json

# Stage 2: 剧本 → 分级改写（每个等级独立输出）
python3 skills/script-rewriting/rewrite_script.py \
  --script ep1_script.json \
  --levels A2,B1,B2,C1 \
  --output-dir rewrites/

# → 产出: rewrites/ep1_A2.json, rewrites/ep1_B1.json, rewrites/ep1_B2.json, rewrites/ep1_C1.json

# Stage 3: 原版分镜 → 改写版分镜
python3 skills/canvas-storyboard/match_to_canvas.py \
  --script ep1_script.json \
  --canvas m2VuuIZfI \
  --output storyboards/original_ep1.md \
  --llm

python3 skills/canvas-storyboard/match_to_canvas.py \
  --script ep1_script.json \
  --rewrite rewrites/ep1_A2.json \
  --canvas m2VuuIZfI \
  --output storyboards/storyboard_ep1_A2.md \
  --llm

# Stage 4: 生成新视频 + 拼接
python3 skills/video-generation/generate_videos.py \
  --storyboard storyboards/storyboard_ep1_A2.md \
  --canvas runs/canvas_data.json \
  --script episode1_script.json \
  --output generated/ep1_A2.mp4
```

## 子 Skill 依赖

| Skill | 依赖 | 说明 |
|---|---|---|
| `script-extraction` | — | 第一步，无上游依赖 |
| `script-rewriting` | `script-extraction`（输出格式兼容） | 消费 Stage 1 的 `ScriptInput` JSON |
| `canvas-storyboard` | `script-rewriting`（输出格式兼容） | 消费 Stage 2 的单等级 `rewrite` JSON |
| `video-generation` | `canvas-storyboard`（输出格式兼容） | 消费 Stage 3 的 storyboard，产出 mp4 |

## 项目结构

```
skills/
├── SKILL.md                          # 本文件 — 管线套件入口
├── script-extraction/
│   ├── SKILL.md                      # Stage 1 skill 定义
│   └── extract_script.py             # 包装脚本
├── script-rewriting/
│   ├── SKILL.md                      # Stage 2 skill 定义
│   └── rewrite_script.py             # 包装脚本
└── canvas-storyboard/
    ├── SKILL.md                      # Stage 3 skill 定义
    └── match_to_canvas.py            # 匹配脚本
└── video-generation/
    ├── SKILL.md                      # Stage 4 skill 定义
    └── generate_videos.py            # 生成+拼接脚本
```
