---
name: analyze-script-with-canvas
description: "四阶段视频剧本管线 (v3.0): 剧本提取 → CEFR分级改写 → LLM-first 时间轴规划 → 视频组装。Use when: (1) extracting structured screenplay from AI-generated videos with ASR, (2) rewriting dialogue at A2/B1/B2/C1 CEFR proficiency levels, (3) generating TimelinePlan via LLM planner + deterministic verifier/normalizer, (4) assembling final video via seedance + ffmpeg. Triggers: 'extract script', 'rewrite script', 'generate timeline plan', 'assemble video', '剧本提取', '分级改写'."
metadata:
  requires:
    bins: ["python3"]
    skills: ["script-extraction", "script-rewriting", "timeline_plan", "video_assembly"]
---

# analyze-script-with-canvas — 视频剧本管线套件

## 概述

一套四阶段的视频剧本处理管线 (v3.0 — LLM-First):

```
Stage 1                Stage 1b            Stage 2              Stage 3                     Stage 4
script-extraction  →  scene-detection  →  script-rewriting  →  timeline_plan           →  video_assembly
提取剧本+ASR            PySceneDetect切点    CEFR分级改写台词       LLM-first 时间轴规划        seedance局部生成+拼接
```

**核心设计原则**：
- 原视频时间轴控制最终剪辑，画布节点仅作为 prompt/参考图资产库
- Stage 3: LLM 处理所有语义决策（line-node 匹配、分组、prompt 改写），确定性代码仅做校验（schema、coverage、时间线几何）和执行（seedance/ffmpeg）
- Verifier 硬门: 无效 LLM draft 绝不允许进入执行链路

## 快速开始 (v3.0)

```bash
# Stage 1: 视频 → 剧本 JSON
python3 skills/script-extraction/extract_script.py \
  --video /path/to/video.mp4 \
  --utterances /path/to/asr.json \
  --output ep1_script.json

# Stage 2: 剧本 → 分级改写
python3 skills/script-rewriting/rewrite_script.py \
  --script ep1_script.json \
  --levels B2 \
  --output-dir rewrites/

# Stage 3: LLM-first 时间轴规划
python3 skills/timeline_plan/generate_plan.py \
  --script ep1_script.json \
  --rewrite rewrites/ep1_B2.json \
  --canvas canvas_data.json \
  --cuts scene_cuts.json \
  --output timeline_plan.json

# Stage 4: 视频组装 (原剧截取 + seedance 生成 + 拼接)
python3 skills/video_assembly/assemble.py \
  --plan timeline_plan.json \
  --video original.mp4 \
  --output final_B2.mp4
```

## 依赖

| Skill | 依赖 | 说明 |
|---|---|---|
| `script-extraction` | — | Stage 1，无上游依赖 |
| `script-rewriting` | `script-extraction` | Stage 2，消费 Stage 1 的 `ScriptInput` JSON |
| `timeline_plan` | `script-rewriting` | Stage 3 (v3.0)，LLM-first 规划器 + 确定性校验 |
| `video_assembly` | `timeline_plan` | Stage 4，消费 TimelinePlan，产出 mp4 |

## 项目结构 (v3.0)

```
skills/
├── SKILL.md                          # 本文件 — 管线套件入口
├── script-extraction/
│   ├── SKILL.md
│   └── extract_script.py
├── script-rewriting/
│   ├── SKILL.md
│   └── rewrite_script.py
├── scene_detection/
│   └── detect_scenes.py              # Stage 1b: PySceneDetect 切点
├── timeline_plan/                    # Stage 3: LLM-first 时间轴规划
│   ├── models.py                     # 确定性执行模型
│   ├── planner_models.py             # LLM 输出 schema (TimelinePlanDraft)
│   ├── evidence_builder.py           # LLM 输入打包
│   ├── llm_planner.py                # 双轮 LLM 规划 (推理 + 结构化输出)
│   ├── planner_verifier.py           # 确定性校验 (硬门)
│   ├── timeline_normalizer.py        # Draft → 可执行 TimelinePlan
│   ├── generate_plan.py              # 编排器 (CLI 入口)
│   ├── cut_fusion.py                 # ScriptShot + PySceneDetect 融合
│   └── validator.py                  # TimelinePlan 校验
└── video_assembly/                   # Stage 4: 视频组装
    └── assemble.py                   # 原剧截取 + seedance 生成 + 拼接
```

---

## v3.0 架构

Stage 1 (unchanged):  剧本提取 → VideoScriptOutput
Stage 1b (new):       场景检测 → CutPoints + KeyFrames
Stage 2 (unchanged):  CEFR 改写 → RewriteJSON
Stage 3 (v3.0):       LLM Planner + Verifier + Normalizer → timeline_plan.json
Stage 4 (unchanged):  视频组装 → final.mp4

### 与旧方案的对比

| 维度 | Legacy (canvas-storyboard) | v3.0 (timeline_plan) |
|------|---------------------------|----------------------|
| Stage 3 核心 | 画布节点匹配 + 规则分类 | LLM-first 规划 + 确定性校验 |
| 匹配方式 | 规则 _classify_operation_type | LLM 统一语义匹配 |
| Prompt 改写 | 按 operation_type 分策略 | LLM 语义编辑 |
| 安全机制 | 无硬门 | Verifier fail-fast + post-normalization check |
```
