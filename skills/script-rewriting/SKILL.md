---
name: script-rewriting
description: "Stage 2: CEFR-graded dialogue rewriting. 将剧本按 A2/B1/B2/C1 等级改写，每个等级产出独立台词 JSON。Use when user says: 'rewrite script', 'CEFR rewrite', '分级改写', 'adapt dialogue level', '重新措辞台词'."
metadata:
  requires:
    bins: ["python3"]
    skills: ["analyze-script-with-canvas", "script-extraction"]
---

# script-rewriting — CEFR 分级剧本改写 (Stage 2)

## 概述

将结构化剧本的对话按 **CEFR 等级**（A2, B1, B2, C1）改写。每个等级**独立调用一次 LLM**，产出**独立的台词 JSON 文件**，供 Stage 3 分别匹配画布节点。

**核心设计**：
- 每等级独立输出 → 与 Stage 3 完全解耦
- 保留原始 shot 上下文（shot_number, scene_description, timing）
- LLM v6 prompt 保证角色一致性 + 自然口语化
- CEFR 词汇索引提供改写提示

## 前置条件

### 必须安装

```bash
# 基础依赖
pip install pydantic openai httpx python-dotenv chromadb nltk tqdm click

# ⚠️ 隐藏依赖（pyproject.toml 未声明）
pip install sentence-transformers spacy openpyxl numpy

# spaCy 模型
python -m spacy download en_core_web_md
```

### 环境变量

在 `~/workspace/shakespeare/.env` 或当前目录 `.env` 中配置：

```bash
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=Pro/zai-org/GLM-5.1
```

### 词汇数据

`~/workspace/shakespeare/data/CuriosSea分级知识点_cleaned.xlsx` 必须存在。

### 首次运行

首次运行需下载模型（~120MB）：
- SentenceTransformer `all-MiniLM-L6-v2` (~80MB)
- spaCy `en_core_web_md` (~40MB)

### shakespeare 导入验证

```bash
cd ~/workspace/shakespeare
PYTHONPATH=src python3 -c "from shakespeare.engine import FullRewriter; print('OK')"
```

## CLI 用法

```bash
python3 skills/script-rewriting/rewrite_script.py \
  --script ep1_script.json \
  --levels A2,B1,B2,C1 \
  --output-dir rewrites/ \
  --output-prefix ep1
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--script` | ✅ | — | Stage 1 产出的 ScriptInput JSON |
| `--levels` | ❌ | `A2,B1,B2,C1` | 目标 CEFR 等级（逗号分隔，支持 A1-C2） |
| `--output-dir` | ❌ | `.` | 输出目录 |
| `--output-prefix` | ❌ | `rewrite` | 输出文件名前缀 |
| `--temperature` | ❌ | `0.3` | LLM temperature |

## 输出格式

每个 CEFR 等级产出独立 JSON。文件名：`{prefix}_{level}.json`

### `{prefix}_A2.json`

```json
{
  "title": "Episode 1",
  "level": "A2",
  "lines": [
    {
      "line_id": "p001_l001",
      "shot_number": 1,
      "speaker": "Donny",
      "original": "this ceremony is boring",
      "rewritten": "This ceremony is boring.",
      "start_seconds": 2.83,
      "end_seconds": 4.19,
      "shot_scene": "Donny stands in the crowded graduation celebration..."
    },
    {
      "line_id": "p001_l002",
      "shot_number": 1,
      "speaker": "Donny",
      "original": "let's see who wants me",
      "rewritten": "Let's see who wants me!",
      "start_seconds": 5.3,
      "end_seconds": 10.6,
      "shot_scene": "Donny stands in the crowded graduation celebration..."
    }
  ],
  "quality": {
    "cefr_precision": 0.85,
    "cefr_recall": 0.12,
    "matched_tokens": 42,
    "total_words": 128
  }
}
```

| 字段 | 说明 |
|---|---|
| `line_id` | 原始行 ID，贯穿三阶段 |
| `shot_number` | ⬅ 从原始剧本回填，Stage 3 用于分镜分组 |
| `original` | 原始台词 |
| `rewritten` | 改写后台词 |
| `shot_scene` | ⬅ 从原始剧本回填 |
| `start_seconds` / `end_seconds` | ⬅ 从原始剧本回填 |
| `quality.cefr_precision` | 改写词汇中属于目标 CEFR 等级的比例 |
| `quality.cefr_recall` | 目标 CEFR 词汇库中被覆盖的比例 |

### 文件产出示例

```bash
# 运行后产出
rewrites/ep1_A2.json    # A2 等级改写台词
rewrites/ep1_B1.json    # B1 等级改写台词
rewrites/ep1_B2.json    # B2 等级改写台词
rewrites/ep1_C1.json    # C1 等级改写台词
```

## 改写引擎

**FullRewriter** (`~/workspace/shakespeare/src/shakespeare/engine/full_rewriter.py`)：
- v6 单次 LLM 调用 / 等级
- 角色感知：保持 personality + 说话风格
- 剧情保持：不改动人名、地点、情节
- 11 项自检规则（反膨胀、反过度解释、反格式改写等）
- CEFR 词汇提示注入（可选）

## 依赖关系

```
Stage 1 输出                     Stage 2 输入
VideoScriptOutput          →     ScriptInput (字段兼容)
  meta.characters[].role_id  →   meta.characters[].role_id  ✅
  meta.characters[].speaker_name → meta.characters[].name    ✅
  meta.summary               →   meta.summary                ✅
  script.title               →   script.title                ✅
  script.shots[].lines[].*   →   script.shots[].lines[].*    ✅
```

Stage 1 → Stage 2 可直接 `.model_dump_json()` → `ScriptInput(**json)`，无需中间转换。

## 已知限制

- **角色画像可选**：`traits/speech_style/typical_vocabulary` 缺失时 LLM 自行推理，质量降级
- **原始台词回填**：FullRewriter 输出不包含 `original_dialogue`，`rewrite_script.py` 按 `line_id` 从输入补回
- **质量评分**：仅计算 CEFR 词汇覆盖率，不代表改写语义质量
- **每个等级一次 LLM 调用**：4 个等级 = 4 次调用，按输入行数线性增长
