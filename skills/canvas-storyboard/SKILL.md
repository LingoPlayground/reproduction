---
name: canvas-storyboard
description: "Stage 3: 两步流程 — (A) 原剧本匹配画布节点产出原版分镜, (B) 在原版映射基础上替换 prompt 台词产出改写分镜。两阶段匹配: fuzzy_match (free, ~89%) + LLM semantic (deepseek, 补全至100%)。Use when: 'match to canvas', '画布匹配', 'storyboard with canvas', '分镜故事板', 'map rewrites to video nodes'."
metadata:
  requires:
    bins: ["python3"]
    skills: ["analyze-script-with-canvas", "script-rewriting"]
---

# canvas-storyboard — 画布匹配分镜 (Stage 3)

## 概述

**两步流程**：

```
Step A (原版)                          Step B (改写版)
原剧本 JSON → 两阶段匹配               原版映射(复用) + 改写台词 JSON
  ├─ fuzzy_match (89%, free)           → prompt 中替换原台词
  └─ LLM semantic (11%, ~$0.0002)      → 改写版分镜
  → 100% 原版分镜
```

**两阶段匹配架构**：

| 阶段 | 方法 | 命中率 | 成本 |
|---|---|---|---|
| Stage 1 | `fuzzy_match_score`（子串→滑窗→词命中） | ~89% | 免费 |
| Stage 2 | LLM 语义匹配（`deepseek-chat`） | 补全至 100% | ~$0.0002/ep |

**核心设计**：
- 匹配同时使用**台词原文**和**场景描述**（`scene_description`），中文匹配中文
- LLM 处理 ASR 转录偏差（`"Face test okay"` ↔ `"扫脸认证"`）
- 改写版**不重新匹配**，复用原版 `line_id → 节点` 映射
- 改写版只改 prompt 中的台词，不改映射关系、参考图、视频 URL

## 前置条件

| 依赖 | 说明 |
|---|---|
| Python 3.x | 运行环境 |
| LibLib Canvas API | `https://api.liblib.tv/api/canvas/project/share/detail?shareId=<id>`（公开分享，需中国 IP） |

## CLI 用法

### Step A: 生成原版 storyboard

```bash
python3 skills/canvas-storyboard/match_to_canvas.py \
  --script episode1_script.json \
  --canvas m2VuuIZfI \
  --output storyboards/original_ep1.md
```

### Step B: 生成改写版 storyboard

```bash
python3 skills/canvas-storyboard/match_to_canvas.py \
  --script episode1_script.json \
  --rewrite rewrites/ep1_A2.json \
  --canvas m2VuuIZfI \
  --output storyboards/storyboard_ep1_A2.md
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--script` | ✅ | 原始剧本 JSON（ScriptInput 格式） |
| `--canvas` | ✅ | 画布 shareId 或本地 JSON |
| `--output` | ✅ | 输出 markdown 路径 |
| `--rewrite` | ❌ | Stage 2 改写 JSON（不提供则产出原版） |
| `--llm` | ❌ | 启用 LLM 语义匹配补全未映射行（默认 false） |
| `--min-score` | ❌ | fuzzy_match 最低分数阈值（默认 55） |

## Step A: 原版映射逻辑

### 两阶段匹配

| 阶段 | 方法 | 命中率 | 成本 |
|---|---|---|---|
| Stage 1 | `fuzzy_match_score`（子串→滑窗→词命中）+ scene_description 辅助 | ~89% | 免费 |
| Stage 2 | LLM 语义（`deepseek-chat`）处理 ASR 偏差和短台词 | 补全至 100% | ~$0.0002/ep |

Stage 1 对每行（台词 + scene_description）与全部 114 视频节点 prompt 比对，取最高分+最新节点。Stage 2 仅对 Stage 1 未命中的行调用 LLM，一次性提交全部未匹配行+全部节点摘要。

### 原版 storyboard 输出格式

```markdown
# Episode 1 — 原版分镜故事板

## 镜头 1 (26.7s) — Donny
*场景描述...*

| line_id | 台词 | 对话 | 对应画布节点 |
|---------|------|------|-------------|
| p001_l001 | 💬 Donny | this ceremony is boring | ✅ 视频节点 2 - 副本 |
| p001_l002 | 💬 Donny | let's see who wants me | *(同上)* |
```

## Step B: Prompt 台词替换

在原版映射基础上：
1. 查找原台词在 prompt 中的**精确位置**
2. 用改写台词替换
3. 保留 prompt 结构、中文上下文、参考图、视频 URL 不变

### 替换策略（4 级降级）

1. **引号内匹配**：`"This ceremony is boring."` → `"This party is boring."`
2. **冒号后匹配**：`说到: This ceremony is boring.` → `说到: This party is boring.`
3. **精确子串**：直接大小写不敏感查找
4. **滑窗定位**：处理 ASR 转录偏差（如 `You are serious, huh?` → prompt 中实际为 `Are you serious?`）

### 改写版 storyboard 输出格式

```markdown
# Episode 1 — A2 等级改写分镜故事板

## 镜头 1 (26.7s) — Donny
*场景描述...*

| line_id | 台词 | 原台词 | A2 改写 | 对应画布节点 | Prompt 替换 |
|---------|------|--------|---------|-------------|-------------|
| p001_l001 | 💬 Donny | this ceremony is boring | This party is boring. | 视频节点 2 - 副本 | ✅ 已替换 |

🔄 **视频节点 2 - 副本 Prompt 替换**:
```
...慵懒地嘟囔："This party is boring."（这派对太无聊了。）...
```

参考图、视频 URL 不变。
```

## 已知限制

| 限制 | 缓解 |
|---|---|
| 极短对话（<5 字符） | Stage 2 LLM 语义匹配接手 |
| ASR 转录偏差（措辞不同） | Stage 2 LLM 语义匹配接手 |
| 画布 API 不稳定（SSL/IncompleteRead） | 使用本地缓存的 `canvas_data.json` |
| LLM 依赖 `--llm` | 需 `DEEPSEEK_API_KEY`（从 `~/workspace/lingolens/backend/.env` 加载） |
