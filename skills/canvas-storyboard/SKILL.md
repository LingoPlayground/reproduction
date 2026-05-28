---
name: canvas-storyboard
description: "Stage 3: LLM e2e匹配 — 用 Rule A (上下文连续性) + Rule B (废片剔除) 评分的 LLM 端到端匹配,将剧本台词映射到画布视频节点。Supports multi-run voting (--llm-runs N) for higher accuracy. Use when: 'match to canvas', '画布匹配', 'storyboard with canvas', '分镜故事板', 'map rewrites to video nodes'."
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
原剧本 JSON → LLM e2e 匹配             原版映射(复用) + 改写台词 JSON
  ├─ 单次匹配 (默认)                    → prompt 中替换原台词
  └─ 多次投票 (--llm-runs 5)            → 改写版分镜
  → 原版分镜
```

**LLM 端到端匹配架构**：

| 配置 | 方法 | 质量保证 |
|---|---|---|
| 默认 (1次) | `llm_end_to_end_match` | Rule A/B 提示词约束 |
| `--llm-runs 5` | 5 次匹配 + `score_mapping` 投票 | 每次随机打乱节点顺序，取最高分 |

**核心设计**：
- **Rule A (上下文连续性)**: Line N → Node X 暗示 Line N+1 → Node X 或 X+1，打破时间线的节点被排除
- **Rule B (废片剔除)**: `has_video=false` 的副本节点优先级低于 `has_video=true`，连续台词不跨副本节点
- 匹配同时使用**台词原文**和**场景描述**，LLM 处理 ASR 偏差和跨语言匹配
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
| `--llm-runs` | ❌ | LLM 投票轮数（默认 5，设 1 则单次匹配） |

## Step A: 原版映射逻辑

### LLM 端到端匹配

LLM 一次性接收全部节点和台词，通过优化后的 system prompt（Rule A/B 评分约束）进行全局匹配。

| 规则 | 说明 |
|---|---|
| Rule A | 上下文连续性：Line N → Node X 暗示 Line N+1 → Node X 或 X+1 |
| Rule B | 废片剔除：`has_video=false` 的副本优先级低于 `has_video=true` |

**评分函数** `score_mapping()` 在投票模式下用于选优：
- 未匹配行: -100/行
- Rule A 违规（同镜内节点跳跃 >5）: -30/次
- Rule B 违规（选了废片而非成片）: -20/次
- 匹配奖励: +5/行

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
| ASR 转录偏差（措辞不同） | LLM 语义匹配 + Rule A/B 评分 |
| 画布含废片副本 | Rule B 剔除 `has_video=false` 副本 |
| 大 Context 下 LLM "Lost in the Middle" | 投票时每轮随机打乱节点顺序 |
| LLM 依赖 `DEEPSEEK_API_KEY` | 从 `~/workspace/lingolens/backend/.env` 加载 |
