# Stage 3: 剪辑计划生成 — 详细设计

## 1. 概述

Stage 3 是 Pipeline v2 的核心。输入 Stage 1 的剧本 + Stage 2 的改写 + LibLib 画布节点，输出一份精确的 `TimelinePlan`，告诉 Stage 4：原剧视频的哪一段需要被 seedance 生成的视频替换，哪一段直接截取原剧。

### 1.1 输入

| 数据 | 来源 | 格式 |
|------|------|------|
| ScriptOutput | Stage 1 (lingolens) | `{script: {shots: [{shot_number, start_seconds, end_seconds, scene_description, lines: [{line_id, dialogue, speaker}]}]}}` |
| RewriteJSON | Stage 2 (shakespeare) | `{level: "B2", lines: [{line_id, shot_number, speaker, original, rewritten, start_seconds, end_seconds, shot_scene}]}` |
| Canvas Nodes | LibLib API | `[{nodeId, prompt, video_url, reference_images}]` |
| CutPoints (可选) | Stage 1b (PySceneDetect) | `[{time_sec, confidence}]` |
| KeyFrames (可选) | Stage 1b | `[{time_sec, image_path, shot_number}]` |

### 1.2 输出

```python
TimelinePlan:
  title: str
  level: str               # CEFR 等级
  pipeline_version: "2.0"
  total_duration_sec: float
  items: [TimelinePlanItem]

TimelinePlanItem:
  shot_id: str             # 标识
  shot_number: int         # 原始 shot 编号
  source: "seedance" | "original"
  start_sec: float         # 在原剧视频中的开始时间
  end_sec: float           # 在原剧视频中的结束时间
  scene_description: str
  ref_images: [str]        # seedance 参考图 URL
  rewritten_prompt: str    # 改写后的 prompt（仅 seedance）
  matched_node_id: str     # 匹配到的画布节点 ID
  match_confidence: float  # 匹配置信度 (0.0-1.0)
  degradation_level: int   # 降级层级 (0=最优, 1=无参考图, 2=无节点匹配)
  seedance_duration: int   # seedance 生成时长 (-1=智能)
  original_duration: float # 原始时间范围
```

---

## 2. 处理流程

```
generat_timeline_plan(Stage3Input)
│
├─ 1. 筛选改写行: original ≠ rewritten
│     rewritten_lines = [rl for rl in all if rl.original != rl.rewritten]
│
├─ 2. LLM CoT 匹配: 行 → 画布节点
│     match_lines_to_nodes(rewritten_lines, canvas_nodes, num_runs=3)
│     输出: {node_id: [line_id, ...]}, {line_id: confidence}
│
├─ 3. Per-node: 连续分组 + merge-up
│     for each (node_id, line_ids):
│       node_rewrite_lines.sort(by start_seconds)
│       groups = _split_contiguous(node_rewrite_lines)
│       for each group:
│         if duration < 4s → fallback original (孤立短组)
│         else → create TimelinePlanItem(source="seedance")
│           ├─ 时间范围: [min(start), max(end)]
│           ├─ 参考图: node.ref_images || keyframes
│           ├─ seedance_duration: -1 (智能)
│           ├─ rewritten_prompt: extract_and_rewrite_prompt(node.prompt, group)
│           └─ match_confidence: mean of cross-run consistency
│
├─ 4. 剩余 shot: original 片段
│     for each shot without rewritten lines → TimelinePlanItem(source="original")
│
├─ 5. 未匹配改写行: degradation fallback
│     无节点匹配的行 → TimelinePlanItem(source="seedance", degradation=2)
│
├─ 6. 重叠移除: seedance 优先
│     移除与 seedance 项时间重叠的 original 项
│
└─ 7. 排序输出
     items.sort(by start_sec) → TimelinePlan
```

---

## 3. LLM 调用 1: 行→节点匹配

**文件**: `canvas_matcher.py::match_lines_to_nodes()`

### 3.1 输入格式

发送给 LLM 的 line_entries:
```json
{
  "line_id": "p001_l001",
  "dialogue": "this ceremony is boring",
  "speaker": "Donny",
  "shot_number": 1,
  "shot_scene": "Donny stands in the crowded graduation...",
  "start_seconds": 2.83,
  "end_seconds": 4.19
}
```

发送给 LLM 的 node_entries (prompt 截断到 3000 chars):
```json
{
  "id": 0,
  "node_id": "13126e5a-...",
  "prompt": "美式情景喜剧，真实短剧..."
}
```

### 3.2 System Prompt

```
## Role
You match script dialogue lines to the canvas nodes that generated them.

## Step 1: Extract Dialogue from Node Prompts (CoT)
Each node's prompt is a video generation instruction. It contains:
- Scene descriptions, camera directions, visual style (in Chinese)
- Actual spoken dialogue (in English, often inside quotation marks "")
- Non-dialogue quoted text: banner signs ("CLASS OF 2026"),
  sound effects ("黑胶唱片划痕声"), inner thoughts, character descriptions

For each node, identify ONLY the actual spoken English dialogue lines.

## Step 2: Match Lines to Nodes
For each script line, find the node whose prompt contains that dialogue.

Matching signals (priority order):
1. Dialogue text in the node's prompt (primary — node prompt is ground truth)
2. Speaker attribution — same character should map to nodes
   where that character appears
3. Scene description — the visual context should match the node's scene

The script dialogue comes from ASR — expect minor errors.
Match semantically.

## Contiguity Constraint
Lines from the same shot should map to the same or adjacent nodes.
Avoid mapping widely separated shots to the same node.

## Output
JSON only:
{"mappings": [
  {"line_id": "p001_l001", "node_index": 0},
  {"line_id": "p001_l003", "node_index": 2}
]}
Use `node_index` (the "id" field), NOT `node_id`.
If no node matches, omit the line.
```

### 3.3 Voting 机制

3 次运行，每次 shuffle node 顺序以消除位置偏见。

**评分** (`_score_mapping`):
```python
score = match_count
for each shot:
  for consecutive lines in shot:
    if mapped to different nodes → score -= 0.5 (contiguity penalty)
```

**置信度** (`_compute_consistency`):
```python
for each line_id:
  confidence = most_common_node_count / total_runs
  # 3 次都匹配到同一节点 → confidence = 1.0
  # 3 次匹配到 3 个不同节点 → confidence = 0.33
```

### 3.4 输出

```python
node_line_groups = {"13126e5a-...": ["p001_l001", "p001_l002"], ...}
line_confidences = {"p001_l001": 0.67, "p001_l002": 1.0, ...}
```

---

## 4. LLM 调用 2: Prompt 改写

**文件**: `prompt_extractor.py::extract_and_rewrite_prompt()`

### 4.1 输入

```
Original Prompt: 画布节点的完整原始 prompt (858-1461 chars)
Dialogue to Rewrite: [{speaker, original, rewritten}, ...]
Scene Context: shot 的场景描述 (fallback 用)
```

### 4.2 System Prompt

```
## Role
You rewrite video generation prompts for seedance, keeping only visual
content tied to rewritten dialogue.

The original prompt mixes style settings (resolution, lighting, camera style),
scene descriptions with camera angles and character actions, and dialogue
in quotes. Style settings apply to the entire video and must be preserved.
Scene descriptions should be kept only if they contain dialogue being
rewritten — within them, keep only the visuals directly around the dialogue
moment and cut background filler. Remove entire scenes with no rewritten
dialogue.

Replace each original dialogue line with its rewritten version, preserving
the speaker attribution format.

## Output
Rewritten prompt text only. No explanations, no JSON.
```

### 4.3 Fallback

LLM 不可用或无 prompt → `_generate_prompt_from_scene()`:
```
"{scene_description}\n{speaker} says: \"{rewritten}\"\n..."
```

---

## 5. 连续分组与 Merge-Up

**文件**: `generate_plan.py::_split_contiguous()`

### 5.1 算法

```
输入: node_rewrite_lines (已按 start_seconds 排序)
常量: MAX_GAP_SEC = 5.0, MIN_SEEDANCE_DURATION = 4.0

Step 1: Split by time gap
  current = [lines[0]]
  for each subsequent line:
    if line.start - current[-1].end > MAX_GAP_SEC:
      groups.append(current); current = [line]
    else:
      current.append(line)
  groups.append(current)

Step 2: Merge-up short groups
  for each group:
    if duration >= MIN_SEEDANCE_DURATION:
      keep
    else:
      try merge-forward → next group
      try merge-backward → previous group
      else → keep as isolated (caller falls back to original)
```

### 5.2 示例

```
改写行时间: [2.8-4.2] [5.3-6.4] [17.5-18.3] [18.8-20.0] [21.1-29.6]

Step 1: 按 gap > 5s 拆分
  group A: [2.8-4.2, 5.3-6.4]    dur=3.6s ← < 4s
  group B: [17.5-18.3, 18.8-20.0, 21.1-29.6]  dur=12.1s ✅

Step 2: merge-up
  group A (3.6s < 4s) → merge-forward → group B
  → 最终: 1 个 group, 17.5-29.6s (含 2.8-6.4s 的内容)
```

---

## 6. 降级层级

| Level | 含义 | 触发条件 |
|-------|------|---------|
| 0 | 最优路径 | 节点匹配成功 + 参考图可用 + prompt 改写成功 |
| 1 | 无参考图 | 节点匹配成功但 reference_images 为空，使用 keyframes |
| 2 | 无节点匹配 | LLM 匹配失败，使用 scene_description 生成 prompt |

---

## 7. 重叠处理

两个时间体系不互相对齐：
- **seedance 时间**: 改写行的 ASR 时间戳 (min/max)
- **original 时间**: 多模态 LLM 的 ScriptShot 边界

处理策略: **seedance 优先**——移除与任何 seedance 项时间范围重叠的 original 项。

```python
for each original item:
  if overlaps with any seedance item:
    skip  # seedance covers this time range
```

---

## 8. CLI 接口

```bash
python3 skills/timeline_plan/generate_plan.py \
  --script episode1_script.json \     # Stage 1 输出
  --rewrite rewrites/ep1_B2.json \    # Stage 2 输出
  --canvas canvas_data.json \         # LibLib 画布节点
  --cuts cuts.json \                  # Stage 1b 输出 (可选)
  --keyframes keyframes.json \        # Stage 1b 输出 (可选)
  --output timeline_plan.json \       # 输出路径
  --level B2                          # CEFR 等级
```

输出:
```
Timeline plan: 8 items -> timeline_plan.json
  [seedance] Shot 1: 2.8s-6.4s node=13126e5a-e6f4...
  [original] Shot 4: 11.0s-12.5s
  [seedance] Shot 3: 32.8s-43.5s node=1a1f4741-7e7e...
  ...
```

---

## 9. 模块依赖

```
generate_plan.py (编排器)
  ├── models.py (数据模型)
  ├── canvas_matcher.py (LLM 匹配)
  │     └── models.py
  ├── prompt_extractor.py (LLM 改写)
  │     └── models.py
  └── cut_fusion.py (时间融合)
        └── models.py
```
