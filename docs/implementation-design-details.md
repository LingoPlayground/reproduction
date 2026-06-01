# 台词改写→画布节点匹配→智能改Prompt→生成新视频→替换回原视频：整体设计与实施细节

> 本文档基于项目代码（skills/）、设计文档（docs/superpowers/specs/、docs/superpowers/plans/）、README、运行产物（runs/）综合梳理而成。
> 
> 日期：2026-05-31

---

## 目录

1. [总体架构概览](#1-总体架构概览)
2. [Stage 1：原视频 → 结构化剧本](#2-stage-1原视频--结构化剧本)
3. [Stage 1b：PySceneDetect 画面切点检测](#3-stage-1bpyscenedetect-画面切点检测)
4. [Stage 2：CEFR 分级台词改写](#4-stage-2cefr-分级台词改写)
5. [Stage 3：核心——改写台词选取、画布节点匹配、Prompt 改写、替换规划](#5-stage-3核心改写台词选取画布节点匹配prompt-改写替换规划)
   - [5.1 主流程入口](#51-主流程入口)
   - [5.2 Evidence 构建](#52-evidence-构建)
   - [5.3 LLM Planner 单次规划](#53-llm-planner-单次规划)
   - [5.4 Planner Verifier 硬门](#54-planner-verifier-硬门)
   - [5.5 Timeline Normalizer：时间轴替换表达](#55-timeline-normalizer时间轴替换表达)
   - [5.6 Post-Normalization 覆盖校验](#56-post-normalization-覆盖校验)
6. [Stage 4：Seedance 生成新视频 + 拼接替换回原视频](#6-stage-4seedance-生成新视频--拼接替换回原视频)
   - [6.1 主入口](#61-主入口)
   - [6.2 Original 片段 → ffmpeg 原样切](#62-original-片段--ffmpeg-原样切)
   - [6.3 Modified 片段 → Seedance 生成](#63-modified-片段--seedance-生成)
   - [6.4 拼接与完整性检查](#64-拼接与完整性检查)
7. [旧 Matcher/Composer 路径（已删除，仅历史参考）](#7-旧-matchercomposer-路径已删除仅历史参考)
   - [7.1 主路径：llm_planner.py](#71-主路径llm_plannerpy)
   - [7.2 旧路径：canvas_matcher.py + edit_planner.py + prompt_composer.py（已删除）](#72-旧路径canvas_matcherpy--edit_plannerpy--prompt_composerpy已删除)
8. [设计文档中的 v3 理想方案 vs 当前实现](#8-设计文档中的-v3-理想方案-vs-当前实现)
9. [数据模型全览](#9-数据模型全览)
10. [已识别风险与设计问题](#10-已识别风险与设计问题)
    - [P0：没有确定性验证 "只改台词，不改环境/动作"](#p0没有确定性验证-只改台词不改环境动作)
    - [P1：短片段扩展策略过于机械](#p1短片段扩展策略过于机械)
    - [P1：Seedance Fallback 可能导致改写台词静默消失](#p1seedance-fallback-可能导致改写台词静默消失)
     - [P1：主路径与旧路径并存，维护复杂度高（已部分解决）](#p1主路径与旧路径并存维护复杂度高已部分解决)
    - [P2：Keyframes 未真正进入多模态匹配](#p2keyframes-未真正进入多模态匹配)
    - [P2：Unmatched Line 直接 Fail，缺少优雅降级](#p2unmatched-line-直接-fail缺少优雅降级)
     - [P2：Duration Resolver 策略未迁移（文件已删除）](#p2duration-resolver-策略未迁移文件已删除)
11. [相关文件清单](#11-相关文件清单)

---

## 1. 总体架构概览

### 核心原则

> 原视频时间轴控制最终剪辑；Liblib TV 画布节点只作为 Prompt/Reference Image 资产库。未改写片段直接从原视频切；只有改写台词相关片段才通过 Seedance 局部重生成。

### 四阶段 Pipeline

```
Stage 1                    Stage 1b                    Stage 2                   Stage 3                               Stage 4
script-extraction   →     scene-detection   →     script-rewriting   →     timeline_plan (LLM-First)   →     video_assembly
                                                                                                                    │
     │                        │                        │                         │                                    │
     ▼                        ▼                        ▼                         ▼                                    ▼
 Video + ASR              PySceneDetect            ScriptOutput              Evidence Dict                        TimelinePlan
     │                    CutPoints + KeyFrames         │                         │                                    │
     │                        │                        │                         │                                    │
     │                        │                        ▼                         │                                    │
     │                        │            RewriteJSON (lines with               │                                    │
     │                        │             original/rewritten/                   │                                    │
     │                        │             timings/shot_context)                │                                    │
     │                        │                        │                         │                                    │
     └────────────────────────┴────────────────────────┴────── Canvas Nodes ─────┘                                    │
                                                                                                                        │
                                                                                        ┌───────────────────────────────┘
                                                                                        ▼
                                                                                  最终视频 (final.mp4)
```

各阶段职责：

| Stage | 目录 | 职责 | 关键外部依赖 |
|-------|------|------|-------------|
| 1 | `skills/script-extraction/` | 视频 + ASR → 结构化剧本 | lingolens VideoScriptExtractor |
| 1b | `skills/scene_detection/` | PySceneDetect 画面切点 + 关键帧 | PySceneDetect |
| 2 | `skills/script-rewriting/` | CEFR 分级改写台词 | shakespeare FullRewriter |
| 3 | `skills/timeline_plan/` | 匹配画布节点、改写 Prompt、规划替换时间轴 | DeepSeek LLM |
| 4 | `skills/video_assembly/` | ffmpeg 切原片 + Seedance 生成 + 拼接 | Seedance 2.0 |

---

## 2. Stage 1：原视频 → 结构化剧本

### 代码位置

```
skills/script-extraction/extract_script.py
skills/script-extraction/SKILL.md
```

### 职责

输入原始 AI 视频 + ASR 转录 → 输出结构化剧本 JSON。

### 实现原理

- 包装外部项目 `lingolens` 的 `VideoScriptExtractor`；
- 多模态 LLM（Doubao Seed）分析视频帧，将 ASR utterances 映射到 shots/lines/characters；
- Azure ASR 提供句子级时间戳。

### 输出结构（ScriptOutput）

```json
{
  "script": {
    "shots": [
      {
        "shot_number": 1,
        "start_seconds": 0.0,
        "end_seconds": 3.0,
        "scene_description": "Donny stands in the crowded graduation celebration, holding a glass of champagne...",
        "lines": [
          {
            "line_id": "p001_l001",
            "speaker": "Donny Li",
            "dialogue": "this ceremony is boring",
            "start_seconds": 2.83,
            "end_seconds": 4.19
          }
        ]
      }
    ]
  },
  "title": "Episode 1"
}
```

### 重要时间体系差异

| 时间系统 | 来源 | 精度 | 示例 |
|---------|------|------|------|
| `shot.start_seconds / end_seconds` | 多模态 LLM 分析视频帧 | 秒级 | 0.0–3.0s |
| `line.start_seconds / end_seconds` | Azure ASR utterance 时间戳 | 毫秒级 | 2.83–4.19s |

两者不对齐，Stage 3 的 `cut_fusion.py` 负责处理这个差异。

---

## 3. Stage 1b：PySceneDetect 画面切点检测

### 代码位置

```
skills/scene_detection/detect_scenes.py
skills/scene_detection/tests/test_detect_scenes.py
```

### 职责

- PySceneDetect content-aware shot boundary detection；
- 在切点位置提取关键帧图像（供 Stage 3 多模态参考）。

### 输出

```json
// CutPoints
[{"time_sec": 0.0, "confidence": 1.0}, {"time_sec": 8.3, "confidence": 0.92}, ...]

// KeyFrames
[{"time_sec": 0.0, "image_path": "/path/to/keyframe_001.jpg", "shot_number": 1}, ...]
```

### 当前实际接入

- `CutPoints` 已传入 Stage 3，用于 `cut_fusion.py` 的 shot 边界融合；
- `KeyFrames` 当前未真正传入 LLM（`evidence_builder.py` 不接 keyframes）。

---

## 4. Stage 2：CEFR 分级台词改写

### 代码位置

```
skills/script-rewriting/rewrite_script.py
skills/script-rewriting/SKILL.md
```

### 职责

输入剧本 JSON → 按指定 CEFR 等级（A2/B1/B2/C1）产生独立改写台词 JSON。

### 实现原理

包装外部项目 `shakespeare` 的组件：

| 组件 | 职责 |
|------|------|
| `LLMClient` | LLM 调用客户端 |
| `FullRewriter` | 按等级独立调用 LLM 改写 |
| `CEFRVocabIndex` | 分级词汇索引（用于质量引导） |
| `QualityVerifier` | 输出质量的 CEFR precision/recall 校验 |

### 关键设计：每等级独立输出 + 上下文回填

```python
# rewrite_script.py 中的回填逻辑
# 从原始 ScriptInput 中，按 line_id 补全：
{
  "title": "Episode 1",
  "level": "B2",
  "lines": [
    {
      "line_id": "p001_l001",
      "shot_number": 1,
      "speaker": "Donny",
      "original": "this ceremony is boring",
      "rewritten": "This ceremony is not entertaining at all.",
      "start_seconds": 2.83,
      "end_seconds": 4.19,
      "shot_scene": "Donny stands in the crowded graduation celebration..."
    }
  ],
  "quality": {
    "cefr_precision": 0.85,
    "cefr_recall": 0.12,
    "matched_tokens": 45,
    "total_words": 312
  }
}
```

Stage 2 回填的 `shot_scene`、`start_seconds`、`end_seconds`、`shot_number` 是 Stage 3 的核心输入，不做这个回填 Stage 3 就无法按时间轴做替换。

---

## 5. Stage 3：核心——改写台词选取、画布节点匹配、Prompt 改写、替换规划

这是整条链路最核心的部分。

### 5.1 主流程入口

**文件**：`skills/timeline_plan/generate_plan.py`

**函数**：

```python
generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan
```

**主路径**：

```text
Stage3Input
  │
  ├─→ build_evidence()           构建统一 evidence 给 LLM
  ├─→ generate_plan_draft()       LLM Planner 单次规划
  ├─→ verify_draft()              确定性验证硬门
  ├─→ normalize_plan()            时间轴归一化（纯几何）
  ├─→ post-normalization 覆盖校验 检查每个改写行都被覆盖
  ├─→ validate_timeline_item()    最终校验
  └─→ TimelinePlan JSON          产出可执行计划
```

### 5.2 Evidence 构建

**文件**：`skills/timeline_plan/evidence_builder.py`

**函数**：

```python
build_evidence(script_shots, rewrite_lines_all, canvas_nodes, cut_points, level) -> Dict
```

#### 改写行筛选

```python
is_rewritten = entry["original"] != entry["rewritten"]
if is_rewritten:
    rewritten_lines.append(entry)   # 进入“必须处理”队列
else:
    unchanged_lines.append(entry)   # 作为上下文参考
```

核心逻辑：**只要 original 和 rewritten 不同，就认为这行需要被匹配、生成和替换**。

#### Canvas Node 传入 LLM

```python
for node in canvas_nodes:
    nodes.append({
        "node_id": node.node_id,
        "prompt": prompt,                         # 完整 prompt（不截断）
        "reference_image_count": len(node.reference_images),  # 只传数量，不传URL
        "video_url": node.video_url,
    })
```

当前不传 reference image URL 给 LLM，也不传 keyframes。

#### Evidence 完整结构

```json
{
  "rewrite_lines": [{             // 需要处理的改写行
    "line_id", "original", "rewritten", "speaker",
    "start_sec", "end_sec", "shot_number", "shot_scene"
  }],
  "neighbor_lines": [...],         // 不改写行（最多50条上下文）
  "canvas_nodes": [{               // 画布节点 prompt
    "node_id", "prompt", "reference_image_count", "video_url"
  }],
  "scene_context": [{              // shot_scene 描述
    "shot_number", "description"
  }],
  "timeline": {
    "scene_cuts": [...],           // PySceneDetect 切点
    "video_duration_sec": ...
  },
  "constraints": {
    "must_cover_every_rewritten_line": true,
    "must_not_duplicate_lines": true,
    "must_preserve_environment_action_style": true,
    "min_modified_duration_sec": 4.0
  }
}
```

### 5.3 LLM Planner 单次规划

**文件**：`skills/timeline_plan/llm_planner.py`

**函数**：

```python
generate_plan_draft(evidence, max_retries=3) -> TimelinePlanDraft
```

**模型**：`deepseek-v4-pro`（可配 `LLM_PLANNER_MODEL` 环境变量）

**一次 LLM 调用完成所有语义工作**：

1. **Line-to-Node Matching**：每行改写台词 → 找到包含其原台词（或视觉匹配）的画布节点；
2. **Node Grouping**：同节点、时间邻近的台词合并为一组；
3. **Prompt Rewriting**：改写 prompt 中的对话部分，保留环境/动作/镜头/风格；
4. **Unmatched / Unused**：标注无法匹配的行和未使用的节点。

#### LLM Prompt 核心规则

```text
1. 每行改写台词必须 exactly once（在 node_generations 或 unmatched_lines）
2. 不能出现重复 line_id
3. 每个 node_generation 必须有非空 rewritten_prompt
4. rewritten_prompt 必须逐字包含改写台词
5. source_time_range 必须精确覆盖被包含台词的真实时间戳
6. line_matches 必须覆盖 group 的全部 line_id
7. preserved_environment / actions / style 需列出原 prompt 保留元素
8. group 时长必须在 4–30 秒
9. Only change spoken dialogue. 全部视觉描述/角色动作/镜头方向/照明/风格关键词逐字保留。
10. 如果 prompt 没有引号英文原文但视觉场景匹配 → 自然地插入改写台词
```

#### 输出结构（TimelinePlanDraft）

```python
@dataclass
class TimelinePlanDraft:
    plan_version: str
    overall_reasoning: str
    node_generations: List[NodeGeneration]    # 每组包含：covered_line_ids, matched_node_ids,
                                              #   source_time_range, rewritten_prompt, line_matches,
                                              #   preserved_*, changed_only_dialogue, confidence
    unmatched_lines: List[UnmatchedLine]      # 未被匹配的台词
    unused_nodes: List[UnusedNode]            # 未被使用的节点
    all_lines_covered: bool
    no_duplicate_coverage: bool
    risk_notes: List[str]
```

#### 重试机制

```python
for attempt in range(1, max_retries + 1):
    draft = run_planner(evidence, validation_feedback=last_errors)
    errors = verify_draft(draft, evidence)
    if not errors:
        return draft
    last_errors = errors    # 喂给下一次作为「修复提示」
```

- 默认 max 3 次；
- 每次 retry 把 fail 原因作为 validation_feedback 传给 LLM；
- 如果 3 次仍 fail → ValueError（不适合继续）。

### 5.4 Planner Verifier 硬门

**文件**：`skills/timeline_plan/planner_verifier.py`

**函数**：

```python
verify_draft(draft, evidence) -> List[str]    # 空列表 = 验证通过
```

**5 类阻塞性校验**：

| 检查 | 内容 | 阻断条件 |
|------|------|---------|
| Schema | `group_id`、`rewritten_prompt`、`covered_line_ids`、`line_matches` 非空 | 任一缺失 |
| Known IDs | 所有 `line_id` / `node_id` 必须在 evidence 中存在 | 出现未知 ID |
| Coverage | 每个改写行恰好出现一次（covered XOR unmatched），无缺失/无重复 | 有遗漏或重复 |
| Time Consistency | `source_time_range` 覆盖被包含台词的 ASR 时间（±2s 容忍） | 超出 ±2s |
| Prompt Integrity | `rewritten_dialogue_in_prompt` 逐字出现在 `rewritten_prompt` | 改写台词不在 prompt 中 |

**任何一项不通过 → 拒绝该 draft → 触发 retry 或直接 ValueError**。

### 5.5 Timeline Normalizer：时间轴替换表达

**文件**：`skills/timeline_plan/timeline_normalizer.py`

**函数**：

```python
normalize_plan(draft, script_shots, canvas_nodes, cut_points, keyframes,
               video_duration, title, level) -> TimelinePlan
```

**纯几何处理，不做语义判断**。职责划分：

| 步骤 | 代码 | 职责 |
|------|------|------|
| 1 | `normalize_plan()` | 每个 `NodeGeneration` → 1 个 `TimelinePlanItem(source="modified")` |
| 2 | `normalize_plan()` | 不足 4 秒的 → pad 到 4 秒 |
| 3 | `normalize_plan()` | 有 unmatched_lines → ValueError（不接受无声丢失） |
| 4 | `normalize_plan()` | 每个 ScriptShot → 1 个 `TimelinePlanItem(source="original")` |
| 5 | `_finalize()` | 从 original 中 carve out modified 片段 |
| 6 | `_finalize()` | 合并相邻 original 片段 |
| 7 | `_finalize()` | 填充 gap |

#### Carve-Out 示意

```text
原始时间轴：0 ───────────────────────────── 30

Modified 片段：         5 ───── 9

最终：   0 ─ 5   |   5 ─ 9   |   9 ─ 30
         origin     modified     origin
```

#### 不足 4 秒处理

```python
if duration < MIN_MODIFIED_DURATION:   # 4.0
    end_sec = start_sec + MIN_MODIFIED_DURATION
```

### 5.6 Post-Normalization 覆盖校验

在 `generate_plan.py` 中，normalizer 之后会再做一次覆盖检查：

```python
all_rewrite_ids = {rl["line_id"] for rl in evidence["rewrite_lines"]}
final_covered = {each plan item's covered_line_ids}

missing = all_rewrite_ids - set(final_covered)     # 遗漏 → error
dups = {lid: shots for lid, shots in final_covered if len(shots) > 1}  # 重复 → error
unknown = set(final_covered) - all_rewrite_ids      # 未知 ID → error
```

这个检查用于捕获 normalizer 在处理过程中因为 merge / split / gap fill 导致的覆盖丢失。

---

## 6. Stage 4：Seedance 生成新视频 + 拼接替换回原视频

### 6.1 主入口

**文件**：`skills/video_assembly/assemble.py`

**函数**：

```python
assemble_video(plan_path, original_video, output_path, skip_seedance=False) -> output_path
```

### 6.2 Original 片段 → ffmpeg 原样切

```python
if source == "original" or skip_seedance:
    subprocess.run([
        "ffmpeg", "-y",
        "-ss", f"{item['start_sec']:.3f}",
        "-i", original_video,
        "-t", f"{item['end_sec'] - item['start_sec']:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        seg_path,
    ])
```

### 6.3 Modified 片段 → Seedance 生成

```python
if source == "modified":
    video_url = await _generate_via_seedance(item, duration)
```

**`_generate_via_seedance()` 内部流程**：

```text
1. 读取 rewritten_prompt
2. 读取 ref_images（画布节点的 reference_image URL）
3. 逐个下载参考图到本地（_download_image_locally）
4. 上传到 Aliyun OSS（_upload_local_image）
5. 创建 seedance asset（client.create_asset）
6. 调用 seedance:
   model = SEEDANCE_2_0_FAST
   ratio = RATIO_9_16
   resolution = RESOLUTION_720P
   generate_audio = True
   prompt = rewritten_prompt
   images = asset IDs
   duration = 计划时长
7. 下载生成视频
```

#### Seedance 输出后处理

| 情况 | 处理 | 日志标记 | 最终结果 |
|------|------|---------|---------|
| 生成太短（< planned - 0.5s） | fallback 到 original | `[SEED-FB]` | 实际使用原视频片段 |
| 生成太长（> planned + 0.3s） | ffmpeg trim 到计划时长 | — | 符合计划时长 |
| 生成成功，时长 OK | 直接使用 | `[SEED]` | 新生成片段 |
| 下载失败 / API 异常 | fallback 到 original | `[SEED-FB]` | 实际使用原视频片段 |

### 6.4 拼接与完整性检查

所有片段产出后：

```text
1. 统一编码：libx264 + yuv420p + aac
2. 统一音量：EBU R128 loudnorm (I=-16, LRA=11, TP=-1.5)
3. ffmpeg concat 拼接
4. 校验：
   - 分段数量 == 计划数量
   - 输出时长偏差 ≤ 2 秒
```

```python
if len(segment_paths) != planned_total:
    raise RuntimeError("Segment count mismatch")

if abs(drift) > 2.0:
    raise RuntimeError("Duration drift too large")
```

---

## 7. 旧 Matcher/Composer 路径（已删除，仅历史参考）

> **⚠️ 文档更新（2026-05-31）**：旧路径文件（`canvas_matcher.py`、`edit_planner.py`、`prompt_composer.py`、`prompt_extractor.py`、`duration_resolver.py`）已从工作树中删除，不再与主路径并存。但旧路径中包含的部分能力尚未迁移到主路径，见下方说明。

### 7.1 主路径：llm_planner.py

**当前 `generate_plan.py` import 的路径**：

```python
from skills.timeline_plan.evidence_builder import build_evidence
from skills.timeline_plan.llm_planner import generate_plan_draft
from skills.timeline_plan.planner_verifier import verify_draft
from skills.timeline_plan.timeline_normalizer import normalize_plan
```

**特点**：

| 维度 | 说明 |
|------|------|
| 调用次数 | 1 次 LLM 调用完成全部语义工作 |
| 匹配方式 | LLM 隐式匹配（prompt 中包含完整的 node prompt 文本） |
| 分组方式 | LLM 按同节点 + 时间邻近决定 |
| Prompt 改写 | LLM 一次性输出 rewritten_prompt |
| 自说明 | LLM 产出 `preserved_environment` / `actions` / `style` / `changed_only_dialogue` |
| 验证 | `planner_verifier.py` 5 层确定性校验 |
| 模型 | `deepseek-v4-pro`（较大模型） |

**优点**：简洁、上下文集中、一次决策避免多步传递错误。  
**风险**：语义决策完全由 LLM 负责，'只改台词'等关键要求缺乏确定性校验。

### 7.2 旧路径：canvas_matcher.py + edit_planner.py + prompt_composer.py（已删除）

以下文件曾在 `skills/timeline_plan/` 下存在、可导入、被测试引用，但从未被主路径 `generate_plan.py` 调用。**当前已从工作树中删除**，仅在 git 历史中保留：

| 文件 | 职责 | 状态 |
|------|------|:----:|
| `canvas_matcher.py` | 多次 LLM run + shuffle + voting 做 line-to-node matching | 已删除 |
| `edit_planner.py` | 判断 operation_type（literal_replace / semantic_insert 等） | 已删除 |
| `prompt_composer.py` | 分层 prompt 编辑 + L3/L4 校验 + style preservation fallback | 已删除 |
| `prompt_extractor.py` | 旧版 prompt 重写（只支持 literal_replace） | 已删除 |
| `duration_resolver.py` | 短组扩展策略（pad_after/before/forced） | 已删除 |

#### canvas_matcher.py 的匹配算法

```
1. 对每组（所有 line + 所有 node）：
   a. 把 node prompt 截断到 3000 字符
   b. LLM CoT 先提取 prompt 中的 spoken dialogue
   c. 然后逐行匹配
2. 默认 3 次 run，每次打乱 node 顺序（防位置偏差）
3. _score_mapping() 评分：
   - 匹配行数为正分
   - 同 shot 连续台词被分到不同节点 → -0.5 惩罚
4. 取 quality score 最高的 run 作为 best mapping
5. 用 cross-run majority vote 增强（补全在 best run 中缺失但其他 run 匹配到的行）
6. 置信度 = most_common_node / total_runs
```

#### prompt_composer.py 的改写策略

| Operation Type | 场景 | 做法 |
|---------------|------|------|
| `literal_replace` | prompt 中有原台词原文 | 直接替换 |
| `fuzzy_replace` | prompt 中有近似原文 | 语义近似匹配 + 替换 |
| `semantic_insert` | prompt 无原台词，但视觉描述匹配 | 保留视觉描述 + 插入改写台词 |
| `section_reconstruct` | prompt 中某 section 不完整/格式错误 | 从 scene_description 重建 |
| style preserving fallback | 以上都失败但还有 style layer | 保留 style prefix + scene + dialogue |
| full fallback | 无可用 node | 纯 scene_description 生成 |

**当前状态**：这些文件已从工作树中删除。"两套架构并存"的维护负担已消除，但旧路径中以下能力**尚未迁移到主路径**，属于功能缺口：

1. **`prompt_composer.py` 的 style-preserving fallback**：在主路径的 prompt rewriting 失败时，旧路径有分层 fallback（保留 style prefix + scene + dialogue）作为兜底，当前主路径缺少这一安全网；
2. **`duration_resolver.py` 的 pad_before/pad_after 策略**：旧路径支持按语义判断向前/向后扩展短片段（避免盲目后移导致内容重叠），主路径当前仅做 `end_sec = start_sec + 4.0` 的简单后移；
3. **`canvas_matcher.py` 的多轮 voting 匹配思想**：通过多次 LLM run + shuffle（防位置偏差）+ cross-run majority vote 增强匹配置信度，主路径仅做一次 LLM 调用完成匹配，缺少多轮交叉验证。

---

## 8. 设计文档中的 v3 理想方案 vs 当前实现

设计文档 `docs/superpowers/specs/2026-05-30-multimodal-canvas-edit-planner-design.md` 提出了更完整的目标架构，当前主路径实现了其中一部分。

### 匹配信号

| v3 设计 | 当前实现状态 |
|---------|:---------:|
| `quoted_dialogue`（引号台词） | ✅ 主路径 prompt 支持 |
| `fuzzy_dialogue`（近似台词） | ✅ 主路径 prompt 支持 |
| `speaker_presence`（角色存在） | ✅ LLM 隐式使用 |
| `visual_action`（动作匹配） | ✅ 可以作为 semantic_insert 触发条件 |
| `shot_scene_similarity`（场景匹配） | ✅ LLM 使用 shot_scene 上下文 |
| `temporal_order`（时间顺序） | ✅ LLM 隐式使用 |
| `implicit_visual_scene`（隐式视觉） | ✅ 主路径 prompt 明确支持 |
| `reference_image_match`（参考图匹配） | ❌ 未传入 LLM |

### Duration 扩展策略

| v3 设计 | 当前实现状态 |
|---------|:---------:|
| `pad_after` | ❌ 代码存在但不接主路径 |
| `pad_before` | ❌ 代码存在但不接主路径 |
| `snap_to_cut` | ❌ 未实现 |
| `borrow_neighbor` | ❌ 未实现 |
| `hold_reaction` | ❌ 未实现 |
| `merge_same_node_group` | ❌ 未实现 |
| `cross_node_merge` | ❌ 未实现 |
| `forced_min_duration` | ⚠️ 当前直接用 `end_sec = start_sec + 4.0` |

### 操作类型

| v3 设计 | 当前实现状态 |
|---------|:---------:|
| `literal_replace` | ✅ `prompt_composer.py`、主路径 prompt 均支持 |
| `fuzzy_replace` | ⚠️ 只在 `prompt_composer.py` 中 |
| `semantic_insert` | ✅ 主路径 prompt 明确支持 |
| `section_reconstruct` | ⚠️ 只在 `prompt_composer.py` 中 |
| `style_preserving_fallback` | ✅ `prompt_composer.py` 有实现 |
| `full_fallback` | ✅ 旧 `prompt_extractor.py` 有 |

### 降级等级

| 设计 Level | 当前实现 |
|:-----------:|:--------|
| 0: direct_canvas_patch | ⚠️ 未显式追踪 |
| 1: implicit_canvas_patch | ❌ |
| 2: style_preserving_fallback | ❌ |
| 3: duration_padded_or_borrowed | ⚠️ 当前 simple pad = level 3 语义，但代码中不标记 |
| 4: cross_node_composite | ❌ |
| 5: full_fallback | ❌ |
| 6: original | ⚠️ Seedance fail → original fallback，但不标记 level |

当前实际 `degradation_level`：

```python
# reference_images 为空 → level 1
if not ref_images:
    degradation_level = 1

# duration < 4s → degradation_reason = "duration_padded"
```

粒度远小于 v3 设计。

---

## 9. 数据模型全览

### models.py（确定性执行模型）

```python
@dataclass
class CanvasNode:
    node_id: str
    prompt: str
    video_url: str
    reference_images: List[str]
    duration_sec: Optional[float]

@dataclass
class TimelinePlanItem:
    shot_id: str
    shot_number: int
    source: Literal["original", "modified"]
    start_sec: float
    end_sec: float
    scene_description: str
    ref_images: List[str]
    rewritten_prompt: Optional[str]
    matched_node_id: Optional[str]
    match_confidence: Optional[float]
    degradation_level: int
    original_duration: Optional[float]
    covered_line_ids: List[str]
    source_node_ids: List[str]
    degradation_reason: str

@dataclass
class TimelinePlan:
    title: str
    level: str
    original_video_path: str
    total_duration_sec: float
    items: List[TimelinePlanItem]
    metadata: Dict[str, Any]

@dataclass
class Stage3Input:
    script_output: Any
    video_cut_points: List[CutPoint]
    keyframes: List[KeyFrame]
    node_cut_points: Dict[str, List[CutPoint]]
    rewrite_json: Dict[str, Any]
    canvas_nodes: List[CanvasNode]
    level: str
```

**常量**：

```python
MIN_MODIFIED_DURATION = 4.0
MAX_MODIFIED_DURATION = 30.0
```

### planner_models.py（LLM 输出模型）

```python
@dataclass
class SourceTimeRange:
    start_sec: float
    end_sec: float

@dataclass
class LineNodeMatch:
    line_id: str
    original_line: str
    rewritten_line: str
    node_id: str
    match_reasoning: str
    original_dialogue_in_prompt: Optional[str]
    rewritten_dialogue_in_prompt: str
    confidence: float

@dataclass
class NodeGeneration:
    group_id: str
    covered_line_ids: List[str]
    matched_node_ids: List[str]
    source_time_range: Optional[SourceTimeRange]
    rewritten_prompt: str
    reference_image_node_ids: List[str]
    line_matches: List[LineNodeMatch]
    grouping_reasoning: str
    prompt_rewrite_reasoning: str
    preserved_environment: List[str]     # LLM 自声明
    preserved_actions: List[str]          # LLM 自声明
    preserved_style: List[str]            # LLM 自声明
    changed_only_dialogue: bool           # LLM 自声明
    confidence: float

@dataclass
class TimelinePlanDraft:
    plan_version: str
    node_generations: List[NodeGeneration]
    unmatched_lines: List[UnmatchedLine]
    unused_nodes: List[UnusedNode]
    overall_reasoning: str
    all_lines_covered: bool
    no_duplicate_coverage: bool
    risk_notes: List[str]
```

---

## 10. 已识别风险与设计问题

### P0：没有确定性验证"只改台词，不改环境/动作"

**严重性：最高**

虽然 LLM prompt 强烈要求"Only change spoken dialogue"、LLM 也输出 `preserved_environment / actions / style` 等自声明字段，但：

- `planner_verifier.py` 只校验 `rewritten_dialogue_in_prompt` 是否在 `rewritten_prompt` 中出现；
- `NodeGeneration.preserved_environment/actions/style/changed_only_dialogue` 是 LLM **自声明**字段，没有确定性交叉验证；
- `validator.py` 的 L4 style preservation 只检查 20 个硬编码风格关键词（美式情景喜剧、8k、cinematic 等）；
- **没有代码检查原始 prompt 的场景名词、角色名、动作短语、镜头类型、运镜方式是否保留**。

```python
# planner_verifier.py 的 prompt integrity check：
for m in g.line_matches:
    if m.rewritten_dialogue_in_prompt not in g.rewritten_prompt:
        error   # 只检查台词本身在不在 prompt 中
```

```python
# validator.py 的 style preservation check：
STYLE_ANCHORS_KEYWORDS = [   # 20 个硬编码关键词
    "美式情景喜剧", "真实短剧", "柔光雾化", "电影级布光",
    "8k", "超高清", "画面通透", "cinematic", ...
]
# 只检查这些关键词保留 ≥60%
```

**建议**：增加从 original prompt 中抽取 scene/action/camera anchors 的代码，在 `planner_verifier` 中对 `rewritten_prompt` 做保留率校验。如果 LLM 把"面部特写，固定机位，男主后退"改成了"中景，推镜头，男主跑开"，应能被发现。

---

### P1：短片段扩展策略过于机械

当前代码：

```python
if duration < MIN_MODIFIED_DURATION:  # 4.0
    end_sec = start_sec + MIN_MODIFIED_DURATION
```

这种方式的问题是：

1. 不知道补进去的 0.x 秒是否包含未改写的台词；
2. 不知道补进去的片段是否跨 PySceneDetect 切点；
3. 不知道是否应该向前补而不是向后补；
4. 没有语义化的 scene cut snap、neighbor borrow、hold reaction 扩展；
5. 扩展的时长策略不记录到 `TimelinePlanItem`，下游无法追溯。

设计文档提出的扩展策略优先级：

```text
1. pad_after          (补 ≤0.5s 到 4s)
2. pad_before         (补 ≤0.5s 到 4s)
3. snap_to_cut        (吸附SceneDetect切点)
4. hold_reaction      (LLM判断情绪/表情可延长)
5. borrow_neighbor    (同node临近未改台词)
6. merge_same_node    (同node远距离组合)
7. cross_node_merge   (不同node合并)
8. forced_min_duration(最后手段)
```

当前只实现了 `forced_min_duration`（效果同 8，但形式不是强制 pad 而是直接 end_sec 后移）。

---

### P1：Seedance Fallback 可能导致改写台词静默消失

`assemble.py` 中：

```python
if actual > 0 and actual < planned_duration - 0.5:
    # fallback to original segment
```

当 Seedance 输出太短时，最终视频里这个 segment 仍会使用原始视频片段。这意味着：

- 最终输出路径上有完整的 mp4；
- 所有 `[SEED-FB]` 信息只打日志；
- `covered_line_ids` 没有对应的替换生效；
- **用户看到完整的视频，以为所有改写都生效了，但实际上这部分台词仍是原文**。

**建议**：在 assembly 结束后生成一份执行报告：

```json
{
  "segment_id": "mod_G1",
  "planned_source": "modified",
  "actual_source": "original_fallback",
  "reason": "seedance_output_too_short",
  "affected_line_ids": ["p001_l001", "p001_l002"]
}
```

这份报告应随最终视频一起产出，让调用方知道哪些改写台词实际生效、哪些没生效。

---

### P1：主路径与旧路径并存，维护复杂度高（已部分解决）

> **⚠️ 更新**：旧路径文件已从工作树中删除，两条路径的代码级并存问题已消除。但旧路径中的 `canvas_matcher.py` 对 `models.NodeSection` 的 import 在旧 models 中已不存在——即便文件恢复也无法直接导入，代码已腐化。

当前仓库 `skills/timeline_plan/` 下保留的文件：

| 路径 | 文件 | 被主 `generate_plan.py` 调用？ |
|------|------|:---:|
| ✅ 主路径 | `evidence_builder.py`, `llm_planner.py`, `planner_verifier.py`, `timeline_normalizer.py`, `validator.py` | 是 |
| ⚠️ 半集成 | `cut_fusion.py` | 是（被 `timeline_normalizer` 调用） |
| ❌ 已删除 | `canvas_matcher.py`, `edit_planner.py`, `prompt_composer.py`, `prompt_extractor.py`, `duration_resolver.py` | 历史存在，现已删除 |

删除后引入的新缺口：旧路径中更细致的策略（style-preserving fallback、pad_before/pad_after、多轮 voting 匹配）未迁移到主路径。详见 [§7.2](#72-旧路径canvas_matcherpy--edit_plannerpy--prompt_composerpy已删除)。

---

### P2：Keyframes 未真正进入多模态匹配

`Stage3Input` 有 `keyframes` 字段，`normalize_plan()` 接收 keyframes 参数，但：

- `evidence_builder.py` 不接收 keyframes；
- LLM planner 看不到关键帧图像；
- 多模态 canvas edit planner 的目标尚未实现。

---

### P2：Unmatched Line 直接 Fail，缺少优雅降级

```python
if draft.unmatched_lines:
    raise ValueError(
        f"Cannot normalize: {len(draft.unmatched_lines)} lines are unmatched."
    )
```

如果某句改写台词确实找不到匹配节点，当前 pipeline 直接失败。

理论上可做优雅降级：

1. style_preserving_fallback（保留 style prefix + scene + dialogue）；
2. 用 shot_scene 作为 prompt 源码；
3. 标记为 `degradation_level=5` 继续执行。

---

### P2：Duration Resolver 策略未迁移（文件已删除）

`duration_resolver.py` 有 3 种 pad 策略：

```python
def pad_after(all_lines_map, line_to_node, rewrite_lines, ...) -> (start_sec, end_sec, strategy):
def pad_before(all_lines_map, line_to_node, rewrite_lines, ...) -> (start_sec, end_sec, strategy):
def forced_min_duration(all_lines_map, line_to_node, rewrite_lines, ...) -> (start_sec, end_sec, strategy):
```

但该文件已随旧路径一同删除。当前主路径里 `timeline_normalizer.py` 直接：

```python
if duration < MIN_MODIFIED_DURATION:
    end_sec = start_sec + MIN_MODIFIED_DURATION
```

没有语义化的 pad_before/pad_after 策略。且旧 `duration_resolver.py` 的 `all_lines_map` 和 `line_to_node` 参数接收后从未被使用，即便恢复也需要重写。

---

## 11. 相关文件清单

### 核心 Pipeline（Stage 3 主路径）

| 文件 | 职责 |
|------|------|
| `skills/timeline_plan/generate_plan.py` | Stage 3 编排器、CLI 入口 |
| `skills/timeline_plan/evidence_builder.py` | LLM 证据包构造 |
| `skills/timeline_plan/llm_planner.py` | LLM 单次规划 |
| `skills/timeline_plan/planner_models.py` | LLM 输出数据结构 |
| `skills/timeline_plan/planner_verifier.py` | 5 层确定性验证硬门 |
| `skills/timeline_plan/timeline_normalizer.py` | 时间轴归一化（几何处理） |
| `skills/timeline_plan/cut_fusion.py` | ScriptShot + PySceneDetect 时间融合 |
| `skills/timeline_plan/models.py` | 确定性执行数据模型 |
| `skills/timeline_plan/validator.py` | 多层级最终校验（1/2/4/5） |

### 旧/并行路径（已删除）

| 文件 | 职责 | 状态 |
|------|------|:----:|
| `skills/timeline_plan/canvas_matcher.py` | 多次 LLM run + shuffle + voting 匹配 | 已删除 |
| `skills/timeline_plan/edit_planner.py` | operation_type 判断 | 已删除 |
| `skills/timeline_plan/prompt_composer.py` | 分层 prompt 编辑 + 多 operation 支持 | 已删除 |
| `skills/timeline_plan/prompt_extractor.py` | 旧版 prompt 改写 | 已删除 |
| `skills/timeline_plan/duration_resolver.py` | 短组扩展策略 | 已删除 |

### 其他 Stage

| 文件 | 职责 |
|------|------|
| `skills/script-rewriting/rewrite_script.py` | CEFR 分级改写 |
| `skills/script-rewriting/SKILL.md` | 改写 skill 定义 |
| `skills/scene_detection/detect_scenes.py` | PySceneDetect 切点检测 |
| `skills/video_assembly/assemble.py` | 视频拼接 + Seedance 生成 |
| `skills/common/env.py` | 环境变量加载 |
| `skills/SKILL.md` | 管道总览 |

### 设计文档

| 文件 | 内容 |
|------|------|
| `docs/superpowers/specs/2026-05-29-asr-timeline-pipeline-design.md` | 原始 pipeline 设计 |
| `docs/superpowers/specs/2026-05-30-multimodal-canvas-edit-planner-design.md` | v3 多模态编辑规划设计（最完整的理想架构） |
| `docs/superpowers/specs/canvas-matching-context.md` | Episode 1 实际数据匹配分析 |
| `docs/superpowers/specs/stage3-detailed-design.md` | Stage 3 详细设计（旧版） |
| `docs/superpowers/specs/pipeline-v2-summary.md` | v2 验证结果 |
| `docs/superpowers/plans/2026-05-31-llm-first-timeline-planner.md` | v3 LLM-first 实施计划 |
| `docs/superpowers/plans/2026-05-31-pipeline-correctness-fixes.md` | 正确性修复计划 |
| `.omo/plans/three-skills-pipeline.md` | 三阶段 skill 管线计划 |

### 测试

| 文件 | 测试内容 |
|------|---------|
| `skills/timeline_plan/tests/test_llm_planner.py` | LLM planner |
| `skills/timeline_plan/tests/test_planner_verifier.py` | 5 层验证 |
| `skills/timeline_plan/tests/test_models.py` | 数据模型 |
| `skills/timeline_plan/tests/test_planner_models.py` | planner 模型 |
| `skills/timeline_plan/tests/test_timeline_normalizer.py` | 时间轴归一化 |
| `skills/timeline_plan/tests/test_cut_fusion.py` | 切点融合 |
| `skills/timeline_plan/tests/test_evidence_builder.py` | 证据构建 |
| `skills/timeline_plan/tests/test_validator.py` | 校验器 |
| `skills/video_assembly/tests/test_assemble.py` | 视频拼接 |
| `skills/scene_detection/tests/test_detect_scenes.py` | 场景检测 |

---

> 本文档基于 2026-05-31 项目代码、设计文档、运行产物综合分析完成。
