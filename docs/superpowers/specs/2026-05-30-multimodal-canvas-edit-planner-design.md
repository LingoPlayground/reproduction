# Stage 3 v3: Multimodal Canvas Edit Planner — 详细设计

> 基于 v2 实际运行经验的根本性重构。
> 将"匹配+替换"升级为"多模态编辑规划"。

## 1. 核心设计理念

### 1.1 当前 v2 系统的隐含模型（有缺陷）

```
Canvas Node Prompt = 文本容器，内含待替换台词
任务：找到台词 → 替换 → 验证包含改写文本
失败：完全降级到 bare prompt，丢失所有视觉质量
```

### 1.2 v3 的正确模型

```
Canvas Node Prompt = 分层生成意图
  └─ Global Style Layer （光照、分辨率、色彩、风格）
  └─ Character / Reference Layer （角色身份、肖像引用）
  └─ Scene / Camera Layer（分镜、景别、运镜、动作）
  └─ Dialogue / Action Patch Layer（台词 + 表情 + 反应）

任务：理解每段台词在 prompt 中的"语义视觉位置"
  └─ 有原文 → literal_replace
  └─ 无原文但有视觉段落 → semantic_insert
  └─ 无段落 → section_reconstruct
  └─ 无可用节点 → style_preserving_fallback

失败时：保留上层风格层，不全部丢失
```

### 1.3 系统架构定位：LLM 作语义规划器，代码作约束执行器

```
┌─────────────────────────────────────────┐
│  LLM / VLM (Multimodal Edit Planner)    │
│  ● 融合 ASR + keyframes + prompt        │
│  ● 输出结构化 EditPlan                  │
│  ● 处理 literal_replace / semantic_insert│
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│  Python (Constraint Executor)           │
│  ● Duration validation (≥4s ≤30s)       │
│  ● Schema validation                    │
│  ● Every rewritten line tracked         │
│  ● Retry routing / fallback             │
│  ● Timeline generation / overlap carving│
│  ● Degradation reporting                │
└─────────────────────────────────────────┘
```

---

## 2. 新 Pipeline 总览

```
Stage 3 v3 Pipeline:

ScriptOutput + RewriteJSON + CanvasNodes + SceneCuts + Keyframes
         │
         ▼
┌─────────────────────────────────────┐
│ Step 1: Evidence Pack 构造           │
│   每段候选改写区间构造多模态证据包    │
│   - ASR 台词 + 时间戳                │
│   - 邻接不改写台词                    │
│   - 相关 canvas node                 │
│   - node prompt                      │
│   - keyframes / scene cuts           │
│   - 角色参考图                       │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ Step 2: Multimodal Edit Planning    │
│   LLM 接收 Evidence Pack →          │
│   输出结构化 EditPlan               │
│   - matched_sections                │
│   - operation_type per line group   │
│   - match_evidence                  │
│   - coverage / duration strategy    │
│   - prompt_patch                    │
│   - risks / confidence              │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ Step 3: Duration Resolver           │
│   约束驱动:<4s 不丢弃, 选择最优策略  │
│   - pad after/before (差 ≤0.5s)    │
│   - snap to scene cut              │
│   - borrow unchanged neighbor line │
│   - hold/pause/reaction expansion  │
│   - cross-node merge (高风险)       │
│   - 所有 rewritten line 进入 plan   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ Step 4: Prompt Patch Composer       │
│   分层 prompt 编辑, 保留 style layer│
│   - 提取 global style layer         │
│   - 提取 character reference layer  │
│   - 确定 scene/camera layer         │
│   - 插入/替换 dialogue action p.    │
│   - final prompt                    │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ Step 5: Multilayer Validator + Retry│
│   L1: JSON schema / field presence  │
│   L2: ID & time validity            │
│   L3: revised_dialogue inclusion    │
│   L4: style preservation            │
│   L5: LLM-judge semantic consistency│
│   验证失败→retry(带错误)→fallback   │
└───────────────┬─────────────────────┘
                │
                ▼
┌─────────────────────────────────────┐
│ Step 6: TimelinePlan Finalization   │
│   overlap carving + gap fill + snap │
│   degradation reporting per item    │
└─────────────────────────────────────┘
```

---

## 3. 核心数据结构

### 3.1 Evidence Pack（构造 → 送入 LLM）

```python
@dataclass
class LineEvidence:
    line_id: str
    speaker: str
    original: str
    rewritten: str
    start_seconds: float
    end_seconds: float
    shot_number: int
    shot_scene: str
    rewrite_status: Literal["rewritten", "unchanged"]

@dataclass
class VideoEvidence:
    keyframes: List[Path]        # PySceneDetect 抽取的关键帧图像
    scene_cuts: List[float]      # 检测到的画面 cut 时间点
    video_path: Optional[str]       # 原视频路径（可选，用于高阶多模态）

@dataclass
class NodeSection:
    section_id: str              # 如 "镜头1" "Scene 1"
    description: str
    contains_quoted_dialogue: bool
    quoted_dialogue: List[str]   # 引号台词
    contains_implicit_dialogue_context: bool
    implicit_context: str

@dataclass
class CanvasNodeEvidence:
    node_id: str
    name: str
    full_prompt: str
    sections: List[NodeSection]
    reference_images: List[str]
    node_video_url: Optional[str]

@dataclass
class Constraints:
    min_seedance_duration: float = 4.0
    max_seedance_duration: float = 30.0
    must_preserve_rewritten_verbatim: bool = True
    max_extension_gap_sec: float = 5.0

@dataclass
class EvidencePack:
    target_lines: List[LineEvidence]
    neighbor_lines: List[LineEvidence]
    video: Optional[VideoEvidence]
    canvas_nodes: List[CanvasNodeEvidence]
    constraints: Constraints
```

### 3.2 Match Evidence（LLM 返回的匹配证据）

```python
@dataclass
class MatchEvidence:
    signal: Literal[
        "quoted_dialogue",
        "fuzzy_dialogue",
        "speaker_presence",
        "visual_action",
        "shot_scene_similarity",
        "temporal_order",
        "reference_image_match",
        "implicit_visual_scene"
    ]
    detail: str
    confidence: float               # 0.0–1.0
```

### 3.3 Coverage Plan（时间覆盖 + 扩展策略）

```python
@dataclass
class CoveragePlan:
    start_sec: float
    end_sec: float
    included_rewritten_line_ids: List[str]
    borrowed_original_line_ids: List[str]
    duration_strategy: Literal[
        "direct",                       # rewriting lines cover ≥4s directly
        "pad_after",                    # pad at end by Δ < 1s
        "pad_before",                   # pad at start by Δ < 1s
        "snap_to_cut",                   # snap to nearest scene cut
        "borrow_neighbor",              # borrow unchanged neighboring line
        "hold_reaction",                # hold character reaction / pause
        "merge_same_node_group",        # merge with same-node but distant group
        "cross_node_merge",             # merge with different node group (high risk)
        "forced_min_duration"           # last resort: force-extend
    ]
    duration_expansion_sec: float = 0.0
```

### 3.4 Prompt Patch Plan（分层 prompt 编辑）

```python
@dataclass
class DialoguePatch:
    line_id: str
    speaker: str
    mode: Literal["replace", "insert"]
    text: str
    placement: str              # LLM 描述插入位置

@dataclass
class PromptPatchPlan:
    operation_type: Literal[
        "literal_replace",          # original dialogue exists → replace
        "fuzzy_replace",            # similar dialogue exists → fuzzy replace
        "semantic_insert",          # no dialogue, but visual section matches
        "section_reconstruct",      # section is broken, rebuild from context
        "style_preserving_fallback",# keep style, generate scene + dialogue
        "full_fallback"             # no usable node evidence
    ]
    global_style: str               # 从 original prompt 提取的风格层
    local_visual_context: str        # 匹配到的 scene 描述
    dialogue_patches: List[DialoguePatch]
    discarded_sections: List[str]    # 被移除的 section 摘要
    final_prompt: str
```

### 3.5 EditPlan（Step 2 的完整输出）

```python
@dataclass
class EditPlan:
    group_id: str
    matched_node_id: Optional[str]
    matched_section_ids: List[str]
    match_confidence: float         # 0.0–1.0
    match_evidence: List[MatchEvidence]
    coverage: CoveragePlan
    prompt_patch: PromptPatchPlan
    degradation_level: int          # 0–6
    degradation_reason: str
    risks: List[str]
```

### 3.6 TimelinePlanItem（扩展后的模型）

```python
@dataclass
class TimelinePlanItem:
    shot_id: str
    shot_number: int
    source: Literal["seedance", "original"]
    start_sec: float
    end_sec: float
    scene_description: str
    ref_images: List[str]
    rewritten_prompt: str
    matched_node_id: Optional[str]
    match_confidence: float
    degradation_level: int
    degradation_reason: str
    # v3 新增字段:
    operation_type: Optional[str]   # 从 EditPlan.prompt_patch.operation_type
    duration_strategy: Optional[str]  # 从 CoveragePlan.duration_strategy
    covered_line_ids: List[str]
    borrowed_line_ids: List[str]
    source_node_ids: List[str]
    validation_report: Optional[dict]
```

---

## 4. 降级层级重定义（0–6）

当前 v2 的 degradation_level: 0/1/2 粒度太粗，且度量和语义不符。

v3 新定义:

| Level | 名称 | 含义 | 触发条件 |
|-------|------|------|---------|
| 0 | `direct_canvas_patch` | 最优路径 | 节点匹配明确 + prompt 视觉层保留 + 台词替换/插入成功 |
| 1 | `implicit_canvas_patch` | 隐式匹配但 prompt 保留 | 依赖隐式视觉/角色信号匹配，但 prompt 视觉层完好 |
| 2 | `style_preserving_fallback` | 局部重建 | Node 可用但局部 section 不可靠，style layer 保留 |
| 3 | `duration_padded_or_borrowed` | 时长扩展 | 为满足 4s 做了 padding / borrow / snap |
| 4 | `cross_node_composite` | 跨节点合成 | 合并了两个不同 canvas node 的内容 |
| 5 | `full_fallback` | 无可靠节点 | 无 node 证据，仅 scene_description 生成 |
| 6 | `original` | 放弃生成 | 改写行无法产出 → fallback 原始视频 |

Levels 可叠加（如 degradation_level=4 且含 padding），取最高 Level。

---

## 5. 五种 Prompt 操作类型详解

### 5.1 literal_replace

**适用场景**: Node prompt 中包含待改写台词的英文原文。
**操作**: 搜索原文位置，替换为改写文本。
**当前成功率**: 高（约 80%+ 当原文存在时）。
**改进点**: 增加 ASR drift 容错。

### 5.2 fuzzy_replace

**适用场景**: Node prompt 中有近似但非精确匹配的原文（ASR 轻差异、标点差异）。
**操作**: 语义近似匹配 + 替换。
**改进点**: 需要用 rewrite 行做语义搜索而非字符串搜索。

### 5.3 semantic_insert

**本质是解决当前 Episode 1 失败的核心操作类型。**

**适用场景**: Node prompt 中无待改写台词的英文文本，但有对应的中文视觉动作/表情描述。

**Episode 1 的判别信号**:
- prompt 包含 "真实的破防"、"面部特写"、"男主刚 Wink 完"等描述。
- 台词语境是 Donny 崩溃（no no no / they all refused me / are they going crazy）。
- 视觉语境与台词语境匹配。

**操作**: 保留视觉描述 + 在语义位置插入改写台词。
**验证**: 不要求"原文被找到并替换"，只要求 rewritten 台词在 final_prompt 中出现。

### 5.4 section_reconstruct

**适用场景**: Node prompt 中某 section 描述不完整或格式错误（如"男主是 "截断）。
**操作**: 从 scene_description 和原 section 上下文重建视觉描述。
**风险**: 可能偏离原始 prompt 风格。

### 5.5 style_preserving_fallback

**适用场景**: 前四种都无法可靠执行。
**操作**:
```
1. 从原始 prompt 中尽可能提取 style layer（全局视觉风格）
2. 从 rewrite_lines 重建设 scene + dialogue
3. 拼接: style_layer + scene_description + speaker says dialogues
```
**关键区别** 于当前 `_generate_prompt_from_scene`: 保留原始风格词。

### 5.6 full_fallback

**适用场景**: 无可用 node、无 style layer。
**操作**: 当前 L2 fallback 行为（保留以备极端情况）。

---

## 6. 匹配信号融合

### 6.1 匹配信号的权重和优先级

LLM matcher 现在为每个 line-group-to-node match 输出一个主信号及证据:

| 信号 | 典型置信度 | 适合场景 | Episode 1 命中 |
|------|-----------|---------|---------------|
| `quoted_dialogue` | 0.9–1.0 | 引号台词 | p001_l001-l002 ✅ |
| `fuzzy_dialogue` | 0.7–0.9 | ASR drift | 可能匹配部分 |
| `speaker_presence` | 0.5–0.8 | 同角色在同节点 | p001_l001-l005 (同 Donny) |
| `visual_action` | 0.6–0.85 | 动作/表情描述匹配台词 | p001_l003-l005 "破防" |
| `shot_scene_similarity` | 0.5–0.8 | 场景描述匹配 | Shot 1 "毕业庆祝" |
| `temporal_order` | 0.4–0.7 | 节点顺序与 line 时间顺序一致 | 间接辅助 |
| `implicit_visual_scene` | 0.5–0.85 | 中文视觉段落含隐式台词语境 | p001_l003-l005 ✅ |

### 6.2 置信度计算

当前: 交叉运行一致性 `most_common_node / total_runs = confidence`

v3 改为: `main_signal_confidence × 0.6 + cross_run_consistency × 0.4`

即：信号质量 + 运行稳定性 加权。

### 6.3 隐式匹配的追踪

当 `primary_signal` 为 `visual_action` / `implicit_visual_scene` / `speaker_presence` 时，`match_confidence` 天然较低。

这会被传播到 `degradation_level = 1`（implicit_canvas_patch），告知下游：匹配基于隐式信号，但 prompt 质量仍然可用。

---

## 7. Duration Resolver 详细设计

### 7.1 总原则

**绝对不要 silent drop rewritten lines。**

当前 `_split_contiguous` 中:

```python
if duration_sec < MIN_SEEDANCE_DURATION:
    continue     # ← 改写行静默消失
```

必须替换为显式策略选择器。

### 7.2 扩展策略优先级

对每个 <4s 的 group，按优先级尝试:

```
1. pad_after:   如果 duration ≥ 3.5s, 向后 pad 至 4s
2. pad_before:  如果 duration ≥ 3.5s, 向前 pad 至 4s
3. snap_to_cut: 附近有 scene cut (≤0.8s), 吸附到 cut
4. hold_reaction: LLM 判断该处有表情/停顿可延长
5. borrow_neighbor: 同 node 临近未改写台词(≤5s gap)
6. merge_same_node_group: 同 node 远处 group
7. cross_node_merge: 不同 node 的临近 group (degradation=4)
8. forced_min_duration: 以上全失败，强制 pad 到 4s
```

### 7.3 可量化判断

| 策略 | 判断条件 | expected_expansion |
|------|---------|-------------------|
| pad_after | 4.0 - duration < 0.5 | ≤0.5s |
| pad_before | 4.0 - duration < 0.5 | ≤0.5s |
| snap_to_cut | nearest cut ≤0.8s from boundary | ≤0.8s |
| hold_reaction | LLM/VLM 判断 | ≤1.5s |
| borrow_neighbor | gap ≤ 5s | ≥0 |
| merge_same_node | gap > 5s, same node | ≥0 |
| cross_node_merge | 无同 node 可用 | ≥0 |

### 7.4 扩展组和原组的 overlap 处理

如果扩展后 seedance 区间扩到原组目标之外:

- Timeline finalization 步骤会做 overlap carving。
- 被覆盖的 original 片段被切分为新生片段。
- 扩展部分不涉及台词改写，prompt 中保留原文或增加 "scene hold" 指令。

---

## 8. Prompt Composer 分层模型详解

### 8.1 输入: 原始 Canvas Node Prompt 的分层解析

```
原始 prompt:
"美式情景喜剧，真实短剧，柔光雾化，画面通透，8k，超高清，电影级布光...
  镜头 1：三层景深与霸总转身
    ..."This ceremony is boring."..."Let's see who wants me。...
  镜头 2：真实的破防（面部特写）
    画面与动作：面部特写，固定机位。男主刚 Wink 完...
  镜头3：眼部超大特写..."
```

解析后:

| Layer | 内容 | 来源 |
|-------|------|------|
| Global Style | 美式情景喜剧，真实短剧，柔光雾化，8k，超高清，电影级布光 | prompt 前缀 |
| Character Ref | {{Portrait 1}}, 男主 Donny | 角色参考图引用 |
| Scene 1 (镜头1) | 三层景深与霸总转身 + quoted dialogue | 第一个 section |
| Scene 2 (镜头2) | 真实的破防 (面部特写) + 固定机位 + 男主 Wink 完 | 第二个 section |
| Scene 3 (镜头3) | 眼部超大特写, 不置入台词 | 第三个 section |

### 8.2 输出: 编辑后的分层 prompt

对 `semantic_insert` (p001_l003-l005):

| Layer | 保留/编辑 | 操作 |
|-------|-----------|------|
| Global Style | ✅完全保留 | 不变 |
| Character Ref | ✅完全保留 | 不变 |
| Scene 1 | ❌移除(不包含改写台词) | remove |
| Scene 2 | ✅保留 + 编辑 | 插入 rewritten dialogue |
| Scene 3 | ✅保留 | 保持原状(延续情绪)|

Composer prompt 的具体指令:

```
## Input
Original Prompt: [full_prompt]
Operation: semantic_insert into Scene 2
Target Dialogue:
  Donny: "No, no, no, this can't be."
  Donny: "Every single one of them rejected me."
  Donny: "Have they gone completely crazy, Donny?"
Original Scene 2 context: 真实的破防（面部特写）, 固定机位, 男主刚 Wink 完, 表情崩塌

## Instructions
1. PRESERVE the Global Style section verbatim.
2. KEEP Scene 2's visual description as the base.
3. INSERT each rewritten dialogue line after the matching visual moment.
4. DISCARD Scene 1 (no rewritten dialogue).
5. KEEP Scene 3 as-is if it provides visual continuity to dialog-free moment.
6. Output: the complete rewritten prompt.
```

### 8.3 验证最终 prompt 时

对 `semantic_insert`, validation 验证:

- ✅ Global style 关键词保留至少 60% (style preservation check)
- ✅ 所有 rewritten dialogue 逐字出现 (exact substring check)
- ❌ 不需要 "original dialogue 被替换" check

---

## 9. Validator + Retry 系统

### 9.1 L1 — Schema Validation

```python
def validate_schema(plan: EditPlan) -> List[str]:
    """Check JSON structure, field presence, types."""
    errors = []
    for field in ["matched_node_id", "operation_type", "coverage", "prompt_patch"]:
        if getattr(plan, field) is None:
            errors.append(f"Missing required field: {field}")
    if plan.coverage.start_sec >= plan.coverage.end_sec:
        errors.append("Invalid coverage: start_sec >= end_sec")
    return errors
```

### 9.2 L2 — ID & Time Validation

```python
def validate_ids_and_time(plan: EditPlan, known_line_ids: Set[str]) -> List[str]:
    errors = []
    for lid in plan.coverage.included_rewritten_line_ids:
        if lid not in known_line_ids:
            errors.append(f"Unknown line_id in coverage: {lid}")
    duration = plan.coverage.end_sec - plan.coverage.start_sec
    if plan.prompt_patch.operation_type != "full_fallback":
        if duration < 4.0:
            errors.append(f"Duration {duration:.1f}s < min 4.0s")
        if duration > 30.0:
            errors.append(f"Duration {duration:.1f}s > max 30.0s")
    return errors
```

### 9.3 L3 — Dialogue Inclusion Validation

```python
def validate_dialogue_in_prompt(plan: EditPlan, rewrite_lines: Dict[str, str]) -> List[str]:
    errors = []
    for lid in plan.coverage.included_rewritten_line_ids:
        rewritten = rewrite_lines.get(lid, "")
        if rewritten and rewritten not in plan.prompt_patch.final_prompt:
            errors.append(f"Rewritten dialogue '{rewritten[:30]}...' not in final_prompt")
    return errors
```

### 9.4 L4 — Style Preservation Validation

```python
STYLE_ANCHORS_KEYWORDS = [
    "美式情景喜剧", "真实短剧", "柔光雾化", "电影级布光",
    "8k", "超高清", "画面通透"
]

def validate_style_preservation(plan: EditPlan, original_prompt: str) -> List[str]:
    if not plan.prompt_patch.global_style:
        return ["No global style extracted from original prompt"]
    present = sum(1 for word in STYLE_ANCHORS_KEYWORDS if word in plan.prompt_patch.final_prompt)
    threshold = max(1, int(len(STYLE_ANCHORS_KEYWORDS) * 0.6))
    if present < threshold:
        return [
            f"Style preservation too low: {present}/{len(STYLE_ANCHORS_KEYWORDS)}"
            f" keywords preserved (threshold: {threshold})"
        ]
    return []
```

### 9.5 Retry 策略

```
plan = llm_planner(evidence_pack)
errors = validate_all(plan, evidence_pack)

if errors:
  for attempt in range(MAX_RETRIES=2):
    plan = llm_planner(evidence_pack, validation_errors=errors)
    errors = validate_all(plan, evidence_pack)
    if not errors:
        break

if errors:
    # Graceful degradation:
    # 1. Try style_preserving_fallback
    plan = style_preserving_fallback(evidence_pack)
    # 2. If still fails → full_fallback
    if validate_all(plan, evidence_pack):
        plan = full_fallback(evidence_pack)
```

---

## 10. 关键文件改造清单

### 10.1 新增文件

| 文件 | 职责 | 规模 |
|------|------|------|
| `skills/timeline_plan/edit_planner.py` | Multimodal EditPlanner (LLM 编排 + EditPlan 生成) | ~300 行 |
| `skills/timeline_plan/evidence_builder.py` | Evidence Pack 构造器 | ~200 行 |
| `skills/timeline_plan/duration_resolver.py` | Duration Resolver (扩展策略选择器) | ~200 行 |
| `skills/timeline_plan/validator.py` | Multilayer Validator + Retry Manager | ~250 行 |

### 10.2 修改文件

| 文件 | 变更 | 重要性 |
|------|------|--------|
| `models.py` | 新增 7 个 dataclass + TimelinePlanItem 扩展 | 高 |
| `prompt_extractor.py` | 重构为 PromptPatchComposer (分层 prompt 编辑) | 高 |
| `canvas_matcher.py` | 扩展为 section-level matching + multi-signal output | 高 |
| `generate_plan.py` | 集成 planner pipeline + 替换 duration 处理逻辑 | 高 |
| `cut_fusion.py` | 基本不变，输出供给 evidence_builder | 低 |

### 10.3 废弃/简化

| 文件 | 说明 |
|------|------|
| `canvas-storyboard/match_to_canvas.py` | 旧版完整保留不动，但不再作为 pipeline 默认路径 |

---

## 11. 对 Episode 1 的完整回放

### 11.1 p001_l001-l002 (Group A, 3.6s)

**Step 1: Evidence Pack**
- target_lines: p001_l001-l002 (2.83–6.43s)
- neighbor_lines: p001_l003-l005 (17.47–29.55s, gap=11.04s)
- canvas_nodes: 13126e5a (prompt 含引号台词 "This ceremony is boring")
- constraints: min=4.0s

**Step 2: Edit Planner → EditPlan**

```json
{
  "matched_node_id": "13126e5a",
  "matched_section_ids": ["镜头1"],
  "operation_type": "literal_replace",
  "match_evidence": [
    {"signal": "quoted_dialogue", "detail": "Found '\"This ceremony is boring\"' in node prompt"}
  ],
  "coverage": {
    "start_sec": 2.83,
    "end_sec": 6.83,
    "duration_strategy": "pad_after",
    "duration_expansion_sec": 0.40
  },
  "prompt_patch": {
    "global_style": "美式情景喜剧，真实短剧，柔光雾化...",
    "local_visual_context": "镜头 1：三层景深与霸总转身...",
    "dialogue_patches": [
      {"mode": "replace", "line_id": "p001_l001", "text": "This ceremony is not entertaining at all."},
      {"mode": "replace", "line_id": "p001_l002", "text": "Let's see who is desperate to hire me."}
    ],
    "final_prompt": "美式情景喜剧...镜头1...Donny says: \"This ceremony is not entertaining at all.\"..."
  },
  "degradation_level": 3,
  "degradation_reason": "duration_padded_0.4s_to_meet_min_4s"
}
```

**结果**: 生成成功，而非当前被 silent drop。 ✅

### 11.2 p001_l003-l005 (Group B, 12.12s ✅)

**Step 2: Edit Planner → EditPlan**

```json
{
  "matched_node_id": "13126e5a",
  "matched_section_ids": ["镜头2", "镜头3"],
  "operation_type": "semantic_insert",
  "match_evidence": [
    {
      "signal": "implicit_visual_scene",
      "detail": "Node prompt section '镜头 2' describes Donny's emotional breakdown close-up which semantically matches p001_l003-l005 lines",
      "confidence": 0.85
    },
    {
      "signal": "temporal_order",
      "detail": "Same node already covers preceding Donny lines in same shot",
      "confidence": 0.60
    }
  ],
  "coverage": {
    "start_sec": 17.47,
    "end_sec": 29.55,
    "duration_strategy": "direct",
    "duration_expansion_sec": 0.0
  },
  "prompt_patch": {
    "operation_type": "semantic_insert",
    "global_style": "美式情景喜剧，真实短剧，柔光雾化，8k，超高清，电影级布光",
    "local_visual_context": "镜头 2：真实的破防（面部特写）画面与动作：面部特写，固定机位。男主刚 Wink 完后表情瞬间崩塌...",
    "dialogue_patches": [
      {
        "mode": "insert",
        "line_id": "p001_l003",
        "text": "No, no, no, this can't be.",
        "placement": "after the visual description of Donny's expression collapsing"
      },
      {
        "mode": "insert",
        "line_id": "p001_l004",
        "text": "Every single one of them rejected me.",
        "placement": "As Donny processes the realization"
      },
      {
        "mode": "insert",
        "line_id": "p001_l005",
        "text": "Have they gone completely crazy, Donny?",
        "placement": "After a beat of disbelief"
      }
    ],
    "final_prompt": "美式情景喜剧，真实短剧，柔光雾化，画面通透，8k，超高清，电影级布光。镜头 2：真实的破防（面部特写）画面与动作：面部特写，固定机位。男主刚 Wink 完后表情瞬间崩塌。台词: Donny says: \"No, no, no, this can't be.\" Donny says: \"Every single one of them rejected me.\" Donny says: \"Have they gone completely crazy, Donny?\" 镜头 3：眼部超大特写，延续 Donny 的崩溃情绪，眼睛睁大，瞳孔微动。"
  },
  "degradation_level": 1,
  "degradation_reason": "implicit_visual_scene_match_no_quoted_dialogue"
}
```

**结果**: 保留了 858chars 的原始视觉质量。而非当前 L2 fallback 丢失所有。 ✅

---

## 12. 实施阶段规划

### Phase 1: 基础设施 + 最小修复（高收益，中等改动）

目标: 先修复能修的，不改架构。

1. 扩展 `models.py` dataclasses（新数据结构）
2. 重构 `prompt_extractor.py` → PromptPatchComposer
   - 支持 `operation_type`
   - `semantic_insert` 操作模式
   - style layer 提取
   - 改进 fallback: 保留 style layer
3. Duration Resolver
   - 删除 silent drop
   - 增加 pad 策略
   - 增加 hold_reaction

### Phase 2: 匹配重构

4. 重构 `canvas_matcher.py` → section-level matching
5. Multi-signal output
6. LLM prompt 升级

### Phase 3: 验证 + 多模态增强

7. `validator.py` Multilayer Validator
8. Retry manager
9. LLM-judge semantic check
10. 视频关键帧 / 音频证据集成

---

## 13. 未解决问题 / 开放风险

1. **效果评估**: 如何定量评估 prompt rewrite 质量？当前只有 binary "pass/fail"。
2. **成本**: 增加 LLM 调用次数和证据包大小会增加 API 开销。
3. **Latency**: 更复杂的 pipeline = 更长的生成时间。
4. **跨节点合并的 prompt 质量**: 不同 node prompt 的风格可能冲突。
5. **Phase 4 多模态证据**: 传视频/音频帧给 LLM/VLM 的效果需要验证。
6. **Legacy 数据兼容**: 已有 timeline_plan.json 输出格式变化需要向后兼容或迁移。

---

## A. 附录: 与 v2 的对比

| 维度 | v2 | v3 |
|------|-----|-----|
| 匹配信号 | 仅引号台词 | 多信号融合 |
| Prompt 操作 | replace / fallback | 6 种操作类型 |
| 短组处理 | silent drop | 6 级策略扩张 |
| Fallback | bare prompt | style-preserving fallback |
| Validation | 单层 substring | 5 层验证 + retry |
| Degradation | 0/1/2 粒度粗 | 0–6 可追踪 |
| 多模态 | 无 | keyframes + ASR + 可选音频/视频 |
| 是否丢弃改写行 | 是（silent） | 否（显式记录状态） |
| Prompt 模型 | 文本容器 | 分层生成意图 |
