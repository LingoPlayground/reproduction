# Stage 3 v4: Segment-First Edit Atom Pipeline — 详细设计

> 2026-06-01  
> 对应 GitHub issue: https://github.com/LingoPlayground/reproduction/issues/5

## 1. 结论摘要

Stage 3 v4 的核心目标不是做一个复杂的多源边界融合器，而是把当前 v3 的“逐行匹配 → 按 node 分组 → normalizer 挖洞”重构为：

```text
Stage 1 scene/shot + ASR line timing
        │
        ▼
Edit Atom Builder        语义切分：台词不拆断，场景尽量不跨越
        │
        ▼
Segment Matcher          Edit Atom → Canvas Node Prompt
        │
        ▼
Generation Window Resolver
                          把短 atom 组合/扩展为可执行的 >=4s 视频生成窗口
        │
        ▼
Window Prompt Rewriter   基于 window 内全部改写台词，一次性重写 prompt
        │
        ▼
Plan Finalizer / Validator
                          输出 Stage 4 可直接执行的 TimelinePlan
```

关键概念：

- **Edit Atom**: 最小可编辑语义单元。可以是一句台词，也可以是同一细分场景内的连续几句台词。它用于匹配原始 canvas node prompt。
- **Generation Window**: Stage 4 真正执行视频生成/替换的时间窗口。它可以包含一个或多个 Edit Atom，必须满足生成模型的执行约束，例如 modified item >= 4s。

这两个概念必须分开。Edit Atom 追求 prompt matching 粒度正确；Generation Window 追求视频生成可执行。

---

## 2. 设计目标

### 2.1 目标

1. **用语义粒度替代逐行粒度**
   - Canvas node prompt 通常描述一个局部场景，包含人物、动作、环境、镜头和台词。
   - 单行台词缺少足够视觉上下文，容易误匹配。
   - Edit Atom 应携带“台词 + 场景描述 + 邻接上下文”，作为匹配单元。

2. **让 LLM 少猜时间**
   - LLM 负责语义匹配和 prompt 改写。
   - 时间范围由 ASR line timing、Stage 1 shot、scene cuts 辅助吸附，以及 deterministic code 决定。

3. **删除 v3 的语义修补式 normalizer**
   - v3 `normalize_plan()` 混合了承接 LLM draft、unmatched fallback、duration padding 和最终几何整理。
   - v4 将 unmatched / duration / prompt 等语义补救前移到 matcher、window resolver 和 prompt rewriter。
   - 最终仍保留纯几何的 carve/fill/sort/final validation；这部分可以复用 v3 `_carve_out()` 思路。

4. **Stage 4 输入格式尽量不变**
   - 最终仍输出 `TimelinePlan.items[]`。
   - `source="modified"` 的 item 仍携带 `rewritten_prompt`、`ref_images`、`matched_node_id`、`covered_line_ids`。
   - `ref_images` 只来自 matched canvas node，不挂在 Edit Atom 上。
   - Stage 4 只需要非常小的适配，最好不需要改主流程。

5. **废弃 keyframes**
   - Stage 4 的参考图应来自 canvas node 的 reference images。
   - 原视频抽帧不可控，清晰度、构图、人物稳定性不保证，不应进入生成参考图链路。

### 2.2 非目标

1. 不设计复杂 boundary voting / fusion engine。
2. 不用 scene cuts 主动制造大量 edit atom。
3. 不让 LLM 输出 source time range。
4. 不在 Stage 3 解决“只改台词、不改环境”的强确定性验证问题；这属于 prompt rewrite validator 的独立增强。

---

## 3. 当前实现问题

当前 v3 代码路径：

```text
generate_plan.py
  build_evidence()
  generate_plan_draft()
    A1 coarse: line -> candidate nodes
    A2 fine: line -> best node
    deterministic grouping by node/time
    per-group prompt rewrite
  normalize_plan()
    pad short modified groups
    carve modified ranges out of original shots
    fill gaps
    validate coverage
```

主要问题：

1. **逐行匹配粒度偏小**
   - `_make_coarse_prompt()` 和 `_make_fine_prompt()` 都以单行 dialogue 为核心输入。
   - 视觉场景、角色、动作信息只是附加字段，不是匹配主体。

2. **grouping 是事后修补**
   - `_group_by_node()` 根据 node_id + 时间间隔把 line matches 合并。
   - 如果前面的 line matching 错了，grouping 只能扩大错误。

3. **normalizer 责任混合**
   - `normalize_plan()` 中的 `_finalize()` 本质是纯几何整理：pad、carve originals、填 gap、排序。
   - 真正需要移除的是 normalizer 内的语义补丁，例如 unmatched fallback 和 duration padded 语义标记。
   - v4 应把语义补丁前移：匹配失败由 matcher/window resolver 决定，短时长由 window resolver 解决；finalizer 只做几何和 validation。

4. **scene cuts 进入太晚**
   - 当前 scene cuts 只通过 `cut_fusion.determine_cut_points()` 修正 shot 边界。
   - Stage 1 多模态 LLM 生成 shots 时没有看到 scene cut reference。

---

## 4. Stage 1 / Stage 1b 设计

### 4.1 Stage 1b: 只输出 CutPoint

文件：`skills/scene_detection/detect_scenes.py`

保留：

```python
@dataclass
class CutPoint:
    time_sec: float

def detect_scene_boundaries(video_path: str, threshold: float = 20.0) -> list[CutPoint]:
    ...
```

删除：

- `KeyFrame`
- `extract_keyframes()`
- 与 keyframes 相关测试和文档入口

说明：

- 不再产出 `confidence`。
- 不再把原视频抽帧作为 reference image。
- `detect_node_internal_cuts()` 可保留为 `detect_scene_boundaries()` 的别名；如果当前没有调用方，也可以一并删除。

### 4.2 Stage 1: 给 lingolens 注入 scene cut reference

本仓库入口：`skills/script-extraction/extract_script.py`  
依赖仓库：`LingoPlayground/lingolens`

需要在 lingolens 修改：

```python
class VideoScriptExtractor:
    async def extract(
        self,
        video_path: str,
        utterances: list[dict],
        duration_seconds: float,
        temp_dir: str = "runs",
        scene_cut_times: list[float] | None = None,
    ) -> VideoScriptOutput:
        prompt = build_multimodal_prompt(
            utterances=utterances,
            duration_seconds=duration_seconds,
            scene_cut_times=scene_cut_times or [],
        )
```

`build_multimodal_prompt()` 增加一段轻量上下文：

```text
# SCENE CUT REFERENCE
The following timestamps are visual cut points detected by software.
Use them as reference points when choosing natural shot boundaries.
Do not create a shot boundary only because a timestamp is listed here;
prefer boundaries that also match dialogue, character focus, and visual scene changes.

Cut points: 2.13s, 6.84s, 11.20s, ...
```

注意：

- scene cuts 是 reference，不是硬约束。
- Stage 1 LLM 仍以视频内容和 ASR utterances 为主。
- 目标是让 `ScriptShot.start_seconds/end_seconds` 更贴近真实视觉切换点。

本仓库 `extract_script.py` 调用：

```python
scene_cuts = detect_scene_boundaries(video_path)
result = await extractor.extract(
    video_path=video_path,
    utterances=utterances,
    duration_seconds=duration,
    temp_dir=temp_dir,
    scene_cut_times=[c.time_sec for c in scene_cuts],
)
```

---

## 5. Stage 3 v4 模块结构

建议文件：

```text
skills/timeline_plan/
  models.py                    # 增加 EditAtom / AtomLine / GenerationWindow
  edit_atom_builder.py          # Stage 1 shot + rewrite lines -> edit atoms
  segment_matcher.py            # edit atom -> canvas node
  generation_window_resolver.py # atom -> >=4s generation windows
  prompt_rewriter.py            # per generation window prompt rewrite
  plan_finalizer.py             # build TimelinePlan + validate
  generate_plan.py              # 新编排
```

可以废弃或大幅简化：

- `evidence_builder.py`
- `llm_planner.py`
- `timeline_normalizer.py`
- `planner_models.py`
- `planner_verifier.py`
- `cut_fusion.py`

`cut_fusion.py` 中“shot boundary snap to nearest cut”的能力可以搬到 `edit_atom_builder.py` 的边界吸附函数中。

---

## 6. 数据模型

### 6.1 AtomLine

```python
@dataclass
class AtomLine:
    line_id: str
    speaker: str
    original: str
    rewritten: str
    start_sec: float
    end_sec: float
    shot_scene: str = ""

    @property
    def is_rewritten(self) -> bool:
        return normalize_text(self.original) != normalize_text(self.rewritten)
```

### 6.2 EditAtom

```python
@dataclass
class EditAtom:
    atom_id: str
    shot_numbers: list[int]
    primary_shot_number: int
    start_sec: float
    end_sec: float
    scene_description: str
    lines: list[AtomLine]

    # Matching result
    matched_node_id: str | None = None
    match_confidence: float | None = None
    match_reasoning: str = ""

    # Debug metadata
    boundary_reason: str = ""
    source_cut_times: list[float] = field(default_factory=list)

    @property
    def rewritten_lines(self) -> list[AtomLine]:
        return [line for line in self.lines if line.is_rewritten]

    @property
    def has_rewritten_lines(self) -> bool:
        return bool(self.rewritten_lines)
```

语义：

- Edit Atom 是 prompt matching 单元。
- Edit Atom 可以短于 4s。
- Edit Atom 不一定直接变成 `TimelinePlanItem`。
- Edit Atom 不持有 `ref_images`，因为参考图属于 canvas node / generation call。
- Edit Atom 不持有最终 `rewritten_prompt`，因为 prompt rewrite 发生在 Generation Window 层。
- v4 初期只需要对 `has_rewritten_lines=True` 的 atom 做 canvas matching。

### 6.3 GenerationWindow

```python
@dataclass
class GenerationWindow:
    window_id: str
    start_sec: float
    end_sec: float
    atoms: list[EditAtom]

    matched_node_id: str | None = None
    match_confidence: float | None = None
    rewritten_prompt: str | None = None
    ref_images: list[str] = field(default_factory=list)

    degradation_level: int = 0
    degradation_reason: str = ""

    @property
    def covered_line_ids(self) -> list[str]:
        ids = []
        for atom in self.atoms:
            ids.extend(line.line_id for line in atom.rewritten_lines)
        return sorted(set(ids))

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)
```

语义：

- Generation Window 是 Stage 4 执行单元。
- `source="modified"` 的 TimelinePlanItem 由 GenerationWindow 生成。
- Window 必须满足 `duration_sec >= MIN_MODIFIED_DURATION`，当前为 4s。

---

## 7. Edit Atom Builder

文件：`edit_atom_builder.py`

### 7.1 输入

```python
def build_edit_atoms(
    script_shots: list[Any],
    rewrite_lines: list[dict],
    scene_cuts: list[CutPoint],
    video_duration: float,
) -> list[EditAtom]:
    ...
```

### 7.2 设计原则

1. **ASR line 不可拆**
   - atom 边界不能落在 line 的 `[start_sec, end_sec]` 内。
   - 如果吸附后的 cut 落在台词内部，放弃该 cut 或吸附到最近 line 边界。

2. **Stage 1 shot 是主要语义容器**
   - 默认不跨 shot 构建 atom。
   - shot 的 `scene_description` 是 atom 匹配 canvas node 的核心上下文。

3. **scene cuts 只做边界吸附，不主动制造 atom**
   - 如果 shot boundary 附近有 cut，则把 shot boundary 吸附到 cut。
   - 不因为 shot 内存在 scene cut 就强行拆 atom，除非 Stage 1 已经把它描述为不同 shot/scene。

4. **ASR gap 只做弱提示**
   - 不把 ASR gap 设计成硬规则。
   - 初版可以不使用 ASR gap 拆分，只在 debug metadata 中记录。

5. **优先为改写行创建 atom**
   - unchanged lines 主要作为 context line 或 generation window 扩展材料。
   - 不需要为每个 unchanged line 都做 canvas matching。

### 7.3 初版 atom 切分算法

推荐初版保持简单：

```text
1. build candidate clusters inside each Stage 1 shot
2. optionally merge adjacent clusters that are semantically continuous
3. optionally merge adjacent-shot clusters when they are clearly the same local scene
4. emit EditAtom for each final cluster
```

#### 7.3.1 基础 cluster

对每个 Stage 1 shot：

```text
lines = rewrite lines belonging to this shot, sorted by start_sec
scan lines in order:
  start a cluster when hitting a rewritten line
  keep adding following rewritten lines while they are adjacent and semantically compatible
  unchanged lines between two rewritten lines may become bridge context, but do not automatically merge clusters
```

Atom 的核心 lines 是 rewritten lines；短的 unchanged bridge line 可以被记录为上下文，但不计入 `covered_line_ids`。

#### 7.3.2 “contiguous”的严格定义

两条 rewritten line 可以进入同一个 cluster，当且仅当：

- 两条 rewritten line 属于同一 shot；
- `next.start_sec - prev.end_sec <= cluster_gap_sec`，建议初始值 `1.5s`；
- 中间没有明显 scene/action break；
- 两条 line 的 speaker / 对话轮次 / 场景描述仍属于同一个局部 prompt intent；
- 如果中间夹着 unchanged line，该 unchanged line 只能作为 bridge context，不能单独决定合并。

#### 7.3.3 变更行和未变更行交替

典型情况：

```text
line 1: rewritten
line 2: unchanged
line 3: rewritten
```

不要只因为 `line1.end -> line3.start <= 1.5s` 就合并。

允许合并的条件：

- line 2 很短，且像同一对话轮次中的承接句、反应、停顿或轻微动作；
- line 1 / line 3 的 speaker 关系连续，例如同一角色连续说话，或两人围绕同一语义动作对话；
- Stage 1 shot scene_description 没有明显动作转折；
- line 2 不包含新的场景设定、明显镜头动作或话题转移。

必须拆分的条件：

- speaker 切换伴随新的动作 beat，例如一个人离开、另一个人转向新对象；
- unchanged line 是完整的新语义节点，例如新的问题、独立反应或旁白；
- 中间存在明显停顿、scene cut、shot boundary，或者 Stage 1 描述显示视觉焦点变化；
- 合并后 atom 的 dialogue/context 已经不再像一个 canvas node prompt 的局部片段。

#### 7.3.4 相邻 shot 的“同类相吸”合并

默认 atom 不跨 shot。但有一类情况应允许跨 shot：Stage 1 把同一个 canvas node 的局部场景切成了相邻 shot。

相邻 shot clusters 可以合并为一个跨 shot atom，当满足：

- 两个 shot 的 location 相同，或 scene_description 文本高度相似；
- 两个 cluster 的时间 gap 小，例如 `<= 1.0s`；
- shot boundary 附近没有强视觉断裂，或 scene cut 只是同一场景内的镜头切换；
- speakers / characters 连续；
- 合并后的 atom 时长仍在合理范围内，例如 `< 8s`；
- 合并后更像一个 canvas node prompt 中的同一段场景描述。

如果实现上没有可靠 embedding，可先用轻量规则：

```text
same location name
OR normalized scene_description token overlap >= 0.45
OR adjacent shots share >= 1 speaker and gap <= 0.7s
```

这条规则的目的不是制造大段 segment，而是避免同一个 prompt intent 被 shot boundary 切成两个 atom，导致后续匹配和重写不一致。

### 7.4 边界吸附

Atom 原始边界来自 ASR line timing：

```text
raw_start = first_rewritten_line.start_sec
raw_end = last_rewritten_line.end_sec
```

仅在以下情况下吸附：

- raw_start 距离 scene cut <= 0.5s，且吸附后不会切断任何 line；
- raw_end 距离 scene cut <= 0.5s，且吸附后不会切断任何 line；
- shot start/end 距离 scene cut <= 0.5s，可优先使用 shot boundary 的吸附结果。

吸附失败时保持 ASR timing。

### 7.5 输出示例

```json
{
  "atom_id": "atom_003",
  "shot_numbers": [4],
  "primary_shot_number": 4,
  "start_sec": 12.42,
  "end_sec": 14.10,
  "scene_description": "Mia stands beside the kitchen counter...",
  "lines": [
    {
      "line_id": "p004_l002",
      "speaker": "Mia",
      "original": "I can't believe you did that.",
      "rewritten": "I really can't believe you did that.",
      "start_sec": 12.42,
      "end_sec": 14.10
    }
  ],
  "boundary_reason": "asr_line_range"
}
```

---

## 8. Segment Matcher

文件：`segment_matcher.py`

### 8.1 目标

把每个 `EditAtom` 匹配到最合适的 canvas node prompt。

这里匹配的不是单行文本，而是：

```text
atom dialogue + speaker + scene_description + local timing order
  -> canvas node prompt
```

### 8.2 LLM 输入

```json
{
  "atoms": [
    {
      "atom_id": "atom_003",
      "shot_numbers": [4],
      "primary_shot_number": 4,
      "scene": "Mia stands beside the kitchen counter...",
      "dialogue": [
        {
          "line_id": "p004_l002",
          "speaker": "Mia",
          "original": "I can't believe you did that.",
          "rewritten": "I really can't believe you did that."
        }
      ]
    }
  ],
  "canvas_nodes": [
    {
      "node_id": "abc",
      "prompt": "..."
    }
  ]
}
```

### 8.3 Prompt 重点

LLM 指令应强调：

- Canvas node prompt 是原始视频生成意图。
- 先理解 node prompt 中的场景、角色、动作、实际 spoken dialogue。
- 对每个 atom，选择最匹配“台词 + 场景”的 node。
- 不要只按单个关键词匹配。
- 如果台词没有逐字出现，可以用语义相似、角色动作和场景描述匹配。
- 如果没有合理 node，返回 unmatched。

### 8.4 输出 schema

```json
{
  "matches": [
    {
      "atom_id": "atom_003",
      "node_id": "abc",
      "confidence": 0.86,
      "reasoning": "The node describes Mia in the kitchen and contains the same accusation line."
    }
  ],
  "unmatched": [
    {
      "atom_id": "atom_007",
      "reason": "No canvas prompt matches the classroom scene."
    }
  ]
}
```

### 8.5 单 node 约束

v4 初版采用：

```text
EditAtom -> 0 or 1 CanvasNode
```

原因：

- Edit Atom 本身足够短。
- 它表达一个局部 scene/prompt matching intent。
- 如果一个 atom 需要多个 node，通常说明 atom 切得太粗，应回到 atom builder 细分。

---

### 8.6 可扩展性策略

初版可以一次 LLM 调用发送所有 target atoms 和所有 canvas nodes。典型输入规模：

```text
5-15 分钟短剧:
  target atoms: 20-40
  canvas nodes: 30-60
```

这个规模通常可接受。但生产规模增大时，一次性全量输入会变大，例如 50 atoms × 100 nodes。

建议保留两级策略：

1. **Coarse candidate retrieval**
   - 低成本文本检索或小 LLM 调用。
   - 对每个 atom 找 Top-K candidate nodes，例如 K=8。
   - 信号包括 dialogue fuzzy match、speaker name、scene token overlap、node prompt quoted text。

2. **Global fine matching**
   - 只发送 atoms + 去重后的 candidate nodes。
   - LLM 在较小上下文里做全局精配。
   - 保留全局视角，避免 per-atom 独立匹配造成局部最优。

这和当前 v3 的 A1/A2 思路兼容，但粒度从 line 升级为 atom。

---

## 9. Generation Window Resolver

文件：`generation_window_resolver.py`

### 9.1 为什么需要它

Edit Atom 可以短于 4s，但 Stage 4 的 modified item 必须满足生成模型约束。

因此需要：

```text
Edit Atom(s) -> Generation Window(s)
```

### 9.2 输入

```python
def resolve_generation_windows(
    atoms: list[EditAtom],
    all_lines: list[AtomLine],
    canvas_nodes: list[CanvasNode],
    video_duration: float,
    min_duration_sec: float = 4.0,
    max_duration_sec: float = 30.0,
) -> list[GenerationWindow]:
    ...
```

### 9.3 生成窗口原则

1. **优先保持 atom 原边界**
   - 如果 atom duration >= 4s，直接生成 window。

2. **短 atom 优先合并同 node、同 shot、相邻 atom**
   - 如果相邻 rewritten atom 匹配同一 node，且两者之间 gap 很小，可合并为一个 window。

3. **不足 4s 时扩展到邻接原视频内容**
   - 扩展窗口时间，但 `covered_line_ids` 只包含 rewritten lines。
   - 扩展内容可以是 unchanged dialogue、反应、停顿、动作，不需要重写。

4. **短 atom 不因 duration 单独 fallback**
   - 如果已经有 matched node 和可重写 prompt，短 atom 应通过扩展 execution window 达到 4s。
   - 这里的“clamp 到 4s”指扩展原视频替换窗口，不是改变 atom 本身，也不是加黑帧/冻帧。
   - fallback 只应发生在无法匹配 node、prompt rewrite 失败、生成失败，或视频边界实在无法形成 4s window 的情况下。

5. **不跨明显 Stage 1 scene/shot，除非别无选择**
   - 默认 window 不跨 shot。
   - 如果单个 shot 内无法达到 4s，可以在相邻 shot 边界内扩展，但需记录 degradation reason。

6. **不切断台词**
   - 扩展边界不能落在任意 ASR line 内部。

7. **不得超过 max_duration_sec**
   - 默认 30s。

### 9.4 短 atom 扩展算法

推荐 deterministic 初版：

```text
for each matched atom:
  window_start = atom.start_sec
  window_end = atom.end_sec

  if duration >= 4s:
    emit window

  else:
    candidate range = current Stage 1 shot boundary
    expand alternately left/right to nearest safe line boundary
    prefer side with:
      1. same shot
      2. no rewritten atom from another matched node
      3. smaller expansion distance
      4. scene cut / shot boundary alignment

    stop when duration >= 4s

  if still < 4s:
    try merge with adjacent atom/window if same matched_node_id

  if still < 4s:
    clamp to nearest safe 4s range within video bounds

  if still impossible:
    emit unresolved window with degradation_reason
```

注意：

- 这个 resolver 是执行窗口解析，不是 prompt matching segmentation。
- 扩展出来的 unchanged 内容只影响视频替换长度，不改变 atom 的语义匹配。
- Window 可以包含 unchanged 时间范围，但 `covered_line_ids` 只记录 rewritten lines。

### 9.5 Prompt 与 window 的关系

Prompt rewrite 发生在 window resolution 之后。

如果一个 window 只包含一个 atom：

- window prompt 基于该 atom 的 matched node prompt 重写。

如果一个 window 包含多个 atom：

- 如果 atoms 匹配同一 node：使用同一个 node prompt，一次性替换多个 atom 的台词；
- 如果 atoms 匹配不同 node：初版不自动合并，优先拆成多个 window；若不足 4s，选择扩展/clamp 或 degradation。

### 9.6 ref_images 数据流

`ref_images` 只属于 canvas node 和 generation window，不属于 edit atom。

```text
CanvasNode.reference_images
        │
        ▼
GenerationWindow.ref_images
        │
        ▼
TimelinePlanItem.ref_images
        │
        ▼
Stage 4 seedance generation
```

规则：

- Atom 只保存 `matched_node_id`。
- Window 根据 `matched_node_id` 从 CanvasNode 读取 reference images。
- 如果 window 包含多个 atoms，初版要求它们匹配同一 node；因此 ref_images 来自该 node。
- 如果 node 没有 reference images，window 标记 warning/degradation，但不从 keyframes 补。

---

## 10. Window Prompt Rewriter

文件：`prompt_rewriter.py`

### 10.1 输入

```python
def rewrite_prompt_for_window(
    window: GenerationWindow,
    node: CanvasNode,
    level: str,
) -> str:
    ...
```

### 10.2 规则

1. 保留原 node prompt 的视觉风格、角色、镜头、环境。
2. 一次性替换 window 内所有 atoms 的 rewritten lines。
3. 如果原文台词没有逐字出现在 prompt 中，允许语义替换/插入，但不能重写整个环境。
4. 输出纯 prompt text，不输出 JSON。
5. Prompt rewrite 必须发生在 window resolution 之后，避免多个 atom 分别重写后又被合并造成不一致。

### 10.3 验证

Prompt rewrite 后至少做：

- `rewritten` 文本包含检查；
- style anchors 保留检查；
- non-empty 检查；
- 可选 LLM judge。

如果失败：

- retry 2-3 次；
- 仍失败则 window 标记 `degradation_level=prompt_rewrite_failed`，由 finalizer/report 决定 fallback。

---

## 11. Plan Finalizer / Validator

文件：`plan_finalizer.py`

### 11.1 职责

它保留 v3 `_finalize()` 中正确的纯几何职责，但不再承担语义修补。也就是说，v4 finalizer 可以复用 carve/fill/sort 的思想，区别在于输入已经是 resolver 产出的可执行 windows。

它只做：

1. 把 GenerationWindow 转成 modified `TimelinePlanItem`。
2. 用 modified windows 从完整时间轴中切出 original items。
3. 确保无 overlap。
4. 确保覆盖 `[0, video_duration]`。
5. 确保所有 rewritten lines 被 covered 或有明确 fallback report。
6. 确保 modified item 有 prompt、时间合法、ref image 状态明确。

### 11.2 输出规则

```python
for window in windows:
    items.append(TimelinePlanItem(
        shot_id=window.window_id,
        shot_number=window.atoms[0].primary_shot_number,
        source="modified",
        start_sec=window.start_sec,
        end_sec=window.end_sec,
        scene_description=window.atoms[0].scene_description,
        ref_images=window.ref_images,
        rewritten_prompt=window.rewritten_prompt,
        matched_node_id=window.matched_node_id,
        match_confidence=window.match_confidence,
        original_duration=window.duration_sec,
        covered_line_ids=window.covered_line_ids,
        degradation_level=window.degradation_level,
        degradation_reason=window.degradation_reason,
    ))
```

然后从 `[0, video_duration]` 中 carve modified windows，生成 original items。

注意：这里的 carve 与 v3 `_carve_out()` 在几何上是同类操作。真正移除的是 v3 `normalize_plan()` 中的 unmatched fallback、duration padding 等语义补丁；这些职责在 v4 已前移到 matcher / resolver / prompt rewriter。

### 11.3 Validation

阻塞错误：

- modified item duration < 4s；
- modified item 缺 prompt；
- modified item `start_sec >= end_sec`；
- timeline overlap；
- timeline 开头/结尾 gap；
- rewritten line 未 covered 且没有 fallback reason；
- 同一 rewritten line 被多个不同 node/window 覆盖。

非阻塞 warning：

- modified item 无 ref images，Stage 4 可能 fallback；
- atom/window 跨 shot；
- match confidence 低于阈值，例如 `<0.5`；
- prompt style anchors 保留率低。

---

## 12. generate_plan.py 新编排

```python
def generate_timeline_plan(input_data: Stage3Input) -> TimelinePlan:
    script_output = input_data.script_output
    shots = list(script_output.script.shots)
    rewrite_lines = input_data.rewrite_json.get("lines", [])
    canvas_nodes = input_data.canvas_nodes
    scene_cuts = input_data.video_cut_points
    video_duration = resolve_video_duration(shots, rewrite_lines)

    atoms = build_edit_atoms(
        script_shots=shots,
        rewrite_lines=rewrite_lines,
        scene_cuts=scene_cuts,
        video_duration=video_duration,
    )

    target_atoms = [a for a in atoms if a.has_rewritten_lines]
    if not target_atoms:
        return build_all_original_plan(shots, scene_cuts, video_duration)

    match_atoms_to_nodes(target_atoms, canvas_nodes)

    windows = resolve_generation_windows(
        atoms=target_atoms,
        all_lines=collect_all_lines(rewrite_lines),
        canvas_nodes=canvas_nodes,
        video_duration=video_duration,
    )

    rewrite_prompts_for_windows(windows, canvas_nodes)

    return finalize_timeline_plan(
        windows=windows,
        shots=shots,
        video_duration=video_duration,
        title=title,
        level=level,
    )
```

---

## 13. 文件变更建议

| 操作 | 文件 | 说明 |
|------|------|------|
| MODIFY | `skills/scene_detection/detect_scenes.py` | 删除 keyframes；CutPoint 只保留 `time_sec` |
| MODIFY | `skills/scene_detection/tests/test_detect_scenes.py` | 删除 keyframe tests |
| MODIFY | `skills/script-extraction/extract_script.py` | 先跑 scene detection，再把 cut times 传给 lingolens |
| NEW | `skills/timeline_plan/edit_atom_builder.py` | 构建 Edit Atom |
| NEW | `skills/timeline_plan/segment_matcher.py` | Atom -> Canvas Node |
| NEW | `skills/timeline_plan/prompt_rewriter.py` | Window-level prompt rewrite |
| NEW | `skills/timeline_plan/generation_window_resolver.py` | Atom -> >=4s executable windows |
| NEW | `skills/timeline_plan/plan_finalizer.py` | TimelinePlan 组装和 validation |
| MODIFY | `skills/timeline_plan/models.py` | 增加 AtomLine/EditAtom/GenerationWindow；调整 CutPoint |
| MODIFY | `skills/timeline_plan/generate_plan.py` | v4 编排 |
| DEPRECATE | `skills/timeline_plan/evidence_builder.py` | 被 atom builder + matcher 输入替代 |
| DEPRECATE | `skills/timeline_plan/llm_planner.py` | 被 segment matcher + prompt rewriter 替代 |
| DEPRECATE | `skills/timeline_plan/timeline_normalizer.py` | 被 generation resolver + plan finalizer 替代 |
| DEPRECATE | `skills/timeline_plan/cut_fusion.py` | 边界吸附逻辑迁入 atom builder |
| KEEP | `skills/video_assembly/assemble.py` | 尽量保持不变 |
| KEEP | `skills/script-rewriting/rewrite_script.py` | 不变 |
| KEEP | `skills/timeline_plan/fetch_canvas.py` | 不变 |

lingolens 变更：

| 操作 | 文件 | 说明 |
|------|------|------|
| MODIFY | `backend/agents/script_extraction/extractor.py` | `extract()` 增加 `scene_cut_times` |
| MODIFY | `backend/agents/script_extraction/prompt.py` | prompt 增加 scene cut reference |
| MODIFY | 对应 tests | 验证 prompt 包含 cut reference 且兼容空列表 |

---

## 14. 迁移计划

### Phase 1: Stage 1b 简化 + lingolens Stage 1 增强

目标：

- `detect_scene_boundaries()` 只产出 cut times。
- `extract_script.py` 能把 cut times 传给 lingolens。
- lingolens script extraction prompt 接收 cut reference。

验证：

- 本仓库 scene detection tests 通过。
- lingolens prompt builder tests 通过。
- 用一个短视频对比新旧 `script.json` 的 shot boundary 是否更贴近 cut times。

### Phase 2: Edit Atom Builder

目标：

- 根据 Stage 1 shots + rewrite lines 构建 atoms。
- 不拆台词；默认不跨 shot，除非满足相邻 shot 的“同类相吸”规则。
- scene cuts 只用于边界吸附。

验证：

- rewritten lines 全部被至少一个 target atom 覆盖。
- atom 边界不落在任何 line 内部。
- 跨 shot atom 必须有明确 similarity / continuity reason。

### Phase 3: Segment Matcher

目标：

- 一次 LLM 调用匹配 atoms 到 canvas nodes。
- 为生产规模保留 coarse retrieval + global fine matching 的扩展路径。

验证：

- 所有 matched atom 有 node_id。
- unmatched atom 有 reason。

### Phase 4: Generation Window Resolver + Window Prompt Rewriter + Plan Finalizer

目标：

- 所有 modified TimelinePlanItem >= 4s。
- 短 atom 通过合并/扩展/clamp 到安全 4s window，而不是直接 fallback。
- Prompt rewrite 在 window resolution 之后执行。
- 输出 timeline 无 overlap/gap。
- covered_line_ids 覆盖所有 rewritten lines 或有明确 fallback。

验证：

- `validate_timeline_items()` 通过。
- rewritten dialogue 出现在 rewritten prompt 中。
- style anchors 保留率达到阈值。
- Stage 4 `--skip-seedance` 可完整组装。
- 至少一条实际 seedance run 验证 prompt/ref_images/duration 正常。

### Phase 5: 删除/归档旧路径

目标：

- 删除 v3 不再使用的 modules 或标记 deprecated。
- 更新 README / SKILL.md / issue 说明。

---

## 15. 待确认问题

这些问题不阻塞设计，但会影响实现细节：

1. **Atom cluster 的默认 gap 阈值**
   - 建议初始值：`1.5s`。
   - 作用：同一 shot 内连续 rewritten lines 间隔小于该值时，可进入语义兼容性判断。
   - 注意：gap 只是必要条件，不是充分条件；交替出现的 unchanged line 需要按 7.3.3 判断。

2. **短 atom 扩展时是否允许跨 shot**
   - 建议默认不跨。
   - 如果无法达到 4s，再允许跨相邻 shot，并记录 degradation reason。

3. **Unmatched atom 的策略**
   - 选项 A：生成 degraded prompt，不依赖 canvas node。
   - 选项 B：直接 fallback original，并在 report 中标记该 rewritten line 未生效。
   - 建议初版采用 A，但不使用 keyframes 补 ref_images；无 node ref images 时 Stage 4 仍可能 fallback。

4. **多个短 atom 不同 node 但相邻时是否合并**
   - 建议初版不合并。
   - 不同 node 表示不同生成意图，强合并容易污染 prompt。

5. **Stage 1 shot boundary 与 ASR line timing 冲突时的优先级**
   - 建议 ASR line timing 优先，shot boundary 不能截断 line。

6. **跨 shot atom 合并的 scene similarity 实现**
   - 初版可用规则/token overlap。
   - 如果后续误合并较多，再引入 embedding similarity。

---

## 16. 最终判断

v4 的正确抽象是：

```text
Edit Atom: prompt matching unit
Generation Window: video execution unit
TimelinePlanItem: Stage 4 assembly unit
```

只要这三层分清楚，Segment-First 方案是正确且可落地的。  
它能移除 v3 中最脆弱的“逐行匹配 + 事后分组 + normalizer 挖洞”链路，同时保留 Stage 4 当前的执行模型。
