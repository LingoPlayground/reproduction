# 基于 ASR 时间轴的视频重生成 Pipeline 设计

> 日期：2026-05-29 | 状态：设计阶段 | 作者：Sisyphus

---

## 1. 概述

### 1.1 动机

当前 pipeline 的核心问题：**Stage 3 试图将台词逐行匹配到画布节点，但画布节点含废片/花絮，匹配不可靠**。

但 Stage 1 已经产出了精确的 ASR 时间戳数据（毫秒级 `start_seconds` / `end_seconds`），以及多模态 LLM 分析视频帧得出的 `ScriptShot` 边界。这些数据在 Stage 3 中**完全未被使用**。

### 1.2 核心思路转变

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| 视频源 | 画布节点视频 | **原剧完整视频** |
| 剪辑位置 | 由节点匹配决定 | **ASR 时间戳 + ScriptShot 边界决定** |
| 画布节点用途 | 视频源 + prompt 改写 | **仅参考图 + prompt 模板** |
| 无改写镜头 | 下载画布节点视频 | **直接截取原剧片段** |
| 废片处理 | LLM 猜测剔除 | **不依赖画布节点，天然规避** |

### 1.3 设计原则

1. **原剧视频是时间轴基准**：所有剪辑位置由原剧视频的时间轴决定
2. **ASR 时间戳是剪刀**：Stage 1 产出的精确时间戳是 cut point 的主信号
3. **画布节点降级为视觉资产库**：仅提供参考图和 prompt 模板，不影响拼接结果
4. **PySceneDetect 是辅助校准器**：在多个环节提供画面级精度补充

---

## 2. 当前代码的问题分析

### 2.1 Stage 1 产出丰富但未被下游使用

`lingolens/backend/agents/script_extraction/` 产出的 `VideoScriptOutput` 包含：

```python
# ScriptShot — 多模态 LLM 分析视频帧 + ASR 得出
ScriptShot:
  shot_number: int
  start_seconds: float       # 镜头逻辑开始时间
  end_seconds: float         # 镜头逻辑结束时间
  location: Optional[str]    # 场景位置名称
  scene_description: str     # LLM 理解后的场景描述
  shot_type: Optional[str]   # 镜头类型
  camera_movement: Optional[str]  # 运镜方式
  mood: Optional[str]        # 情绪氛围
  lines: List[ScriptLine]    # 该镜头内的台词

# ScriptLine — ASR 精确时间戳填充
ScriptLine:
  line_id: str
  parent_line_id: Optional[str]  # 分割后的父行ID
  role_id: Optional[str]
  start_seconds: float           # ASR 开始时间（ms精度，_enrich_lines_from_asr 填充）
  end_seconds: float             # ASR 结束时间（_enrich_lines_from_asr 填充）
  asr_utterance_index: int       # 对应 ASR utterance 索引
  asr_word_start_index: Optional[int]
  asr_word_end_index: Optional[int]
  asr_word_start_seconds: Optional[float]  # ⚠️ 字段存在但 normalizer 从未填充，始终为 None
  asr_word_end_seconds: Optional[float]    # ⚠️ 同上
  speaker: str
  dialogue: str
  action: Optional[str]
  speech_mode: SpeechMode
```

**⚠️ 关键发现：`asr_word_*` 字段是"空承诺"**

`normalizer.py` 的 `_enrich_lines_from_asr()` (L112-145) 只填充了 `dialogue`、`start_seconds`、`end_seconds` 三个 utterance 级别的字段。词级时间戳虽然在 ASR utterances 的 `words[]` 数组中可用，但正常化流程**没有任何一步**将它们写入 `asr_word_start_seconds` / `asr_word_end_seconds`。`splitter.py` 虽然使用词级数据来切分长句，但切分后只覆盖了 `start_seconds` / `end_seconds`，未写入 `asr_word_*`。

**影响**：新 Pipeline 若需要词级精度的剪辑信号，需先在 Stage 1 中修复此问题（见 §4.1 补充说明）。当前行级 `start_seconds`/`end_seconds` 精度（毫秒级）已足够驱动剪辑，词级字段作为未来增强项。

但在 Stage 3 (`match_to_canvas.py`) 的 `extract_lines_from_script()`（第 95-113 行）中，仅提取了 `start_seconds` 和 `end_seconds`，**完全忽略了 `asr_*` 字段和 `ScriptShot` 结构**，转而使用纯文本匹配。

### 2.2 画布节点匹配的不可靠性

`match_to_canvas.py` 中的 `llm_end_to_end_match()`（第 359-529 行）将所有节点和所有台词一次性发送给 DeepSeek，通过 LLM 做全局匹配。问题：

1. **纯文本匹配**：LLM 通过文本判断哪个节点对应哪句台词，无法利用时序信息
2. **废片干扰**：画布节点的 prompt 可能包含 B-roll/花絮/NG 镜头的台词，LLM 需猜测剔除
3. **ASR 偏差放大**：ASR 转写错误 + 中英混合 + 同名多态，纯文本匹配易出错
4. **匹配失败影响全局**：一次匹配失败可能导致后续节点全部错位

### 2.3 现有资产总结

| 资产 | 位置 | 精度 | 当前使用状态 |
|------|------|------|------------|
| ScriptShot 边界 | Stage 1 输出 | LLM + 视觉分析 | ❌ Stage 3 未使用 |
| ASR 行级时间戳 | Stage 1 输出 | 毫秒级（已填充） | ❌ Stage 3 未使用 |
| ASR 词级时间戳 | Stage 1 输出 | 帧级（字段存在但未填充，始终 None） | ❌ 暂不可用，需先修复 normalizer.py |
| 改写台词 | Stage 2 输出 | — | ✅ Stage 3 使用 |
| 画布节点 prompt | LibLib API | — | ✅ Stage 3 使用 |
| 画布节点参考图 | LibLib API | — | ✅ Stage 4 使用 |
| 原剧完整视频 | 输入 | — | ❌ 仅 Stage 1 使用 |

---

## 3. 新 Pipeline 架构

### 3.1 整体流程

```
Stage 1 (不变):  原剧视频 → Multimodal LLM → ScriptOutput
Stage 1b (新增): 原剧视频 → PySceneDetect → CutPoints + KeyFrames
Stage 2 (不变):  ScriptOutput → CEFR ReWriter → RewriteJSON
Stage 3 (重写):  剪辑计划生成 → TimelinePlan
Stage 4 (重写):  视频组装 → final.mp4
```

### 3.2 数据流

> **数据模型统一说明**：Stage 1 产出的是 `VideoScriptOutput`（lingolens 模型），当前 Stage 3 消费的是 `ScriptInput`（shakespeare 模型，丢失了 `asr_*` 字段和 ScriptShot 元数据）。新 Stage 3 直接消费 `VideoScriptOutput`，保留完整数据结构。

```
┌─ Stage 1 ─────────────────────────────────────────────────────┐
│ 原剧视频.mp4 + utterances[] → Multimodal LLM                  │
│ → VideoScriptOutput（lingolens 模型，保留完整 asr_* 字段）    │
│   shots[]{ shot_number, start_s, end_s, scene_desc, lines[] } │
│   lines[]{ line_id, dialogue, start_s, end_s, asr_utr_idx }   │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌─ Stage 1b (新增) ─────────┼──────────────────────────────────┐
│ 原剧视频.mp4 → PySceneDetect → VideoCutPoints[]              │
│ 画布节点视频 → PySceneDetect → NodeCutPoints[]               │
│ 原剧视频.mp4 + CutPoints → KeyFrames[] (fallback参考图)      │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌─ Stage 2 (不变) ──────────┼──────────────────────────────────┐
│ ScriptOutput → CEFR ReWriter → RewriteJSON                   │
│   lines[]{ line_id, original, rewritten, start_s, end_s }    │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌─ Stage 3 (重写) ──────────┼──────────────────────────────────┐
│                                                              │
│  for each ScriptShot:                                        │
│    ├─ cut = fuse_CutPoint(shot, VideoCutPoints)              │
│    ├─ has_rewrite = any(line in shot has rewritten ≠ orig)    │
│    │                                                         │
│    ├─ if has_rewrite:                                        │
│    │   ├─ match_canvas_node(shot) → CanvasNode               │
│    │   ├─ extract_prompt_fragment(node.prompt, shot)         │
│    │   ├─ replace_dialogue(fragment, rewrite_lines)          │
│    │   ├─ collect_ref_images(node.images + KeyFrames)        │
│    │   └─ source = "seedance"                                │
│    │                                                         │
│    └─ if not has_rewrite:                                    │
│        └─ source = "original"                                │
│                                                              │
│  output → TimelinePlan[]{                                     │
│    shot_id, source, start_s, end_s,                          │
│    ref_images[], rewritten_prompt, original_video_url         │
│  }                                                            │
└────────────────────────────┬──────────────────────────────────┘
                             │
┌─ Stage 4 (重写) ──────────┼──────────────────────────────────┐
│                                                              │
│  for each item in TimelinePlan:                              │
│    ├─ source="original" → ffmpeg trim(original_video,        │
│    │                      item.start_s, item.end_s)          │
│    └─ source="seedance" → seedance API(                      │
│    │                      ref_images, rewritten_prompt)      │
│                                                              │
│  ffmpeg concat all segments → final.mp4                      │
└──────────────────────────────────────────────────────────────┘
```

---

## 4. 各 Stage 详细设计

### 4.1 Stage 1：剧本提取（不变）

**文件**：`skills/script-extraction/extract_script.py`

完全复用现有实现。产出 `VideoScriptOutput`，包含：
- `script.shots[]`：每个 shot 的边界、场景描述、台词列表
- `script.shots[].lines[]`：每行台词的精确 ASR 时间戳

**改动**：仅需在 lingolens 侧修复 `_enrich_lines_from_asr()`，将 ASR utterances 的 `words[]` 数据写入 `asr_word_start_seconds` / `asr_word_end_seconds`（当前字段存在但始终为 None）。这是**可选增强**——行级时间戳已足够驱动剪辑；词级时间戳为未来帧级精度需求做准备。

### 4.2 Stage 1b：场景检测增强（新增）

**新文件**：`skills/scene-detection/detect_scenes.py`

#### 4.2.1 输入

- 原剧视频文件路径
- 画布节点视频 URL 列表（可选，用于节点内部分析）

#### 4.2.2 处理

**任务 A：原剧视频镜头检测**

> **API 说明**：PySceneDetect 的 `get_scene_list()` 返回 `list[tuple[TimeCode, TimeCode]]`（每个元素是 (start, end) 二元组），而非 `(scene, score)`。置信度需从 detector 的事件统计中自行计算。以下代码示例中的 `confidence` 为简化占位值。

```python
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector

def detect_scene_boundaries(video_path: str) -> list[CutPoint]:
    """
    返回原剧视频的镜头切换点列表。
    使用 ContentDetector（HSV 色彩直方图差异）+ AdaptiveDetector（自适应阈值）。
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    
    # AI 生成视频画面过渡更平滑，阈值较实拍视频放宽
    # 实拍视频默认：27.0；AI 生成视频建议：15.0-22.0
    scene_manager.add_detector(ContentDetector(threshold=20.0))
    scene_manager.add_detector(AdaptiveDetector())
    scene_manager.detect_scenes(video)
    
    # get_scene_list() 返回 [(start_TimeCode, end_TimeCode), ...]
    scene_list = scene_manager.get_scene_list()
    return [
        CutPoint(time_sec=cut[0].get_seconds(), confidence=1.0)
        for cut in scene_list
    ]
```

> **阈值调优**：`ContentDetector(threshold=20.0)` 适用于 AI 生成视频（画面过渡比实拍更平滑）。若漏检过多，降低至 15.0；若误检过多（同一镜头内检测出切换），升高至 25.0。推荐在项目配置文件中暴露此参数。

**任务 B：画布节点视频内部分段**

```python
def detect_node_internal_cuts(node_video_url: str) -> list[CutPoint]:
    """
    分析画布节点视频的内部镜头切换点。
    当一个画布节点的 prompt 覆盖多个 ScriptShot 时，
    帮助定位 prompt 中「镜头1」「镜头2」等段落的对应关系。
    """
    # 下载节点视频 → PySceneDetect → 返回内部切换点
```

**任务 C：关键帧提取（参考图 fallback）**

```python
def extract_keyframes(video_path: str, cut_points: list[CutPoint]) -> list[KeyFrame]:
    """
    在镜头切换点前后提取关键帧。
    当画布节点匹配失败时，作为 seedance 参考图的 fallback。
    """
```

#### 4.2.3 输出

```python
@dataclass
class CutPoint:
    time_sec: float       # 切换点时间（秒）
    confidence: float     # 置信度 (0.0-1.0)

@dataclass 
class KeyFrame:
    time_sec: float       # 帧在视频中的时间
    image_path: str       # 提取的帧图片路径
    shot_number: int      # 对应 ScriptShot

@dataclass
class SceneDetectionOutput:
    video_cut_points: list[CutPoint]           # 原剧视频切换点
    node_cut_points: dict[str, list[CutPoint]] # node_id → 内部切换点
    keyframes: list[KeyFrame]                   # fallback 参考图
```

### 4.3 Stage 2：剧本改写（不变）

**文件**：`skills/script-rewriting/rewrite_script.py`

完全复用现有实现。产出改写 JSON：
```json
{
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
  ]
}
```

**无需任何改动**。

### 4.4 Stage 3：剪辑计划生成（重写）

**新文件**：`skills/timeline-plan/generate_plan.py`（替代 `match_to_canvas.py`）

这是整个新方案的核心。不再做逐行台词→节点匹配，而是生成一个结构化的 `TimelinePlan`。

#### 4.4.1 步骤 1：确定剪辑边界

```python
def fuse_cut_boundary(
    shot: ScriptShot,
    video_cut_points: list[CutPoint],
    tolerance: float = 0.5  # 容差窗口（秒）
) -> tuple[float, float]:
    """
    融合 LLM Shot 边界和 PySceneDetect 切换点，确定最终 cut。
    
    算法：
    1. shot.start_s 在 video_cut_points 中找最近的切换点（tolerance 内）
    2. shot.end_s 同理
    3. 若找到 → 使用 PySceneDetect 的精确点
    4. 若未找到 → 使用 LLM 的边界
    """
```

**融合策略**：

| 情况 | LLM 边界 | PySceneDetect | 结果 |
|------|---------|---------------|------|
| 两者一致（容差内） | shot.start | cut.time ∈ [shot.start±0.5s] | cut.time（精确帧） |
| 仅 LLM 有 | shot.start | 无 | shot.start |
| 仅 PySceneDetect | 无（LLM 未分镜） | cut.time | cut.time（可能是空镜/过渡） |

#### 4.4.2 步骤 2：判断是否改写

```python
def shot_needs_rewrite(
    shot: ScriptShot,
    rewrite_lines: list[RewriteLine]
) -> bool:
    """
    判断一个 shot 是否需要重新生成。
    条件：shot 中任意一行的 original ≠ rewritten。
    """
    shot_line_ids = {line.line_id for line in shot.lines}
    for rl in rewrite_lines:
        if rl.line_id in shot_line_ids and rl.original != rl.rewritten:
            return True
    return False
```

#### 4.4.3 步骤 3：画布节点匹配（仅对有改写的 shot）

```python
def match_canvas_node(
    shot: ScriptShot,
    canvas_nodes: list[CanvasNode],
    rewrite_lines: list[RewriteLine]
) -> Optional[CanvasNode]:
    """
    通过台词文本 + 场景描述的语义相似度匹配画布节点。
    
    匹配信号（按优先级）：
    1. 台词文本相似度：shot 中 lines 的 dialogue 与 node.prompt 的文本重叠度
    2. 场景描述相似度：shot.scene_description 与 node.prompt 的语义相似度
    3. 角色一致性：shot 中 speaker 是否出现在 node.prompt 中
    
    返回最佳匹配的节点。若置信度 < 阈值，返回 None（使用 fallback）。
    """
```

**关键改进**：匹配不需要精确到逐行。一个 shot 匹配到一个画布节点即可——即使节点覆盖多个 shot，后续会做 prompt 片段提取。

#### 4.4.4 步骤 4：Prompt 片段提取（核心新逻辑）

一个画布节点的 prompt 可能覆盖多个 shot。例如节点 `0e19c709` 的 prompt 包含 "镜头 1" 到 "镜头 5" 五个镜头描述，对应 ScriptShot 5 和 6。

对于目标 shot，需要从长 prompt 中**提取与之对应的片段**：

**前置验证：时间轴对齐确认**

画布节点的 prompt 内时间顺序**不一定**与 ScriptShot 的剧情时间顺序一致（可能是拍摄顺序、B-roll 混合）。在提取片段前，用 PySceneDetect 分析节点视频的时间戳（见 §5.3 角色 2），与 ScriptShot 的 ASR 时间戳交叉比对，确认两者的时间轴对应关系：

- **对齐**：prompt 中的「镜头 N」与 ScriptShot N 一一对应 → 直接做结构化提取
- **错位**：prompt 中的顺序与剧情顺序不一致 → 降级到 Level 2（LLM 语义分段）或 Level 3（关键词定位）
- **不可判定**：节点视频与 ScriptShot 无对应关系 → 降级到 Level 4（全新生成）

```python
def extract_prompt_fragment(
    full_prompt: str,
    target_shot: ScriptShot,
    rewrite_lines: list[RewriteLine]
) -> str:
    """
    从画布节点的长 prompt 中提取目标 shot 对应的视觉描述片段。
    
    策略（按优先级降级）：
    
    1. 结构化解析：prompt 中有「镜头 N」「镜头 N：」等标记，
       匹配 shot 的 scene_description 与各镜头的文本描述，
       定位到对应的镜头段落。
    
    2. 语义分段：用 LLM 将长 prompt 按镜头语义分段，
       每个段落与 target_shot.scene_description 计算相似度，
       取最相似的段落。
    
    3. 台词定位：在 prompt 中搜索 shot 中台词的关键词，
       定位包含这些台词的行及上下文。
    
    4. 降级策略：若以上都失败，使用 target_shot.scene_description 
       作为视觉描述基底，完全用 LLM 生成 prompt。
    
    提取后，调用 replace_dialogue_in_fragment() 替换台词。
    """
```

**台词替换（在提取的片段内）**：

```python
def replace_dialogue_in_fragment(
    fragment: str,
    rewrite_lines: list[RewriteLine]
) -> str:
    """
    在 prompt 片段中，将原台词替换为改写台词。
    
    沿用现有 match_to_canvas.py 的 replace_dialogue_in_prompt() 的
    4级降级策略（引号内 → 冒号后 → 精确子串 → 滑动窗口），
    但操作范围从「整个 prompt」缩小到「提取的片段」。
    """
```

#### 4.4.5 步骤 5：参考图收集

```python
def collect_reference_images(
    matched_node: Optional[CanvasNode],
    shot: ScriptShot,
    keyframes: list[KeyFrame]
) -> list[str]:
    """
    收集 seedance 生成需要的参考图。
    
    优先级：
    1. 画布节点的参考图（如果匹配成功）
    2. 原剧视频的关键帧（从 PySceneDetect 提取的对应 shot 的帧）
    3. 同集其他画布节点的参考图（fallback）
    
    返回图片路径或 URL 列表。
    """
```

#### 4.4.6 输出：TimelinePlan

```python
@dataclass
class TimelinePlanItem:
    shot_id: str                    # shot 标识（如 "shot_1"）
    shot_number: int                # ScriptShot.shot_number
    source: Literal["original", "seedance"]  # 片段来源
    start_sec: float                # 在原剧视频中的开始时间
    end_sec: float                  # 在原剧视频中的结束时间
    duration_sec: float             # 片段时长
    ref_images: list[str]           # 参考图（仅 seedance）
    rewritten_prompt: Optional[str] # 改写后的 prompt（仅 seedance）
    original_video_url: str         # 原剧视频路径
    scene_description: str          # 场景描述（用于日志/debug）

@dataclass
class TimelinePlan:
    title: str
    total_duration: float
    original_video_path: str
    items: list[TimelinePlanItem]    # 按时间顺序排列
```

### 4.5 Stage 4：视频组装（重写）

**新文件**：`skills/video-assembly/assemble.py`（替代 `generate_videos.py`）

#### 4.5.1 处理

```python
async def assemble_video(plan: TimelinePlan, output_path: str):
    segment_paths: list[str] = []
    
    for item in plan.items:
        if item.source == "original":
            # 直接从原剧视频截取
            seg_path = f"/tmp/seg_{item.shot_number}.mp4"
            ffmpeg_trim(
                input=item.original_video_url,
                start=item.start_sec,
                end=item.end_sec,
                output=seg_path
            )
            segment_paths.append(seg_path)
            
        elif item.source == "seedance":
            # 调用 seedance API 生成新视频
            seg_path = f"/tmp/seg_{item.shot_number}_gen.mp4"
            await seedance_generate(
                images=item.ref_images,
                prompt=item.rewritten_prompt,
                duration=item.seedance_duration,  # 使用归一化后的时长
                audio=True,  # 生成音频（同 generate_videos.py L201）
                output=seg_path
            )
            segment_paths.append(seg_path)
    
    # 编码统一化 + 音频归一化后再拼接
    normalized_paths = normalize_segments(segment_paths)
    concat_list = write_concat_file(normalized_paths)
    ffmpeg_concat(concat_list, output_path)
```

#### 4.5.2 Seedance 时长归一化

seedance 2.0 fast 的 `duration` 参数有步长约束（通常为整数秒，范围 5-30 秒）。若 `shot.duration_sec = 3.7s` 或 `12.3s`，需做归一化处理：

```python
def normalize_seedance_duration(target_sec: float) -> int:
    """
    四舍五入到最接近的整数秒，并 clamp 到 [5, 30] 范围。
    记录 original_duration 与 seedance_duration 的偏差，用于 QA 评估。
    """
    return max(5, min(30, round(target_sec)))
```

若目标 shot 时长超出 [5, 30] 范围（如超长镜头），拆分为多个 seedance 调用或降级为 `source="original"`。

#### 4.5.3 编码统一化（解决 -c copy 兼容性问题）

`-c copy` 要求所有 segment 编码参数完全一致。seedance 生成的视频和 ffmpeg trim 的原剧片段可能有不同的 `h264_profile`、`pix_fmt` 或 `keyint`，直接 concat 可能失败或产生跳帧。

```python
def normalize_segments(segment_paths: list[str]) -> list[str]:
    """将所有 segment 统一转码为相同编码参数后返回新路径列表"""
    normalized = []
    for seg_path in segment_paths:
        norm_path = seg_path.replace(".mp4", "_norm.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", seg_path,
            "-c:v", "libx264", "-profile:v", "high",
            "-pix_fmt", "yuv420p", "-crf", "18",
            "-c:a", "aac", "-ar", "44100", "-b:a", "192k",
            norm_path
        ], check=True)
        normalized.append(norm_path)
    return normalized
```

#### 4.5.4 音频连续性处理

`source="original"` 片段的音频来自原剧，`source="seedance"` 的音频来自 seedance。两者在 concat 前需要音量一致：

```python
def normalize_audio_loudness(segments: list[str]) -> list[str]:
    """使用 EBU R128 loudnorm filter 统一所有片段的响度"""
    normalized = []
    for seg_path in segments:
        norm_path = seg_path.replace(".mp4", "_loud.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", seg_path,
            "-af", "loudnorm=I=-16:LRA=11:TP=-1.5",
            "-c:v", "copy",  # 视频流不变
            norm_path
        ], check=True)
        normalized.append(norm_path)
    return normalized
```

建议将 §4.5.3 编码统一化和 §4.5.4 音频归一化合入同一个 `normalize_segments()` 步骤，避免两次转码。

#### 4.5.5 ffmpeg 拼接

```bash
# 截取原剧片段
ffmpeg -ss {start} -to {end} -i original.mp4 -c copy seg_{n}.mp4

# 拼接所有片段
ffmpeg -f concat -safe 0 -i concat.txt -c copy final.mp4
```

---

## 5. PySceneDetect 集成详解

### 5.1 在 Pipeline 中的三个角色

| 角色 | 输入 | 输出 | 使用位置 |
|------|------|------|---------|
| **角色 1：剪辑点精修** | 原剧视频 | `VideoCutPoints[]` | Stage 3 步骤 1（确定 cut） |
| **角色 2：节点内部分段** | 画布节点视频 | `NodeCutPoints[]` | Stage 3 步骤 4（prompt 片段提取辅助） |
| **角色 3：关键帧提取** | 原剧视频 + CutPoints | `KeyFrames[]` | Stage 3 步骤 5（参考图 fallback） |

### 5.2 角色 1 详解：剪辑点精修

**问题**：Stage 1 的 LLM 观看的是压缩后的视频（480p, 8fps），其确定的 `ScriptShot.start_seconds` / `end_seconds` 可能不够精确（误差可达 1-2 秒）。

**方案**：PySceneDetect 以原始分辨率分析视频，检测 Content-Aware 镜头切换。在 LLM 边界 ± tolerance 窗口内，若有 PySceneDetect 检测到的切换点，则以此精修后的点作为最终 cut。

```
ScriptShot boundary (LLM):     |──────────── Shot 3 ────────────|
                                 32.7s                      43.5s

PySceneDetect cut points:  ...30.2s......32.5s*......40.1s.....43.8s*....46.2s...
                                    ↑                      ↑
                              选取 32.5s              选取 43.8s

Final cut:                 32.5s ──────────────────────────── 43.8s
```

**容差窗口设置**：

```python
CUT_FUSION_TOLERANCE = 0.5  # 秒，可配置

# 若 LLM 边界和 PySceneDetect 的距离 < tolerance，采用后者
# 若 LLM 边界附近无 PySceneDetect 点，保留 LLM 边界
```

### 5.3 角色 2 详解：节点内部分段

**问题**：画布节点的 prompt 覆盖多个 shot，需要知道 prompt 中「镜头 1」「镜头 2」等段落各自对应哪个 ScriptShot。

**方案**：
1. 下载画布节点视频
2. PySceneDetect 分析该视频的内部镜头切换点
3. 结合 prompt 中的「镜头 N」标记，建立 {镜头段落 → 视频时间范围} 的映射
4. 辅助 `extract_prompt_fragment()` 更精确地定位目标 shot 对应的 prompt 段落

**注意**：此步骤为辅助性质，不是必须的。Prompt 片段提取有独立的降级策略（见 4.4.4）。

### 5.4 角色 3 详解：关键帧提取

**问题**：当画布节点匹配失败或置信度低时，需要 fallback 参考图。

**方案**：
1. 在 PySceneDetect 检测到的每个切换点处，从原剧视频提取一帧
2. 将关键帧按时间戳与 `ScriptShot` 的边界关联
3. 匹配失败时，使用该 shot 最近的关键帧作为 seedance 参考图

**帧提取命令**：
```bash
ffmpeg -ss {time_sec} -i original.mp4 -frames:v 1 keyframe_{shot_num}.png
```

---

## 6. 画布节点匹配与 Prompt 片段提取

### 6.1 匹配策略（替代旧的 LLM e2e matching）

旧方案（`match_to_canvas.py::llm_end_to_end_match`）：把所有节点和所有台词一次性发给 DeepSeek，LLM 做全局匹配。

新方案：**分步匹配，每步有明确的降级策略**。

```python
def match_canvas_node_for_shot(
    shot: ScriptShot,
    nodes: list[CanvasNode],
    rewrite_lines: list[RewriteLine]
) -> Optional[tuple[CanvasNode, float]]:
    """
    为一个 ScriptShot 匹配画布节点。
    返回 (节点, 置信度) 或 None。
    """
    # Step 1: 文本重叠度评分（复用 pipeline.py 的 fuzzy_match_score 逻辑）
    # 该函数经多轮迭代，对 ASR 噪声（标点/中英混合/语气词）有较好容错性
    shot_dialogue_text = " ".join(line.dialogue for line in shot.lines)
    candidates = []
    for node in nodes:
        score = text_overlap_score(shot_dialogue_text, node.prompt)
        if score > TEXT_OVERLAP_THRESHOLD:
            candidates.append((node, score))
    
    # Step 2: 场景描述语义相似度（仅对候选节点）
    if len(candidates) > 1:
        candidates = rerank_by_semantic_similarity(
            shot.scene_description, candidates
        )
    
    # Step 3: 返回最佳匹配
    if candidates:
        best_node, confidence = candidates[0]
        if confidence > CONFIDENCE_THRESHOLD:
            return (best_node, confidence)
    
    return None  # 将使用 fallback
```

### 6.2 Prompt 片段提取（替代旧的 replace_dialogue_in_prompt）

旧方案：在**整个**节点 prompt 中替换台词。问题是 prompt 可能包含不相关 shot 的台词。

新方案：先**定位**目标 shot 对应的 prompt 片段，再在其中替换：

```python
def extract_and_rewrite_prompt(
    full_prompt: str,
    target_shot: ScriptShot,
    rewrite_lines: list[RewriteLine],
    node_cut_points: Optional[list[CutPoint]] = None
) -> str:
    """
    主流程：
    1. 定位目标片段
    2. 提取视觉描述
    3. 替换台词
    """
    
    # Level 1: 结构化解析（prompt 中有「镜头N」标记）
    fragment = extract_by_section_headers(full_prompt, target_shot)
    if fragment:
        return replace_dialogue_in_fragment(fragment, rewrite_lines)
    
    # Level 2: LLM 语义分段
    segments = llm_split_prompt_into_segments(full_prompt)
    best_segment = max(segments, key=lambda s: 
        semantic_similarity(s, target_shot.scene_description))
    if best_segment and semantic_similarity(best_segment, target_shot.scene_description) > 0.5:
        return replace_dialogue_in_fragment(best_segment, rewrite_lines)
    
    # Level 3: 台词关键词定位
    fragment = extract_by_dialogue_keywords(full_prompt, target_shot)
    if fragment:
        return replace_dialogue_in_fragment(fragment, rewrite_lines)
    
    # Level 4: 完全降级 — 用 scene_description 生成全新 prompt
    return generate_prompt_from_scene(target_shot, rewrite_lines)
```

### 6.3 废片节点的处理

新方案的天然优势：即使匹配到的画布节点是废片（B-roll、花絮），也不影响最终视频质量。

**原因**：
1. 画布节点**不直接用作视频源** — 仅取参考图和 prompt 模板
2. prompt 片段提取时，只取与目标 shot 语义相关的部分
3. 废片节点的 prompt 若完全不相关，匹配分数会很低 → 降级到 Level 4（LLM 全新生成 prompt）
4. 参考图有 fallback（原剧关键帧）兜底

---

## 7. 剪辑点融合算法

### 7.1 算法伪代码

```python
def determine_cut_points(
    script_shots: list[ScriptShot],
    scene_cuts: list[CutPoint],
    video_duration: float,
    tolerance: float = 0.5
) -> list[tuple[float, float]]:
    """
    输入：ScriptShot 列表 + PySceneDetect 切换点列表 + 视频总时长
    输出：每个 shot 的精确 (start, end) 时间
    """
    results = []
    
    for shot in script_shots:
        # 边界合理性检查：clamp 到 [0, video_duration] 范围内
        raw_start = max(0.0, min(shot.start_seconds, video_duration))
        raw_end = max(0.0, min(shot.end_seconds, video_duration))
        if raw_end <= raw_start:
            raw_end = min(raw_start + 1.0, video_duration)  # 最小 1 秒
        
        # 找 start 最近的 PySceneDetect 点
        start_cut = find_nearest_cut(scene_cuts, raw_start, tolerance)
        final_start = start_cut.time_sec if start_cut else raw_start
        
        # 找 end 最近的 PySceneDetect 点
        end_cut = find_nearest_cut(scene_cuts, raw_end, tolerance)
        final_end = end_cut.time_sec if end_cut else raw_end
        
        results.append((final_start, final_end))
    
    # 处理间隙：相邻 shot 之间的空白区域
    results = fill_gaps(results, scene_cuts)
    
    return results


def find_nearest_cut(
    cuts: list[CutPoint], 
    target: float, 
    tolerance: float
) -> Optional[CutPoint]:
    """在 tolerance 窗口内找最近的切换点"""
    candidates = [
        cut for cut in cuts 
        if abs(cut.time_sec - target) <= tolerance
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.time_sec - target))
```

### 7.2 间隙处理

当相邻 shot 的 (end, start) 之间存在时间间隙时（约 0-2 秒），这段区域可能包含无台词的过渡镜头。处理策略：

1. 若 PySceneDetect 在间隙中检测到切换点 → 将其间作为独立段落
2. 若间隙 < 0.5s → 合并到前一个 shot（避免黑帧）
3. 若间隙 ≥ 0.5s → 作为独立段落，标记为 `source="original"`

---

## 8. 数据模型定义

### 8.1 TimelinePlan（新增核心模型）

```python
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class CutPoint:
    """PySceneDetect 检测到的镜头切换点"""
    time_sec: float
    confidence: float

@dataclass
class KeyFrame:
    """从视频中提取的关键帧"""
    time_sec: float
    image_path: str
    shot_number: int

@dataclass
class CanvasNode:
    """画布节点数据（从 LibLib API 获取）"""
    node_id: str
    prompt: str
    video_url: str
    reference_images: list[str]
    duration_sec: Optional[float] = None

@dataclass
class TimelinePlanItem:
    """剪辑计划中的单个片段"""
    shot_id: str
    shot_number: int
    source: Literal["original", "seedance"]
    start_sec: float
    end_sec: float
    scene_description: str
    ref_images: list[str] = field(default_factory=list)
    rewritten_prompt: Optional[str] = None
    matched_node_id: Optional[str] = None
    match_confidence: Optional[float] = None
    degradation_level: int = 0        # 降级层级（0=最优路径, 1-3=降级, 对应 §9.4）
    seedance_duration: Optional[int] = None  # seedance 归一化后的时长（整数秒）
    original_duration: Optional[float] = None  # 原始 shot 时长（用于计算偏差）

@dataclass
class TimelinePlan:
    """完整的剪辑计划"""
    title: str
    level: str  # CEFR 等级
    pipeline_version: str = "2.0"      # pipeline 版本号，区分不同版本的 plan
    original_video_path: str
    total_duration_sec: float
    items: list[TimelinePlanItem]
    metadata: dict = field(default_factory=dict)
```

### 8.2 Stage 3 输入接口

```python
@dataclass
class Stage3Input:
    """Stage 3 的输入数据结构"""
    # 来自 Stage 1
    script_output: VideoScriptOutput  # ScriptShot + ScriptLine
    
    # 来自 Stage 1b
    video_cut_points: list[CutPoint]
    keyframes: list[KeyFrame]
    node_cut_points: dict[str, list[CutPoint]]  # node_id → 内部切换点
    
    # 来自 Stage 2（格式：{"level": str, "lines": [{line_id, original, rewritten, start_seconds, end_seconds, shot_scene}]}）
    rewrite_json: dict  # 结构见 §4.3 示例
    
    # 外部数据
    canvas_nodes: list[CanvasNode]  # 从 LibLib API
    level: str  # 目标 CEFR 等级
```

---

## 9. 错误处理与降级策略

### 9.0 降级日志

所有降级事件需写入 `TimelinePlanItem.degradation_level`（0=最优路径，1-3=逐级降级），并同步输出到 `timeline_plan.json` 的 debug 段。seedance 生成质量的自动化检查：提取生成视频的首帧和末帧，与参考图做结构相似度（SSIM）比对，若 SSIM < 0.4 则标记为质量可疑，降级使用原剧片段。

### 9.1 画布节点匹配失败

| 场景 | 处理 |
|------|------|
| 无节点匹配 | 使用原剧关键帧作为参考图，Level 4 降级（LLM 从 scene_description 生成 prompt） |
| 匹配置信度低 | 使用匹配节点的参考图 + Level 2/3 降级的 prompt 片段 |
| 节点 prompt 无法提取对应片段 | 使用匹配节点的全部参考图 + Level 4 降级 |
| 节点参考图获取失败 | 使用原剧关键帧 fallback |

### 9.2 seedance 生成失败

| 场景 | 处理 |
|------|------|
| API 超时/报错 | 重试 3 次，若仍失败 → 降级为 `source="original"` |
| 生成视频质量低 | 人工标记，二次生成时调整参数 |
| 生成视频时长不匹配 | 若时长偏差 < 10%，使用生成视频；否则使用原剧片段 |

### 9.3 PySceneDetect 未检测到切换点

| 场景 | 处理 |
|------|------|
| 画面切换不明显（淡入淡出） | 降级到 LLM 边界 |
| 静态场景 | 降级到 LLM 边界 + 时间间隔切分 |

### 9.4 整体降级链

```
最优路径: canvas node matched + prompt extracted + seedance generated
    ↓ (canvas match fail)
降级 1:  keyframes + prompt extracted + seedance generated
    ↓ (prompt extraction fail)
降级 2:  keyframes + LLM generated prompt + seedance generated
    ↓ (seedance fail)
降级 3:  original video segment (完全使用原剧)
```

---

## 10. 实现计划

### 10.1 新增文件

| 文件 | 职责 | 预估行数 |
|------|------|---------|
| `skills/scene-detection/detect_scenes.py` | Stage 1b: PySceneDetect 集成 | ~200 |
| `skills/timeline-plan/generate_plan.py` | Stage 3: 剪辑计划生成（替代 match_to_canvas.py） | ~400 |
| `skills/timeline-plan/prompt_extractor.py` | Stage 3 子模块: Prompt 片段提取 | ~250 |
| `skills/timeline-plan/cut_fusion.py` | Stage 3 子模块: 剪辑点融合 | ~150 |
| `skills/timeline-plan/models.py` | Stage 3 数据模型（含 6+ dataclass、类型定义、默认值） | ~150 |
| `skills/video-assembly/assemble.py` | Stage 4: 视频组装（含编码统一化、音频归一化、时长适配） | ~400 |

### 10.2 修改文件

| 文件 | 改动 |
|------|------|
| `skills/canvas-storyboard/match_to_canvas.py` | 标记为 legacy（保留 LLM e2e matching 备用） |
| `skills/video-generation/generate_videos.py` | 标记为 legacy（保留 seedance 调用逻辑复用） |
| `skills/SKILL.md` | 更新 pipeline 文档 |

### 10.3 依赖新增

```
pip install scenedetect[opencv]  # PySceneDetect
```

---

## 11. 与旧方案的兼容

### 11.1 渐进式迁移

新旧方案可以**共存**。通过在 Stage 3 入口处增加一个 `--mode` 参数。两种模式的数据格式**互不兼容**——legacy 输出 storyboard markdown（被旧的 `generate_videos.py` 消费），timeline 输出 `TimelinePlan` JSON（被新的 `assemble.py` 消费）。

```bash
# 旧方案（文本匹配 → storyboard → generate_videos.py）
python3 skills/canvas-storyboard/match_to_canvas.py --mode legacy ...

# 新方案（时间轴驱动 → TimelinePlan → assemble.py）
python3 skills/timeline-plan/generate_plan.py --mode timeline ...
```

### 11.2 旧代码保留

- `match_to_canvas.py` → 保留不动，`--mode legacy` 时使用，输出 storyboard markdown
- `generate_videos.py` → 保留不动，仅 legacy 模式使用，消费 storyboard markdown
- `assemble.py` → 仅 timeline 模式使用，消费 `TimelinePlan` JSON
- `pipeline.py` → 保留不动（最早的 fuzzy_match 原型）

---

## 12. 质量保证

### 12.1 验证标准

| 指标 | 目标 | 验证方式 |
|------|------|---------|
| 剪辑点准确度 | ±0.2s 以内 | 对比 LLM 边界 → 最终 cut 的偏移量（放宽至 ±0.2s，约 6 帧@30fps，以覆盖淡入淡出等过渡场景） |
| 画布节点匹配率 | ≥80% 的 shot 匹配到节点 | 统计 TimelinePlan 中 `matched_node_id` 非空比例，同时跟踪 `match_confidence` 的均值和中位数 |
| seedance 生成成功率 | ≥90% | API 调用成功率 |
| seedance 时长偏差 | < 10% | `|seedance_duration - original_duration| / original_duration` |
| 最终视频完整性 | 无跳帧、无黑屏、无静音段 | 人工质检 + ffmpeg 自动化检测（见 §12.3） |

### 12.2 Debug 输出

Stage 3 产出 `timeline_plan.json` 和 `timeline_plan.md`（人类可读），包含：
- 每个 shot 的匹配置信度
- 采用的降级策略（`degradation_level`: 0-3）
- 最终剪辑点与原始 LLM 边界的偏移量
- PySceneDetect 检测到的所有切换点

### 12.3 自动化质量检测

```bash
# 黑帧检测（blackdetect filter）
ffmpeg -i final.mp4 -vf "blackdetect=d=0.5:pix_th=0.10" -f null -

# 静音检测（silencedetect filter）
ffmpeg -i final.mp4 -af "silencedetect=n=-50dB:d=1.0" -f null -

# 帧完整性校验
ffmpeg -v error -i final.mp4 -f null - 2>&1 | grep -c "error"
```

---

## 附录 A：关键文件路径

```
analyze_script_with_canvas/
├── skills/
│   ├── scene-detection/          # [新增] Stage 1b
│   │   └── detect_scenes.py
│   ├── timeline-plan/            # [新增] Stage 3
│   │   ├── generate_plan.py
│   │   ├── prompt_extractor.py
│   │   ├── cut_fusion.py
│   │   └── models.py
│   ├── video-assembly/           # [新增] Stage 4
│   │   └── assemble.py
│   ├── script-extraction/        # [不变] Stage 1
│   ├── script-rewriting/         # [不变] Stage 2
│   ├── canvas-storyboard/        # [legacy] 旧 Stage 3
│   └── video-generation/         # [legacy] 旧 Stage 4
│
├── lingolens/backend/agents/script_extraction/  # ⬆️ 数据源
│   ├── extractor.py              # VideoScriptExtractor
│   ├── prompt.py                 # build_multimodal_prompt()
│   ├── normalizer.py             # _enrich_lines_from_asr()
│   ├── builder.py                # build_output()
│   └── splitter.py               # split_long_lines()
│
└── lingolens/backend/core/models.py  # ScriptShot, ScriptLine 数据模型
```

## 附录 B：ASR 时间戳精度说明

数据来源：Azure Speech-to-Text API

```
utterance:
  start_time: 2830   # 毫秒（即 2.830s）
  end_time: 4190     # 毫秒（即 4.190s）
  text: "this ceremony is boring"
  speaker: "speaker_0"
  words: [
    { text: "this",    start: 2.83, end: 2.95 },
    { text: "ceremony", start: 3.02, end: 3.45 },
    { text: "is",      start: 3.50, end: 3.62 },
    { text: "boring",  start: 3.68, end: 4.19 }
  ]
```

经过 `normalizer.py::_enrich_lines_from_asr()` 处理后：
- `line.start_seconds = 2.83`（取 utterance.start_time / 1000）
- `line.end_seconds = 4.19`（取 utterance.end_time / 1000）
- 词级时间戳可通过 `asr_word_start_seconds` / `asr_word_end_seconds` 获取（帧精度）—— ⚠️ 字段存在但当前未被填充（始终为 None），见 §2.1 警示。需在 `normalizer.py` 中额外将 `words[]` 数据写入后方可启用
