# ASR Timeline-Driven Pipeline v2 — 方案说明

> 2026-05-29 | 5 集端到端验证通过

## 1. 核心思路

**原剧视频是时间轴基准，ASR 时间戳决定剪辑位置，画布节点降级为视觉资产库。**

旧方案试图将台词逐行文本匹配到画布节点，但画布节点含废片/花絮，匹配不可靠。新方案不再依赖画布节点作为视频源——直接用原剧完整视频，在 ASR 时间戳标注的位置做剪辑替换。

| 维度 | 旧方案 | 新方案 |
|------|--------|--------|
| 视频源 | 画布节点视频 | 原剧完整视频 |
| 剪辑位置 | 节点匹配决定 | ASR 时间戳 |
| 画布节点用途 | 视频源 + prompt 改写 | 仅参考图 + prompt 模板 |
| 无改写镜头 | 下载画布节点视频 | 直接截取原剧片段 |
| 废片处理 | LLM 猜测剔除 | 不依赖画布节点 |

## 2. Pipeline 架构

```
Stage 1 (lingolens):     视频 → Azure ASR → 多模态 LLM → VideoScriptOutput
Stage 1b (PySceneDetect): 视频 → 画面镜头切换点检测 → CutPoints
Stage 2 (shakespeare):    ScriptOutput → CEFR ReWriter → RewriteJSON
Stage 3 (timeline_plan):  剪辑计划生成 → TimelinePlan
Stage 4 (video_assembly): 视频组装 → final.mp4
```

### Stage 1: 剧本提取（复用）
多模态 LLM 观看压缩视频 + 阅读 ASR 转录，输出结构化剧本：shot 边界、台词行、角色、场景描述。每行台词携带精确的 ASR 时间戳（毫秒级 `start_seconds` / `end_seconds`）。

### Stage 1b: 场景检测（新增）
PySceneDetect 在原剧视频上做 Content-Aware 检测，产出画面级镜头切换点，辅助 Stage 3 的剪辑点精修。

### Stage 2: CEFR 改写（复用）
将剧本台词按 CEFR 等级（A2/B1/B2/C1）改写，产出 `{line_id, original, rewritten, start_seconds, end_seconds}` 格式的 JSON。

### Stage 3: 剪辑计划生成（重写）

这是整个新方案的核心。输入：Stage 1 的 script + Stage 2 的 rewrite + LibLib 画布节点数据。输出：TimelinePlan。

```
TimelinePlan:
  items[]:
    ┌─ source="seedance": 需生成新视频替换的片段
    │    shot_number, start_sec, end_sec,
    │    ref_images[], rewritten_prompt,
    │    seedance_duration, matched_node_id, match_confidence
    │
    └─ source="original": 直接截取原剧的片段
         shot_number, start_sec, end_sec
```

**关键步骤**：

1. **筛选改写行**：比较 `original ≠ rewritten`，确定哪些行需要重新生成
2. **LLM CoT 匹配**：将改写行与画布节点匹配。LLM 先识别每个节点 prompt 中的实际台词（区分对话 vs 标语/音效），再逐行匹配。3 次运行 + shuffled node order → voting 选最优，contiguity penalty 保证连续性
3. **连续分组 + merge-up**：改写行按时间连续性分组（gap > 5s → 新组），短组（< 4s）合并到最近邻，孤立短组回退 original
4. **Prompt 改写**：LLM 处理每个节点组——保留风格设定，只保留含改写台词的 scene 段落，精准替换对话，移除不含改写台词的 scene
5. **重叠处理**：seedance 项覆盖的时间范围内，移除 original 项（seedance 优先）

### Stage 4: 视频组装（重写）

按 TimelinePlan 执行：
- `source="original"`: ffmpeg 从原剧截取
- `source="seedance"`: seedance API 生成（duration=-1 智能时长），probe 实际时长，过长裁剪，过短接受
- 所有 segment 统一编码（libx264 + aac）+ 音频归一化（loudnorm）
- ffmpeg concat 拼接

## 3. LLM 调用点

共 2 个 LLM 调用，全部轻量：

### 3.1 行→节点匹配 (`canvas_matcher.py`)

```
System: 识别每个节点 prompt 中的实际英文对话（区分标语/音效/场景描述），
        然后将每行 ASR 台词匹配到包含该对话的节点。

Input:  {line_id, dialogue, speaker, shot_scene, start_seconds}
        + {node_id, prompt}

Output: {"mappings": [{"line_id": "p001_l001", "node_index": 0}, ...]}
```

3 次运行 + shuffled node order → 选匹配数最多的 run，contiguity score 作为 tiebreaker。跨 run 一致性计算置信度。

### 3.2 Prompt 改写 (`prompt_extractor.py`)

```
System: 重写视频生成 prompt。保留所有风格/画质设定。
        每个 scene：含改写台词 → 保留，只留对话时刻的视觉描述；
        不含改写台词 → 整段移除。替换原台词为改写台词。

Input:  原始 prompt + {speaker, original, rewritten}

Output: 改写后的 prompt 文本
```

简洁自然语言指令，不设规则框死 LLM。

## 4. 关键设计决策

### 4.1 为什么不用规则做 quote extraction？
LLM 能区分 `"This ceremony is boring."`（对话）和 `"CLASS OF 2026"`（横幅）、`"黑胶唱片划痕声"`（音效）、中文内心独白。规则做不到。

### 4.2 为什么做连续分组 + merge-up？
- **连续分组**：避免一个 seedance 视频跨越 15s 的 gap（含不改写内容）
- **merge-up**：解决种子时长限制（≥4s）。短组（< 4s）合并到最近邻，gap ≤5s 在视觉上自然连贯。孤立短组回退 original

### 4.3 为什么不用 padding？
用户明确禁止帧冻结。seedance 短了就让 concat 变短——无黑帧无缝拼接。

### 4.4 为什么 seedance duration = -1？
智能时长让 seedance 根据 prompt 内容自动决定最佳长度。prompt 裁剪后只剩对话时刻的视觉描述，seedance 应生成短而精准的 clip。

## 5. 端到端验证结果

5 集全部通过 Stage 1→2→3：

| Ep | Shots | 改写行 | Canvas 节点 | Seedance 项 | 状态 |
|----|-------|--------|------------|-------------|------|
| 1 | 20 | 22 | 36 (m2VuuIZfI) | 6 | ✅ |
| 2 | 8 | 10 | 36 (m2VuuIZfI) | 3 | ✅ |
| 3 | 9 | 7 | 9 (X4alzeDc6) | 1 | ✅ |
| 4 | 24 | 3 | 44 (UrIk7VdcV) | 1 | ✅ |
| 5 | 9 | 37 | 44 (UrIk7VdcV) | 5 | ✅ |

全部 16 个 seedance 项 ≥ 4s，匹配置信度 0.33–1.00。

## 6. 文件结构

```
skills/
├── scene_detection/          # Stage 1b: PySceneDetect
│   └── detect_scenes.py
├── timeline_plan/            # Stage 3: 核心新模块
│   ├── models.py             # 6 个 dataclass + MIN_SEEDANCE_DURATION
│   ├── canvas_matcher.py     # LLM CoT 匹配 + voting
│   ├── prompt_extractor.py   # LLM prompt 改写
│   ├── cut_fusion.py         # ScriptShot + PySceneDetect 融合
│   └── generate_plan.py      # Stage 3 编排器 + CLI
├── video_assembly/           # Stage 4: 视频组装
│   └── assemble.py           # ffmpeg trim + seedance + concat
├── script-extraction/        # Stage 1: 复用 lingolens
├── script-rewriting/         # Stage 2: 复用 shakespeare
└── SKILL.md                  # Timeline mode v2.0 文档

tests/                         # 46 单元测试
docs/superpowers/
├── specs/2026-05-29-asr-timeline-pipeline-design.md  # 完整设计文档
└── plans/2026-05-29-asr-timeline-pipeline.md         # 实现计划
```
