# Canvas Node Matching & Prompt Rewriting — 完整上下文信息

> 仅供专家参考。不含方案建议，仅陈述事实。

## 1. 问题定义

我们有原始剧本（含 shot 描述、台词、ASR 精确时间戳）。对其中部分台词做了 CEFR B2 改写。希望复用 LibLib 画布节点（原视频制作时使用的 AI 生成项目，含角色参考图、场景描述、生成 prompt）作为视觉资产，为改写台词生成新视频并替换回原剧。

**核心矛盾**：
- 画布节点可能是废片（B-roll/blooper），prompt 可能覆盖多个 shot
- 改写台词可能只有 2-3s，但 seedance 最低 4s，必然要带入不改写内容
- 找到合适的节点、正确改写 prompt、处理生成视频覆盖范围大于改写台词的问题

---

## 2. 输入数据格式

### 2.1 Stage 1: Script Extraction Output

来源：lingolens `VideoScriptExtractor`（Azure ASR + 多模态 LLM doubao-seed）

```json
// VideoScriptOutput → ScriptOutput
{
  "script": {
    "shots": [{
      "shot_number": 1,
      "start_seconds": 0.0,        // 多模态 LLM 划定的镜头边界（秒级精度）
      "end_seconds": 3.0,
      "scene_description": "Donny stands in the crowded graduation celebration...",
      "lines": [{
        "line_id": "p001_l001",
        "speaker": "Donny Li",
        "dialogue": "this ceremony is boring",
        "start_seconds": 2.83,     // ASR 时间戳（毫秒级精度）
        "end_seconds": 4.19
      }]
    }]
  }
}
```

**时间体系差异**：
- `ScriptShot.start_seconds/end_seconds`：多模态 LLM 分析视频帧得出的逻辑镜头边界，秒级精度
- `ScriptLine.start_seconds/end_seconds`：Azure ASR 的 utterance 时间戳，毫秒级精度
- 两者不对齐（例子：Shot 1 覆盖 0.0–3.0s，但其台词 p001_l001 实际发生在 2.83–4.19s）

### 2.2 Stage 2: Rewrite Output

来源：shakespeare `FullRewriter`（LLM CEFR 分级改写）

```json
{
  "level": "B2",
  "lines": [{
    "line_id": "p001_l001",
    "shot_number": 1,
    "speaker": "Donny",
    "original": "this ceremony is boring",
    "rewritten": "This ceremony is not entertaining at all.",
    "start_seconds": 2.83,
    "end_seconds": 4.19,
    "shot_scene": "Donny stands in the crowded graduation celebration, holding a glass of champagne, checking his phone..."
  }]
}
```

### 2.3 LibLib Canvas Node Data

来源：`https://api.liblib.tv/api/canvas/project/share/detail?shareId={id}`

```json
[{
  "nodeId": "13126e5a-e6f4-4194-b17d-214ebe8f65ea",
  "name": "视频节点 2 - 副本",
  "prompt": "美式情景喜剧，真实短剧，柔光雾化，画面通透...镜头 1：三层景深与霸总转身...\"This ceremony is boring.\"（这派对太无聊了。）...镜头 2：真实的破防（面部特写）...镜头3：眼部超大特写...",
  "video_url": "https://libtv-res.liblib.art/...",
  "reference_images": ["https://libtv-res.liblib.art/...", ...]
}]
```

**Prompt 结构特征**：
- 开头：全局视觉风格设定（"美式情景喜剧，柔光雾化，8k，超高清，电影级布光"）
- 中间：分镜描述（"镜头 1：..."、"镜头 2：..."），混合中英文
- 英文台词通常出现在引号（`""`）内，也可能直接嵌入中文描述中无引号
- 中文描述含角色名、景别、运镜、表情、动作
- 可能包含 `{{Portrait N}}` 占位符（LibLib 角色参考图引用）

---

## 3. 真实数据实例（Episode 1）

### 3.1 改写台词

| line_id | Speaker | 原台词 | 改写 B2 | ASR 时间 |
|---------|---------|--------|---------|---------|
| p001_l001 | Donny | this ceremony is boring | This ceremony is not entertaining at all. | 2.83–4.19s |
| p001_l002 | Donny | let's see who wants me | Let's see who is desperate to hire me. | 5.31–6.43s |
| p001_l003 | Donny | no no no | No, no, no, this can't be. | 17.47–18.27s |
| p001_l004 | Donny | they all refused me | Every single one of them rejected me. | 18.83–20.03s |
| p001_l005 | Donny | are they going crazy Donnie | Have they gone completely crazy, Donny? | 21.15–29.55s |
| p003_l001 | Donny | what going on? | What's going on? | 32.75–33.55s |
| ... | ... | ... | ... | ... |

### 3.2 画布节点覆盖（人工标注 ground truth）

节点 `13126e5a`（视频节点 2 - 副本）覆盖 Shot 1 全部 5 行台词：
```
prompt 858 chars:
  美式情景喜剧，真实短剧，柔光雾化...8k，超高清...
  镜头 1：三层景深与霸总转身
    ..."This ceremony is boring."..."Let's see who wants me。...
  镜头 2：真实的破防（面部特写）
    画面与动作：面部特写，固定机位。男主刚 Wink 完...（无英文引号台词）
  镜头3：眼部超大特写...
```

**关键事实**：
- p001_l001-l002 的台词在 prompt 中作为英文引号内容出现 ✅
- p001_l003-l005 的台词在 prompt 中**没有英文引号**，仅在镜头 2 的中文描述中隐含（"真实的破防"、"面部特写"）
- 一个画布节点可覆盖同一 shot 中时间不连续的所有台词（2.83s–4.19s 和 17.47s–29.55s，间隔 13s）

### 3.3 当前 Pipeline 的处理结果（实际跑出的）

**LLM 匹配**：p001_l001-l005 全部匹配到节点 `13126e5a`，confidence=1.0 ✅

**连续分组**：
- Group A：p001_l001-l002（2.83–6.43s，3.6s）— **< 4s，无法 seedance**
- Group B：p001_l003-l005（17.47–29.55s，12.1s）— **>= 4s ✅**

**Prompt 改写结果（Group B）**：
- LLM 改写失败：原始 prompt 中没有 "no no no" 的英文引号 → LLM 找不到替换位置
- 验证失败 → 降级到 L2 fallback
- 产出 prompt：`"A cinematic scene\nDonny says: 'No, no, no, this can't be.'..."`（164 chars）
- **丢失了原始 prompt 的 858 chars 视觉质量**（风格设定、镜头描述、运镜、情绪等全部丢失）

**Group A 处理结果**：
- `_extend_short_group` 尝试在同节点内找临近台词扩展至 4s → 失败（最近邻在 17.5s，gap=11s > 5s）
- 回退 original → 这两行改写台词不会被生成

### 3.4 其他成功案例

Shot 3（p003_l001-l005，32.75–43.54s）匹配到节点 `1a1f4741`：
- 该节点的 prompt 包含所有 5 行台词的英文引号
- LLM 改写成功：1091 chars 原 prompt → 1141 chars 改写 prompt
- "what going on?" → "What's going on?" 等全部替换 ✅

Shot 5+6（p005_l001-l002 + p006_l001-l004，59.0–71.2s）匹配到节点 `0e19c709`：
- LLM 改写成功：1461 chars → 1553 chars
- 父亲台词被连锁改写（"100 million dollars" → "one hundred million dollars"）

---

## 4. 当前技术方案的架构

### 4.1 LLM 调用点

**调用 1：行→节点匹配**（`canvas_matcher.py::match_lines_to_nodes`）
- 输入：所有改写行（line_id, dialogue, speaker, shot_scene, start/end） + 所有画布节点（prompt 截断到 3000 chars）
- Prompt：CoT — 先识别每个 node prompt 中的实际英文对话，再逐行匹配
- 3 次运行 + shuffled node order → Best-Run Selection（contiguity score + tie-breaker on confidence）
- 输出：`{node_id: [line_ids]}` + `{line_id: confidence}`

**调用 2：Prompt 改写**（`prompt_extractor.py::extract_and_rewrite_prompt`）
- 输入：节点原始 prompt + 改写行（speaker, original, rewritten） + scene_description
- Prompt：保留风格设定，只保留含改写台词的 scene，替换对话，移除不含改写台词的 scene
- 验证：检查改写台词是否出现在输出中 → 不在则 fallback 到 `_generate_prompt_from_scene`

### 4.2 关键算法

**连续分组 + merge-up**（`generate_plan.py::_split_contiguous`）
- 按 start_seconds 排序 → gap > 5s 拆分
- 短组（< 4s）检查 gap ≤ 5s → 合并；gap > 5s → 保持为孤立组

**扩展短组**（`generate_plan.py::_extend_short_group`）
- 在同节点内向前/后搜索临近台词（含不改写台词）
- 不改写台词的 `rewritten = original`（在 prompt 中保留原文）
- 扩展至 ≥ 4s 或失败 → caller 回退 original

**Timeline 后处理**（`generate_plan.py::_finalize_timeline`）
- overlap 切分（original 按 seedance 范围切割）
- 微小碎片清理（< 0.5s 并入相邻 seedance）
- gap 填充 + 相邻 original 合并
- 边界 snap（ASR 吸附到 PySceneDetect cut points）

### 4.3 降级层级

| Level | 含义 | 触发条件 |
|-------|------|---------|
| 0 | 最优 | 节点匹配 + 参考图 + prompt 改写全部成功 |
| 1 | 无参考图 | 节点匹配成功但 reference_images 为空 |
| 2 | 无节点匹配 / prompt 改写失败 | LLM 匹配失败 或 改写后的 prompt 不含改写台词 |

---

## 5. 已知约束和边界条件

1. **seedance 最小时长 4s**：`MIN_SEEDANCE_DURATION = 4.0`
2. **seedance 最大时长约 30s**：超长镜头需拆分或降级
3. **画布节点 prompt 可能含有**：
   - 多个场景/镜头描述（一个节点覆盖多个 shot）
   - 废片/B-roll 内容（与改写台词不相关的其他对话）
   - 非台词的引号内容（横幅文字如 "CLASS OF 2026"、音效描述如 "黑胶唱片划痕声"）
   - 中英文混合（中文视觉描述 + 英文台词）
   - 不完整场景描述（如 "男主是 " 后半截缺失）
4. **改写台词可能**：
   - 短于 4s → 需要借入临近内容
   - 时间不连续（同一 shot 内被 gap 分隔）
   - 孤立（单行、无同节点临近行）
5. **时间对齐**：
   - ASR 时间戳（毫秒级）≠ LLM shot 边界（秒级）
   - PySceneDetect 提供画面级 cut point（依赖 Stage 1b 是否运行）

---

## 6. 涉及的文件

```
skills/timeline_plan/
  generate_plan.py      — Stage 3 编排器（匹配→分组→TimelinePlan 输出）
  canvas_matcher.py     — LLM CoT 行→节点匹配 + voting
  prompt_extractor.py   — LLM prompt 改写 + 验证 + fallback
  cut_fusion.py         — ScriptShot + PySceneDetect 时间融合
  models.py             — 数据模型 + 常量

skills/canvas-storyboard/
  match_to_canvas.py    — 旧版 LLM e2e 匹配（含 replace_dialogue_in_prompt 函数）
```
