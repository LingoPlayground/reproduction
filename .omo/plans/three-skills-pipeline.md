# Three-Skill Pipeline Plan

## 管线总览

```
原始视频 + ASR 转录
    │
    ▼ Skill 1: script-extraction
VideoScriptExtractor (lingolens)
    │  输出: ep1_script.json (ScriptInput 格式)
    ▼
    │
    ├─→ Skill 2: script-rewriting ──┐
    │   FullRewriter (shakespeare)   │
    │   输出: ep1_A2.json            │
    │         ep1_B1.json            │
    │         ep1_B2.json            │
    │         ep1_C1.json            │
    │   (每等级独立台词 JSON)          │
    │                                │
    └─→                               │
                                      ▼
                              Skill 3: canvas-storyboard
                              对每个等级 JSON 独立匹配画布节点
                              输出: storyboard_ep1_A2.md
                                    storyboard_ep1_B1.md
                                    storyboard_ep1_B2.md
                                    storyboard_ep1_C1.md
```

## 目录结构

```
/Users/hupan/workspace/analyze_script_with_canvas/
├── skills/
│   ├── script-extraction/
│   │   ├── SKILL.md              # 视频剧本提取 skill 定义
│   │   └── extract_script.py     # 包装脚本（调用 lingolens VideoScriptExtractor）
│   ├── script-rewriting/
│   │   ├── SKILL.md              # CEFR 分级改写 skill 定义
│   │   └── rewrite_script.py     # 包装脚本（调用 shakespeare FullRewriter）
│   └── canvas-storyboard/
│       ├── SKILL.md              # 画布匹配分镜 skill 定义
│       └── match_to_canvas.py    # 改写台词 → 画布节点匹配脚本（需新建）
└── pipeline.py                   # 现有单文件（将被 skills 引用）
```

## Skill 1: script-extraction（视频剧本提取）

### 职责
输入原始视频 + ASR 转录 → 输出结构化剧本 JSON

### 输入
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--video` | path | ✅ | 原始视频文件路径 |
| `--utterances` | path | ✅ | ASR 转录 JSON 文件路径 |
| `--output` | path | ❌ | 输出 JSON 路径（默认 `script_output.json`） |
| `--temp-dir` | path | ❌ | 调试输出目录（默认 `runs/`） |

ASR JSON 格式：
```json
[{"speaker":"speaker_0","start_time":3000,"end_time":5200,"text":"Hey!",
  "emotion":"happy","words":[{"text":"Hey!","start":3.0,"end":5.2}]}]
```

### 依赖
- **Hard**: multimodal LLM（需支持视频输入的 LLM 服务，如 GLM-5.1）
- **Hard**: `~/workspace/lingolens/` 项目可导入（`backend.agents.script_extraction.VideoScriptExtractor`）
- **Optional**: ffmpeg（视频压缩优化）

### 输出
- `VideoScriptOutput` → 序列化为 JSON
- 格式兼容 `ScriptInput`，可直接喂给 Step 2

### 实现要点
- `extract_script.py` 做薄包装：CLI 参数解析 → 初始化 LLM → 调用 extractor → 写 JSON
- 需处理 `BaseLLMService` 的实例化（lingolens 依赖注入）
- 需处理 ASR 文件格式校验

---

## Skill 2: script-rewriting（CEFR 分级改写）

### 职责
输入剧本 JSON → 按 CEFR 等级产出独立改写台词文件

### 关键设计：每等级独立输出

改写引擎（FullRewriter）本身是**按等级独立调用 LLM** 的（每等级一次 LLM 调用）。因此输出自然应该分等级独立保存：
- `ep1_A2.json` — A2 等级改写台词
- `ep1_B1.json` — B1 等级改写台词
- `ep1_B2.json` — B2 等级改写台词
- `ep1_C1.json` — C1 等级改写台词

每个文件包含对应等级的台词列表，然后 Step 3 各自独立匹配画布。

### 输入
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--script` | path | ✅ | ScriptInput 格式的剧本 JSON |
| `--levels` | str | ❌ | 目标 CEFR 等级，逗号分隔（默认 `A2,B1,B2,C1`） |
| `--output-dir` | path | ❌ | 输出目录（默认当前目录） |
| `--output-prefix` | str | ❌ | 输出文件前缀（默认 `rewrite`） |

### 输出格式（每等级一个文件）

`{prefix}_{level}.json`：
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
      "shot_scene": "毕业典礼现场..."
    }
  ],
  "quality": {
    "cefr_precision": 0.85,
    "cefr_recall": 0.12
  }
}
```

> **重要**：输出格式需**回填 shot 信息**（shot_number, shot_scene, start_seconds, end_seconds）。shakespeare 现有输出不包含这些字段，`rewrite_script.py` 需要在输出时从原始 ScriptInput 中补全。

### 依赖
- **Hard**: `~/workspace/shakespeare/` 项目可导入
- **Hard**: OpenAI-compatible API（`.env` 配置 `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `LLM_MODEL`）
- **Hard**: `data/CuriosSea分级知识点_cleaned.xlsx` 存在于 shakespeare 项目 data 目录
- **Hidden deps**: `sentence-transformers`, `spacy`+`en_core_web_md`, `openpyxl`, `numpy` — 需在 SKILL.md 中明确声明
- **First-run**: 需下载模型（~120MB）

### 实现要点
- `rewrite_script.py` 包装 shakespeare 的 `FullRewriter`
- 支持按等级拆分输出
- 回填 shot 上下文到输出中
- 处理 `original_dialogue` 为空的问题（从输入按 `line_id` 补回）

---

## Skill 3: canvas-storyboard（画布匹配分镜）

### 职责
输入一个等级的改写台词 JSON + 画布数据 → 匹配画布视频节点 → 输出分镜故事板

### 关键设计：与改写完全解耦

每个 CEFR 等级的改写台词 JSON **独立**调用画布匹配。这样：
- Step 2（LLM 改写）和 Step 3（算法匹配）完全独立
- 可以只匹配某个等级（如只关心 A2 分镜）
- 每个等级产出独立的故事板 markdown

### 输入
| 参数 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `--rewrite` | path | ✅ | 单等级的改写台词 JSON（Step 2 产出） |
| `--canvas` | path/str | ✅ | 画布数据源：shareId（如 `m2VuuIZfI`）或本地 JSON 文件路径 |
| `--output` | path | ❌ | 输出 markdown 路径（默认 `storyboard_{level}.md`） |
| `--episode` | str | ❌ | 剧集过滤标识（如 `ep1`），解决跨集污染 |
| `--min-score` | int | ❌ | 匹配最低分数阈值（默认 40） |

### 匹配逻辑
复用 `pipeline.py` 的 `fuzzy_match_score` + `find_best_match` + `format_node`。
- 改写台词 `rewritten` → 匹配画布视频节点（type=3）的 `prompt` 字段
- 匹配算法：标准化子串 → 滑窗 → 去前导词 → 词命中率
- 同分数取 `updatedAtMs` 最新
- `--episode` 用于按节点名称/prompt 内容过滤跨集节点

### 输出格式（每个镜头一张表格）

```markdown
# Episode 1 — A2 等级分镜故事板

## 镜头 1 (26.7s) — Donny
*Donny stands in the crowded graduation celebration...*

| line_id | 台词 | 原台词 | A2 改写 | 匹配画布节点 | 得分 |
|---------|------|--------|---------|-------------|------|
| p001_l001 | 💬 Donny | this ceremony is boring | This ceremony is boring. | ✅ 视频节点 2 - 副本 | 100 |
| p001_l002 | 💬 Donny | let's see who wants me | Let's see who wants me! | *(同上)* | — |
| p001_l003 | 💬 Donny | no no no | No, no, no! | ✅ 视频节点 7 | 80 |

📸 **参考图**:
- 视频节点 2: [4 张参考图 URL 列表]
- 视频节点 7: [2 张参考图 URL 列表]

📝 **生视频 Prompt** (视频节点 2):
```
美式情景喜剧，真实短剧，柔光雾化...
```

📝 **生视频 Prompt** (视频节点 7):
```
美式情景喜剧，柔光雾化...
```

---
## 镜头 2 (8.3s) — Donny, Lily
...
```

### 依赖
- **Hard**: `pipeline.py`（同项目内，复用 `fuzzy_match_score`, `find_best_match`, `format_node`）
- **Hard**: LibLib Canvas API 可访问（需中国 IP）
- 无 LLM 依赖，纯算法

### 实现要点
- **新建 `match_to_canvas.py`**（对应 shakespeare SKILL.md 引用的缺失脚本）
- 输入是**单等级** rewrite JSON，不需要"4行 CEFR 对比表"
- 输出是**逐镜头的台词-匹配-参考图-Prompt 表格**
- Episode 过滤逻辑需要实现（按 `--episode` 参数，用节点名/prompt 关键词匹配）
- 相邻台词匹配到同一节点时标记 `*(同上)*`

---

## 实现顺序

1. **写 Plan 审查** ← 当前步骤
2. **创建目录结构**：`skills/script-extraction/`, `skills/script-rewriting/`, `skills/canvas-storyboard/`
3. **新建 `skills/canvas-storyboard/match_to_canvas.py`**（目前缺失的核心脚本）
4. **写 `skills/canvas-storyboard/SKILL.md`**（依赖最少，逻辑最独立）
5. **写 `skills/script-rewriting/SKILL.md`** + `rewrite_script.py`
6. **写 `skills/script-extraction/SKILL.md`** + `extract_script.py`（依赖最多，放最后）

---

## 已识别风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| shakespeare `pyproject.toml` 缺少依赖声明 | Step 2 安装失败 | SKILL.md 中显式列出全部依赖 + `pip install` 命令 |
| lingolens `BaseLLMService` 强依赖注入 | Step 1 script 需了解 lingolens 架构 | 在 `extract_script.py` 中做适配层 |
| canvas 画布跨集污染 | Step 3 误匹配非本集节点 | `--episode` 过滤 + prompt 关键词匹配 |
| rewrite JSON 缺 shot 上下文 | Step 3 无法生成分镜结构 | `rewrite_script.py` 输出时回填 shot 字段 |
| 多模态 LLM 视频限制 | Step 1 大视频可能超限 | ffmpeg 压缩（360p/2fps）+ 分段处理 |
