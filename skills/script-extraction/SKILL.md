---
name: script-extraction
description: "Stage 1: 从 AI 生成视频中提取结构化剧本。输入原始视频 + ASR 转录 → 多模态 LLM 分析 → 输出含 shots/lines/characters/timing/场景描述的结构化剧本 JSON。Use when user says: 'extract script from video', '提取剧本', '视频转剧本', 'generate screenplay from video'."
metadata:
  requires:
    bins: ["python3", "ffmpeg"]
    skills: ["analyze-script-with-canvas"]
---

# script-extraction — 视频剧本提取 (Stage 1)

## 概述

从 AI 生成的影视视频中提取结构化剧本。结合 **ASR 语音转录**和**多模态 LLM**（视频理解），产出包含镜头、台词、角色、时间戳、场景描述的完整剧本。

**核心设计**：LLM 只引用 ASR utterance 的 index，不产出文本或时间戳——软件层从 ASR 数据回填，消除 LLM 幻觉和时间戳漂移。

## 前置条件

### 必须安装

| 依赖 | 说明 |
|---|---|
| Python 3.11+ | 运行环境 |
| `ffmpeg` | 视频压缩（可选但建议：`brew install ffmpeg`） |
| `~/workspace/lingolens/` | 可导入 `backend.agents.script_extraction.VideoScriptExtractor` |
| 多模态 LLM 服务 | 需支持视频输入的 LLM（如 GLM-5.1 via SiliconFlow） |
| ASR 服务 | 需提供 whisper 或云端 ASR 的转录结果 |

### lingolens 导入验证

```bash
cd ~/workspace/lingolens
python3 -c "from backend.agents.script_extraction import VideoScriptExtractor; print('OK')"
```

## 输入格式

### ASR 转录 JSON (`utterances`)

```json
[
  {
    "speaker": "speaker_0",
    "start_time": 3000,
    "end_time": 5200,
    "text": "Hey, over here!",
    "emotion": "happy",
    "words": [
      {"text": "Hey,", "start": 3.0, "end": 3.4},
      {"text": "over", "start": 3.5, "end": 3.8},
      {"text": "here!", "start": 3.9, "end": 5.2}
    ]
  }
]
```

| 字段 | 类型 | 必填 | 说明 |
|---|---|---|---|
| `speaker` | str | ✅ | Diarization speaker ID（如 `speaker_0`） |
| `start_time` | int/float | ✅ | 开始时间（**毫秒**） |
| `end_time` | int/float | ✅ | 结束时间（**毫秒**） |
| `text` | str | ✅ | 转录文本 |
| `emotion` | str | ❌ | 情绪标签（可选，用于 LLM 上下文） |
| `words` | list | ❌ | 词级时间戳（可选但建议：提高长句切分质量） |

## CLI 用法

```bash
python3 skills/script-extraction/extract_script.py \
  --video /path/to/video.mp4 \
  --utterances /path/to/asr.json \
  --output ep1_script.json \
  --temp-dir runs/
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--video` | ✅ | — | 原始视频文件路径 |
| `--utterances` | ✅ | — | ASR 转录 JSON 文件路径 |
| `--output` | ❌ | `script_output.json` | 输出 JSON 路径 |
| `--temp-dir` | ❌ | `runs/` | 调试输出目录 |

## 输出格式

输出为 `VideoScriptOutput` 的 JSON 序列化，兼容 Stage 2 的 `ScriptInput` 格式：

```json
{
  "meta": {
    "characters": [
      {
        "role_id": "role_donny",
        "speaker_name": "Donny",
        "role": "protagonist",
        "appearance": "Young man in graduation gown, dark hair..."
      }
    ],
    "locations": [
      {"description": "Crowded graduation ceremony hall..."}
    ],
    "summary": "Donny discovers multiple job rejections during his graduation...",
    "speaker_mappings": [
      {"speaker_id": "speaker_0", "role_id": "role_donny", "character_name": "Donny", "confidence": 1.0}
    ]
  },
  "script": {
    "title": "Video Remixed Scene",
    "total_duration_seconds": 86.7,
    "shots": [
      {
        "shot_number": 1,
        "start_seconds": 0.0,
        "end_seconds": 26.7,
        "location": "Graduation Hall",
        "scene_description": "Donny stands in the crowded graduation celebration...",
        "shot_type": "medium shot",
        "camera_movement": "dolly back",
        "mood": "celebratory turning to shock",
        "lines": [
          {
            "line_id": "p001_l001",
            "role_id": "role_donny",
            "speaker": "Donny",
            "dialogue": "this ceremony is boring",
            "start_seconds": 2.83,
            "end_seconds": 4.19,
            "speech_mode": "dialogue"
          }
        ]
      }
    ]
  }
}
```

## 关键字段说明

| 字段 | 说明 |
|---|---|
| `meta.characters[].role_id` | 角色唯一 ID，Stage 2 使用此字段做角色一致性 |
| `script.shots[].lines[].line_id` | 格式 `p{shot}_l{line}`，Stage 2/3 用此 ID 做跨阶段匹配 |
| `script.shots[].lines[].speech_mode` | `dialogue` / `voiceover` / `narration` |
| `meta.speaker_mappings` | ASR speaker ID → 角色 ID 的映射表 |

## 错误处理

| 错误 | 原因 | 处理 |
|---|---|---|
| `RuntimeError: Multimodal LLM not enabled` | LLM 未配置或不可用 | 检查 LLM 服务配置 |
| `RuntimeError: Structural conflict` | ASR 同一段被分配给多个 speaker | 检查 ASR 转录质量 |
| Blocking conflict 日志 | Non-blocking 问题（如 speaker 不一致） | 查看 `temp-dir/llm/` 下的 trace 文件 |

## 已知限制

- **时间戳精度**：对话时间戳来自 ASR，精度取决于 ASR 质量
- **Speaker 识别**：LLM 视频视觉识别为主，ASR diarization 为辅
- **长视频**：建议用 ffmpeg 预压缩（脚本自动处理），否则 LLM 延迟显著
