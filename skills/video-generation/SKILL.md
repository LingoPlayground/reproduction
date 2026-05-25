---
name: video-generation
description: "Stage 4: 基于改写 storyboard 调用 seedance 生成新视频 + 下载原视频 + 拼接全片。Use when: 'generate video', 'seedance', '生成视频', 'concat', '拼接'."
metadata:
  requires:
    bins: ["python3", "ffmpeg"]
    skills: ["analyze-script-with-canvas", "canvas-storyboard"]
---

# video-generation — 视频生成与拼接 (Stage 4)

## 概述

基于 Stage 3 产出的改写 storyboard，对 prompt 有改动的节点调用 **seedance 2.0 fast** 重新生成视频，下载其余原视频，拼接成完整剧集。

```
storyboard (含替换后 prompt + 参考图 URL)
    │
    ├─ 有改动的节点 → seedance 2.0 fast 生成新视频
    │   └─ 上传参考图 → 提交生成 → 等待完成 → 下载
    │
    ├─ 无改动的节点 → 从画布 URL 下载原视频
    │
    └─ 按 shot 顺序拼接 → 全剧 mp4
```

## 前置条件

| 依赖 | 说明 |
|---|---|
| Python 3.x | 运行环境 |
| `ffmpeg` | 视频拼接 |
| `AQINFO_SEEDANCE_API_KEY` | 从 `~/workspace/lingolens/backend/.env` 加载 |
| 画布本地缓存 | `canvas_data.json`（避免 API 波动） |

## CLI 用法

```bash
# 生成单集（仅重生成有改动的节点）
python3 skills/video-generation/generate_videos.py \
  --storyboard storyboards/storyboard_ep1_B2.md \
  --canvas /path/to/canvas_data.json \
  --output generated/ep1_B2.mp4 \
  --script /path/to/episode1_script.json
```

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `--storyboard` | ✅ | Stage 3 产出的改写 storyboard markdown |
| `--canvas` | ✅ | 画布本地缓存 JSON |
| `--output` | ✅ | 输出 mp4 路径 |
| `--script` | ✅ | 原始剧本 JSON（用于获取 node→video URL 映射） |
| `--only-shot` | ❌ | 只生成指定镜头（如 `--only-shot 1`，用于测试） |
| `--dry-run` | ❌ | 仅列出待生成列表，不实际调用 seedance |

## 生成逻辑

1. 解析 storyboard → 提取每个节点的 prompt 替换状态（🔄 替换 / 📝 原版）
2. 对 `🔄 替换` 的节点：**本地下载参考图** → 上传 seedance → 生成新视频
3. 对 `📝 原版` 的节点：从画布 URL 下载原视频
4. 按 storyboard 中的出现顺序拼接

> ⚠️ liblib 参考图 URL 可能被 seedance 服务器 403 拒绝——脚本先将图片下载到本地再上传。

## seedance 生成参数

| 参数 | 值 |
|---|---|
| 模型 | `doubao-seedance-2-0-fast-260128` |
| 比例 | `9:16` |
| 分辨率 | `720p` |
| 时长 | 沿用原节点设定 |
| 音频 | `generate_audio=True` |
| 参考图 | 从 storyboard 提取（≤9 张） |

## 已知限制

- seedance 生成约 3-5 分钟/节点（排队+生成）
- 部分 liblib 原视频 URL 可能失效（404），需从画布实时拉取
- 拼接后的 mp4 不含字幕（台词信息在 prompt 中，非渲染字幕）
