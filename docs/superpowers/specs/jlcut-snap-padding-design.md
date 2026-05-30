# Stage 3/4: 音画精确剪辑 — Snap、Padding、J/L Cut 设计

## 1. 问题背景

Pipeline v2 的核心是用 seedance 生成的视频替换原剧中改写台词对应的片段。剪辑边界由 ASR 时间戳决定。但存在两个精度问题：

1. **ASR 边界 ≠ 画面切换点**：ASR 是音频级别的，PySceneDetect 是像素级别的，两者之间有 0.1s–1.5s 的偏差
2. **导演剪辑语法 J/L Cut**：原剧中存在声音先于画面（J-Cut）或画面先于声音（L-Cut）的剪辑手法

一刀切用 ASR 时间戳替换，会导致视觉闪帧或破坏原剧的镜头语言。

## 2. 核心原则

```
ASR 时间戳 = 台词锚点（不可动摇）
PySceneDetect = 画面边界参考（向它吸附）

当两者接近时 → Snap（吸附到画面边界）
当两者有显著偏差 → 保留偏差 → 触发 J/L Cut（音画分离）
当无画面边界参考 → Padding（向外扩展防切音）
```

## 3. 三层策略

### 层 1：Shot-Level Replacement（全镜头替换）

**触发条件**：改写段覆盖某个 PySceneDetect 镜头 **> 60%** 的时长

**策略**：不再精修边界，直接将整个镜头交给 seedance 重新生成。

```
PySceneDetect cuts:    [10.0s] ───────────── [25.0s]
改写 ASR:                   [11.0s ─── 22.0s]
覆盖率:               (22-11)/(25-10) = 73% > 60%
→ seedance 覆盖:        [10.0s ───────────── 25.0s]  全镜头替换
```

**优点**：所有拼接点都在原剧本身的"硬切点"上，零跳切风险。

### 层 2：Snap & Padding（吸附与留白）

**触发条件**：ASR 边界与最近 PySceneDetect cut 的距离 ≤ `SNAP_WINDOW` (0.8s)

```
场景 A（向后吸附）：
  ASR start = 10.3s, cut = 10.0s, gap = 0.3s ≤ 0.8s
  → start 吸附到 10.0s（从镜头切换的第一帧开始替换，消除 0.3s 闪现）

场景 B（向前吸附）：
  ASR end = 25.2s, cut = 25.7s, gap = 0.5s ≤ 0.8s
  → end 吸附到 25.7s（让 AI 生成无言反应直到镜头切走）

场景 C（安全留白）：
  四周 0.8s 内无 cut → audio_padding ±0.15s（防 ffmpeg 切音吞字）
```

### 层 3：J/L Cut（音画解耦）

**触发条件**：ASR 边界与 cut 的距离在 `MIN_SPLIT` (0.3s) 到 `MAX_SPLIT` (1.5s) 之间

```
J-Cut（音频先行）:
  ASR start = 10.0s, cut = 11.0s, gap = 1.0s ∈ [0.3, 1.5]
  → audio_start = 10.0s（立刻切入改写台词音频）
  → video_start = 11.0s（保持原剧画面直到镜头自然切换）
  → edit_type = "j_cut"

L-Cut（画面先行）:
  ASR start = 21.0s, cut = 20.0s, gap = 1.0s
  → video_start = 20.0s（提前切入 seedance 新画面）
  → audio_start = 21.0s（直到台词开始才切音频）
  → edit_type = "l_cut"

Straight Cut:
  gap < 0.3s → 直接对齐到 ASR（差距太小，强制合一）
  gap > 1.5s → 无关联，按独立长镜头处理
```

## 4. 数据模型升级

`TimelinePlanItem` 增加音画分离字段。为了向后兼容，保留 `start_sec` / `end_sec` 作为默认值。

```python
@dataclass
class TimelinePlanItem:
    # 向后兼容字段（straight cut 时 video 和 audio 相同）
    start_sec: float
    end_sec: float

    # 音画解耦字段（j_cut / l_cut 时与 start_sec/end_sec 不同）
    video_start_sec: Optional[float] = None
    video_end_sec: Optional[float] = None
    audio_start_sec: Optional[float] = None
    audio_end_sec: Optional[float] = None

    edit_type: str = "straight"  # "straight" | "j_cut" | "l_cut"

    @property
    def effective_video_start(self) -> float:
        return self.video_start_sec if self.video_start_sec is not None else self.start_sec

    @property
    def effective_audio_start(self) -> float:
        return self.audio_start_sec if self.audio_start_sec is not None else self.start_sec
    # ... 同理 end
```

## 5. Stage 3 决策逻辑

放在 `generate_plan.py` 的 `_snap_boundaries()` 函数中：

```python
SNAP_WINDOW = 0.8       # 吸附阈值
AUDIO_PADDING = 0.15    # 安全留白
MIN_SPLIT = 0.3         # J/L Cut 最小错位
MAX_SPLIT = 1.5         # J/L Cut 最大错位
SHOT_COVERAGE = 0.6     # 全镜头替换覆盖率阈值
JLCUT_ENABLED = False   # 开关：当前默认关闭，等 Stage 4 支持

def _snap_boundaries(start, end, cut_points):
    """返回 (video_start, video_end, audio_start, audio_end, edit_type)"""
    if not cut_points:
        return start - AUDIO_PADDING, end + AUDIO_PADDING, None, None, "straight"

    cut_times = sorted([c.time_sec for c in cut_points])

    # Layer 1: Shot-level replacement (> 60% coverage)
    for i in range(len(cut_times) - 1):
        s, e = cut_times[i], cut_times[i + 1]
        if start >= s and end <= e:
            if (end - start) / max(e - s, 0.1) > SHOT_COVERAGE:
                return s, e, None, None, "straight"

    # Layer 2: Snap & Padding
    final_v_start = max(0.0, start - AUDIO_PADDING)
    final_v_end = end + AUDIO_PADDING
    final_a_start = final_v_start
    final_a_end = final_v_end

    closest_start = min(cut_times, key=lambda x: abs(x - start))
    closest_end = min(cut_times, key=lambda x: abs(x - end))

    # Layer 3: J/L Cut detection (disabled by default)
    if JLCUT_ENABLED:
        delta_start = start - closest_start
        delta_end = end - closest_end

        for delta, is_start in [(delta_start, True), (delta_end, False)]:
            if MIN_SPLIT <= abs(delta) <= MAX_SPLIT:
                if is_start:
                    if delta > 0:  # L-Cut: video leads
                        final_v_start = closest_start
                        final_a_start = start
                        edit_type = "l_cut"
                    else:          # J-Cut: audio leads
                        final_a_start = start
                        final_v_start = closest_start
                        edit_type = "j_cut"
                # ... same for end
                return final_v_start, final_v_end, final_a_start, final_a_end, edit_type

    # Layer 2 fallback: snap within window
    if abs(closest_start - start) <= SNAP_WINDOW:
        final_v_start = closest_start
        final_a_start = closest_start
    if abs(closest_end - end) <= SNAP_WINDOW:
        final_v_end = closest_end
        final_a_end = closest_end

    return final_v_start, final_v_end, None, None, "straight"
```

## 6. Stage 4 处理（规划）

### Straight Cut（当前实现，无需改动）
```bash
ffmpeg -ss {start} -to {end} -i seedance.mp4 -c copy seg.mp4
ffmpeg concat ...
```

### J-Cut / L-Cut（未来实现）
```bash
# J-Cut: 原剧画面 + seedance 音频
# 冲突区间 [10.0s, 11.0s]：用原剧视频 + seedance 音频混流
ffmpeg -ss 10.0 -to 11.0 -i original.mp4 -vn -c:a copy seg_audio.m4a  # 原剧静音
ffmpeg -ss 10.0 -to 11.0 -i seedance.mp4 -an -c:v copy seg_video.mp4 # seedance 画面
ffmpeg -ss 10.0 -to 11.0 -i seedance.mp4 -vn -c:a copy seg_jcut.m4a  # seedance 音频
# 混合：原剧画面 + seedance 音频
ffmpeg -i seg_video.mp4 -i seg_jcut.m4a -c:v copy -c:a aac output.mp4
```

## 7. 实施计划

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | `_snap_boundaries()` 层 1 + 层 2（Shot-level + Snap & Padding） | ✅ 已实现 |
| Phase 2 | 数据模型升级（video_start_sec 等字段） | 📋 待定 |
| Phase 3 | 层 3 J/L Cut 决策逻辑（`JLCUT_ENABLED = True`） | 📋 待定 |
| Phase 4 | Stage 4 ffmpeg 音画分离混合 | 📋 待定 |

Phase 1 已就绪。Phase 2-4 建议等第一批真实 seedance 视频产出后，根据实际画面过渡效果决定优先级。
