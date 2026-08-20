# Stage 4：Event、Journey 与命名方法论

## 1. 目标与对象定义

Stage 4 处理两种不同层级的照片组织结果：

```text
照片
├── Event：一次具体活动或短暂记忆
│   └── Event Family：允许合理 GPS 边界差异的容错事件集合
└── Parent Journey：一次连续的离家旅行
    ├── Journey Stop：旅行中的城市或地区停留段
    ├── Event
    └── 无 EXIF 的零散照片
```

Event 不是简单按时间切片；Journey 也不是“同一城市的所有照片”。一次长途旅行可以包含多个城市 Stop，多个 Event 也可以组成一次 Parent Journey。

## 2. 数据可信度与 Event 规则

### 2.1 元数据优先级

- EXIF 拍摄时间：可信的时间锚点。
- EXIF GPS：用于地点距离和硬冲突判断。
- 文件系统时间：只表示文件被处理或上传的时间，不能直接当作拍摄时间。
- 无 EXIF 图片：使用上传批次、CLIP、人物和外观证据，但不能伪造时间或 GPS。

### 2.2 无 EXIF 图片的严格处理

```text
可信时间/GPS一致
        ├── 有 EXIF：时间 + GPS + CLIP
        ├── 同一上传批次无 EXIF：中高 CLIP 置信度才附着
        └── 跨上传批次无 EXIF：需要高 CLIP/人物置信度
```

原则是优先避免误合并：无 EXIF 图片只有在视觉和人物证据足够强时才加入已有 Event；没有可靠证据时保留为零散照片，而不是强行归入某个事件。

### 2.3 人物与外观证据

Event 发现使用以下辅助证据：

- InsightFace identity embedding：判断是否可能是同一人物。
- 同一人物是否在多张照片中共同出现。
- 人物衣着、妆容和颜色等上半身外观 embedding。
- 多人合照形成的关联关系。例如三人合照中出现的两个人，其单人照片可以获得关联加分。
- CLIP 图像相似度：判断活动、场景和对象是否相近。

人物只作为加分证据。出现明显时间或 GPS 冲突时，不能仅凭“同一个人”合并 Event。

### 2.4 双层 Event 真值与评价

人工标注保留两种边界：

- `event_id`：用户首选的严格事件边界。
- `event_family_id`：位置有一定距离，但人工认为合并也合理的容错边界。
- `label_confidence`：`high` 或 `medium`，表示人工对边界的确定程度。

同时报告：

- Strict Event Precision / Recall / F1；
- Tolerant Event Family Precision / Recall / F1；
- 可以容忍的边界合并关系；
- 真正的硬误合并关系。

这样可以区分“算法违反了明确边界”和“不同人对同一事件的 GPS 范围理解不同”。例如 E23/E24 可以在严格结果中保持分开，在容错结果中允许合并。

## 3. Journey 方法

### 3.1 常驻地自动推断

生产流程不预置“成都、香港、Boston、大理、三亚”等城市。地点由照片 GPS 推断：

```text
EXIF GPS
  → 40 km 地点聚类
  → 按 active_months、active_days、photo_count 判断长期高频簇
  → 最高可信地点簇作为常驻地候选
  → 其他地点簇作为旅行目的地候选
  → 离线反向地理编码得到最近地点名称
```

当前实现使用 40 km 聚类阈值，并要求目的地候选至少有 2 张带 GPS 的照片。`reverse_geocoder` 使用离线地点库，不上传个人 GPS。用户显式传入的常驻地或地点名称只是覆盖自动推断的 `manual_override`，不是系统默认知识。

### 3.2 Parent Journey

Parent Journey 表示一次连续离开常驻地的完整旅行。判断依据包括：

- 是否离开常驻地范围；
- 可信时间是否连续；
- 中间是否重新回到常驻地；
- 相邻旅行事件之间是否超过 `max_gap_days`。

因此，香港之后继续前往 Boston 可以属于同一个 Parent Journey，而不是被强制拆成两次旅行。

### 3.3 Journey Stop

Parent Journey 内部再根据 GPS 拆成城市或地区 Stop：

- 默认从 GPS 地点簇生成候选区域；
- 使用离线反向地理编码返回最近 locality；
- 默认 `stop_radius_km=150`；
- 无 GPS 照片根据时间邻接、人物和视觉证据附着到最可信 Stop；证据不足时保留不确定状态。

人工标注中的香港 → Boston 是一个 Parent Journey 的两个 Stop：

```text
Parent Journey: 2024-07-25 至 2024-08-07
├── 香港 Stop：33,34,36,39,40,41,42,43
└── Boston Stop：45,47,50,58,75,77,85,89
```

人工标注字段：

- `journey_id`：原始旅行或地点分组；
- `journey_parent_id`：完整大 Journey；
- `journey_stop_id`：城市或地区 Stop；
- 无 EXIF 照片仍保留 Journey 标签，用于验证自动附着能力。

## 4. Event / Journey 命名

### 4.1 EventName

EventName 只使用已经提取出的结构化事实：人物、地点、活动、主要对象、时间段和持续长度。事实不足时使用保守的“记忆”类模板，不凭空补充活动。

### 4.2 JourneyName 与 JourneyNote

JourneyName/JourneyNote 可以使用：

- Parent Journey 起止时间；
- 经过的城市或地区 Stop；
- 同行人物；
- 所包含的 Event 和活动。

无法确认的信息不写入名称或游记。命名只接收结构化事实，不发送原图和原始 embedding。

### 4.3 LLM 优先、模板降级

默认行为：

1. 同时存在 `MEMORA_LLM_URL` 和 `MEMORA_LLM_MODEL` 时优先调用 LLM；
2. 第一次网络错误、超时、非法 JSON 或字段不完整时触发熔断；
3. 当前批次剩余 Event/Journey 立即改用模板，不重复等待失败接口；
4. 每条结果用 `name_source=llm|template` 记录来源；
5. API 返回 `narrative_backend`，包含配置、可用性和降级原因。

没有配置 LLM 时直接使用模板。LLM 接口保留，但不会阻塞当前版本运行。

## 5. 人工反馈设计

人工反馈不直接覆盖原始算法证据，而是保存为可审计的修订层，例如：

- Merge Person 3 and Person 7；
- Remove photo from Person 2；
- 合并两个可接受的 Event Family；
- 将同一 Parent Journey 拆为多个 Journey Stop；
- 修改地点显示名、EventName 或 JourneyName。

原始 embedding、GPS、时间和算法置信度继续保留，以便重新运行和比较版本。

## 6. 当前运行入口

自动 GPS 推断和 LLM 自动降级：

```powershell
.\.venv\Scripts\python.exe scripts\generate_ver4_narratives.py `
  --index-path .\data\index-openclip-prepared-v4-enriched.json `
  --people-path .\data\people-v4.json `
  --output .\data\ver4_named_events_journeys.json `
  --encoder open_clip
```

Journey 发现可以省略常驻地坐标：

```powershell
memora journeys `
  --max-gap-days 5 --stop-radius-km 150 `
  --index-path data/index-openclip-prepared-v4-enriched.json `
  --people-path data/people-v4.json
```

不传 `--home-lat/--home-lon` 时为 `auto_gps`；传入坐标时为 `manual_override`。API `POST /journeys/discover` 的 `home` 参数同样可以省略。

## 7. 已知限制

- 无 EXIF 照片仍缺少可靠的时间和地点锚点，只能依靠附着证据；
- 离线反向地理编码可能返回街道或城区名，而不是用户习惯的城市名；
- 常驻地推断需要跨月份或跨日期的足够 GPS 照片；
- OpenCLIP 的活动和对象是候选，不等于人工确认；
- `stop_radius_km` 应允许按城市密度和用户偏好调整。
