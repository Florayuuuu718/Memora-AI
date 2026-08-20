# Stage 4：Event、Journey 与命名实验结果

测试日期：2026-08-18  
图片目录：`photos_prepared`  
图片数量：126  
增强索引：`data/index-openclip-prepared-v4-enriched.json`  
人物索引：`data/people-v4.json`  
人工标注：`data/ver4_annotations.csv`  
事件评估算法：`strict_event_people`

## 1. 输入索引与人工标注

本次使用的是已经格式化后的 `photos_prepared`，不是其他数据集。

| 项目 | 结果 |
|---|---:|
| 图片总数 | 126 |
| OpenCLIP 向量维度 | 512 |
| 检测到的人脸 | 132 张人脸 |
| 含人脸的照片 | 70 张 |
| DBSCAN 人物组 | 8 组 |
| 已人工标注照片 | 96 |
| high-confidence | 77 |
| medium-confidence | 19 |
| 严格 Event | 25 |
| 容错 Event Family | 24 |
| 原始 Journey Stop | 12 |
| Parent Journey | 11 |

每张照片的增强质量字段包括：`sharpness`、`sharpness_score`、`exposure_score`、`face_quality_score`、`composition_score` 和 `score`。

## 2. Event 对比实验

### 2.1 全部可用标注

| 策略 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Time Only | 0.2485 | 0.9938 | 0.3975 |
| Time + CLIP | 0.2417 | 0.9938 | 0.3889 |
| Time + CLIP + GPS | 0.2485 | 0.9938 | 0.3975 |
| Strict Event | **0.9375** | 0.3704 | 0.5310 |
| Strict Event + People | 0.8554 | 0.4383 | 0.5796 |
| Strict Event + People（容错边界） | **0.9091** | **0.4678** | **0.6178** |

### 2.2 只使用 high-confidence 标注

| 策略 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Time Only | 0.2530 | 0.9932 | 0.4033 |
| Time + CLIP | 0.2454 | 0.9932 | 0.3935 |
| Time + CLIP + GPS | 0.2530 | 0.9932 | 0.4033 |
| Strict Event | **0.9333** | 0.3810 | 0.5411 |
| Strict Event + People | 0.8462 | 0.4490 | 0.5867 |
| Strict Event + People（容错边界） | **0.8974** | **0.4636** | **0.6114** |

容错评价识别出 9 对“边界不同但合并可接受”的照片关系；仍有 8 对真正的错误合并关系。严格分数仍然保留，用来衡量算法是否遵守人工选择的细粒度边界。

## 3. Event 命中与未命中

| 类型 | 数量 | 具体内容 |
|---|---:|---|
| 严格完整命中 | 15/25 | E01、E09–E22 |
| 严格找全但有混入 | 4/25 | E03、E23、E24、E25 |
| 部分命中 | 4/25 | E05、E06、E07、E08 |
| 只命中一张 | 2/25 | E02、E04 |
| 容错完整命中 | 17/24 Event Family | E01、E09–E22、EF23_24、EF25 |

人工复核后的重要修订：

| 照片或原事件 | 修订结果 | 评价含义 |
|---|---|---|
| 17、18、53 | E18 | 确认属于同一 Event，修订后完整命中 |
| 1、24、35 | E20 | 可属于同一 Event；照片 1 因 GPS 边界保留 medium |
| 75、77、89 | E21 | 确认属于同一 Boston Event，修订后完整命中 |
| E23（42、43）与 E24（40、41） | 严格分开；容错为 EF23_24 | 有距离，但合并不算严重错误 |
| E25（34、36）与 33、39 | 严格分开；容错为 EF25 | GPS 不够近，但可接受为宽松事件 |

当前最需要继续优化的是 E02、E04–E08 的事件拆散，以及 E03/E07 之间的硬误合并。GPS 边界造成的 E23/E24、E25 差异不应单独作为严重失败。

## 4. Journey 对比实验

为了和本轮人工标签做可复现对照，下面的 benchmark 临时使用成都 `30.66,104.06`、半径 80 km，以及人工标签中的香港/Boston 解释。这个配置只用于评价，不是生产系统的固定地点；生产流程使用 GPS 自动推断。

### 4.1 全部可用标注

| 评价方式 | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|
| 旧版 Flat Journey | 0.6330 | 0.5498 | 0.5885 | 0.7051 |
| Parent Journey | **0.9266** | **0.6413** | **0.7580** | 0.7051 |
| Journey Stop | **0.8961** | 0.5498 | **0.6815** | 0.7051 |

### 4.2 只使用 high-confidence 标注

| 评价方式 | Precision | Recall | F1 | Coverage |
|---|---:|---:|---:|---:|
| 旧版 Flat Journey | 0.7043 | 0.4821 | 0.5724 | 0.6885 |
| Parent Journey | **0.8609** | **0.5323** | **0.6578** | 0.6885 |
| Journey Stop | **0.8351** | 0.4821 | **0.6113** | 0.6885 |

### 4.3 Journey 命中情况

| 层级 | 完整命中 | 部分或误合并 | 未命中 |
|---|---|---|---|
| Parent Journey | BIG_J11_J12、J04、J05、J08、J10 | J02、J06、J07、J09 | J01、J03 |
| Journey Stop | HONG_KONG、BOSTON、J04、J05、J08、J10 | J02、J06、J07、J09 | J01、J03 |

香港 → Boston 被识别为一个 Parent Journey，并正确拆为两个 Stop：

| 层级 | 时间 | 照片 |
|---|---|---|
| Parent Journey：香港 → Boston | 2024-07-25 至 2024-08-07 | 两个 Stop 共 16 张 |
| 香港 Stop | 2024-07-25 至 2024-07-26 | 33、34、36、39、40、41、42、43 |
| Boston Stop | 2024-07-28 至 2024-08-07 | 45、47、50、58、75、77、85、89 |

J01、J03 未命中的主要原因是 EXIF/GPS 缺失，系统缺少可信的 away-from-home 锚点；当前 Coverage 约为 70.5%。

## 5. GPS 自动地点推断结果

这是生产命名流程的实际结果，不使用预先写死的成都、香港、Boston、大理或三亚。

| 项目 | 结果 |
|---|---:|
| 有 GPS 的照片 | 71 |
| GPS 地点簇 | 15 |
| 作为 Journey 目的地的地点簇 | 11 |
| GPS 聚类半径 | 40 km |
| 目的地最少照片数 | 2 |

推断的常驻地候选：`Xipu, Sichuan`，中心 `30.7472, 103.9330`，20 张照片，6 个活跃月份、15 个活跃日期，推断半径约 57.8 km。它说明数据中存在成都西侧附近的长期高频区域，但显示名称来自离线反向地理编码，未把“成都”写死。

主要目的地地点簇：

| 地点名称 | 中心坐标 | 照片数 |
|---|---|---:|
| Weizhou, Sichuan | 31.5382, 103.5145 | 9 |
| Wan Chai, Wanchai | 22.2711, 114.1816 | 8 |
| Cambridge, Massachusetts | 42.3593, -71.0856 | 8 |
| Yuhu, Yunnan | 25.9053, 100.1925 | 4 |
| Tiandu, Hainan | 18.2544, 109.6677 | 4 |
| Gongtan, Chongqing Shi | 28.9247, 108.3474 | 4 |
| Zunyi, Guizhou Sheng | 27.7230, 106.7822 | 3 |
| Nanping, Chongqing Shi | 29.5608, 106.5921 | 2 |
| Guandu, Chongqing Shi | 30.0810, 106.3802 | 2 |
| Yuanhou, Guizhou Sheng | 28.3984, 105.9696 | 2 |
| Shilu, Jiangsu Sheng | 31.3182, 120.5881 | 2 |

## 6. Event / Journey 命名结果

结果文件：`data/ver4_named_events_journeys.json`。

| 项目 | 结果 |
|---|---:|
| Event 总数 | 76 |
| 已生成 EventName | 76/76 |
| Journey 总数 | 10 |
| 已生成 JourneyName/Note | 10/10 |
| Event 不同名称 | 39 |
| “未命名记忆”模板兜底 | 4 |
| `name_source=template` | Event 76 / Journey 10 |

自动生成的 Journey 名称示例：

| Journey | 自动名称 |
|---:|---|
| 0 | Wan Chai, Wanchai 与 Cambridge, Massachusetts 之旅 |
| 1 | Yuhu, Yunnan 之旅 |
| 2 | Tiandu, Hainan 之旅 |
| 3 | Gongtan, Chongqing Shi 之旅 |
| 4 | 2025-11 的旅行（目的地证据不足） |
| 9 | Zunyi, Guizhou Sheng 与 Yuanhou, Guizhou Sheng 之旅 |

Event 示例包括：`17、18、53` 的“Tiandu, Hainan 海边游玩”、`75、77、89` 的“Cambridge, Massachusetts 看日落”、`90、93、95` 的“生日聚会”。名称、Summary、活动候选、对象候选、人物 ID 和照片文件名均写入 JSON。

## 7. LLM 降级实验

本轮没有配置真实 LLM，因此正常实跑使用模板。另用不可用测试地址验证降级链路：

| 项目 | 结果 |
|---|---|
| LLM configured | true |
| LLM available | false |
| 错误 | `URLError: timed out` |
| generation_mode | `template_fallback_llm_unavailable` |
| 最终已命名 Event | 76/76 |
| 最终已命名 Journey | 10/10 |

单元测试同时验证了可用接口返回 LLM JSON，以及不可用接口只实际尝试一次，之后由熔断器降级。

## 8. 可复现命令与结论

```powershell
python scripts/evaluate_events.py --index-path .\data\index-openclip-prepared-v4-enriched.json
python scripts/evaluate_journeys.py --index-path .\data\index-openclip-prepared-v4-enriched.json
python scripts/generate_ver4_narratives.py `
  --index-path .\data\index-openclip-prepared-v4-enriched.json `
  --people-path .\data\people-v4.json `
  --output .\data\ver4_named_events_journeys.json `
  --encoder open_clip
```

本轮结论：容错 Event F1 为 0.6178，Parent Journey F1 为 0.7580；香港和 Boston 已能同时表达为一个大 Journey 与两个 Stop。下一轮应优先解决无 EXIF 照片造成的 J01/J03 漏检，以及 E02、E04–E08 的事件拆散。
