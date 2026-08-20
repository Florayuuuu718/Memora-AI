# Stage 5：相似照片与最佳照片

## 1. 功能目标

先找出同一时间窗口内的相似照片，再从每组中选择质量更高、信息更完整的一张作为代表照片。

```text
pHash + OpenCLIP 相似度 + 时间窗口
              ↓
       Similar Shot Group
              ↓
Sharpness + Exposure + Face Quality + Composition
              ↓
          Best Shot
```

## 2. 当前实现

相似照片候选同时满足以下约束：

- pHash 距离不超过 `10`；
- OpenCLIP 相似度不低于 `0.90`；
- 时间窗口不超过 `30s`。

最佳照片评分使用增强索引中的：

- `sharpness_score`：清晰度；
- `exposure_score`：曝光；
- `face_quality_score`：人脸检测质量、脸部面积等当前可计算特征；
- `composition_score`：当前的轻量构图/边缘能量与三分法估计。

这些分数用于排序，不等同于人工审美模型。

## 3. 实验结果

数据来自 126 张照片的增强索引：`data/index-openclip-prepared-v4-enriched.json`。

| 项目 | 结果 |
|---|---:|
| 相似照片组 | 12 组 |
| 组内照片 | 25 张 |
| 有效代表照片 | 12/12 |
| 评分后 Best Shot 改变 | 1 组 |

发生变化的是第 9 组：

| 照片 | 综合分数 | Face Quality | Composition |
|---|---:|---:|---:|
| 原代表 `000097.jpg` | 0.4657 | 0.1479 | 0.7125 |
| 新代表 `000086.jpg` | **0.4742** | **0.2519** | 0.6943 |

这说明 Face Quality 和 Composition 已经实际参与 Best Shot 排序；但这不是人工标注的 Top-1 正确率实验，只能说明功能和排序变化已经生效。

## 4. 当前限制

- Composition 是轻量级图像特征估计，不是审美模型；
- Face Quality 目前主要反映检测质量和脸部面积，不能完整表达表情、闭眼和遮挡；
- 尚未建立人工 Best Shot 真值，因此暂时不能报告准确率、Precision 或 Recall；
- 相似照片阈值仍需在更大数据集上验证，避免把连续但不同的活动合并。

## 5. 运行与验证

重新生成增强索引后，检查每张照片是否存在 `sharpness_score`、`exposure_score`、`face_quality_score`、`composition_score` 和 `score`，再运行 Ver5 相似照片/最佳照片评估脚本。当前实验结果已记录在 Stage 4–6 历史报告中，本文作为 Stage 5 的独立主文档。
