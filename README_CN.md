# Memora AI

[English](README.md) | 简体中文

Memora AI 是一个独立的照片理解与检索服务，可与 [Immich](https://immich.app/)
等自托管照片管理器配合使用：

```text
Immich（上传、相册、时间线、界面）
        | REST API：元数据与预览缩略图
        v
Memora AI（检索、人物、事件、相似照片、最佳照片）
        | 资源 UUID 与鉴权缩略图代理
        v
面向 Immich 的界面
```

项目聚焦于照片智能处理中的以下能力：

- 图像与自然语言的语义检索；
- 基于时间、视觉特征和 GPS 的事件发现；
- 连拍和近重复照片分组；
- 图像质量分析与最佳照片排序；
- 可插拔的 OpenCLIP、InsightFace 与 Qdrant 集成；
- 可复现的评估流程与暴力检索基线。

## 开源依赖与致谢

Memora AI 基于开源软件构建。本仓库保持自身实现独立，同时集成了以下关键项目：

| 项目 | 在 Memora AI 中的用途 |
| --- | --- |
| [FastAPI](https://fastapi.tiangolo.com/) 与 [Uvicorn](https://www.uvicorn.org/) | Python HTTP API 与本地服务运行时。 |
| [Vue](https://vuejs.org/) 与 [Vite](https://vite.dev/) | Web 控制台与前端开发工具链。 |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) 与 [PyTorch](https://pytorch.org/) | 可选的图文嵌入模型与语义检索能力。 |
| [InsightFace](https://github.com/deepinsight/insightface) 与 [ONNX Runtime](https://onnxruntime.ai/) | 可选的人脸检测、人脸嵌入和人物聚类。 |
| [FAISS](https://github.com/facebookresearch/faiss)、[Qdrant](https://qdrant.tech/) 与 [hnswlib](https://github.com/nmslib/hnswlib) | 可选向量索引后端与检索基准测试。 |
| [OpenCV](https://opencv.org/)、[scikit-learn](https://scikit-learn.org/) 与 [ImageHash](https://github.com/JohannesBuchner/imagehash) | 图像分析、聚类和感知哈希重复检测。 |
| [Immich](https://immich.app/) | 可选的自托管照片库集成；照片资产仍以 Immich 为准。 |

分发或部署本项目时，请遵守各依赖的许可证、版权声明和模型权重条款。
预训练模型权重、数据集与外部服务的条款，可能不同于加载它们的软件包许可证。
完整的直接依赖列表见 [`pyproject.toml`](pyproject.toml) 和
[`frontend/package.json`](frontend/package.json)。

## 私有测试数据

仓库不包含开发者的私有测试照片。`photos/` 原始照片目录、
`photos_prepared/` 标准化 JPEG 数据集，以及 `data/` 下的索引、清单、人物聚类和评估 JSON
都不会提交到 Git，因为其中可能包含私有路径、元数据或向量嵌入。

请将自己的照片放入 `photos/`，转换到 `photos_prepared/` 后在本地建立索引。未来的生产环境将通过应用上传或 API 接收照片，而不是通过仓库提交文件。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,vision]"
memora index .\photos_prepared --index-path .\data\index.json
memora search "a beach" --index-path .\data\index.json
uvicorn memora.api.main:app --reload
```

默认编码器是确定性的轻量编码器，不下载模型也可运行。若需要真实的语义检索，请安装 OpenCLIP：

```powershell
pip install -e ".[openclip]"
```

OpenCLIP 的预训练权重会在 `OpenCLIPEncoder` 首次加载所选模型时下载；更换模型或编码器后，需要重新建立索引。

```powershell
memora index .\photos_prepared --index-path .\data\index-openclip.json --encoder open_clip
memora search "海边" --index-path .\data\index-openclip.json --encoder open_clip --strategy query_enhancement
```

如果源目录包含 HEIC/HEIF 文件，请先安装预处理依赖并转换为 JPEG：

```powershell
pip install -e ".[preprocess]"
python scripts/prepare_dataset.py .\photos --output .\photos_prepared
```

## 核心能力

### 语义搜索与元数据筛选

支持原始 CLIP 查询、提示词集成与查询扩展三种检索策略，并可将自然语言中的时间条件、明确日期范围和 GPS 边界框转换为筛选条件。

```powershell
memora search "last year beach photos" `
  --index-path data/index-openclip-prepared.json `
  --encoder open_clip `
  --reference-date 2026-08-17
```

### 人物聚类

人物聚类为可选能力，处理流程为 `InsightFace -> 人脸嵌入 -> DBSCAN -> 质量加权人物原型`。

```powershell
pip install -e ".[people]"
python scripts/cluster_people.py `
  --index-path data/index-openclip-prepared.json `
  --people-path data/people.json `
  --ctx-id 0
```

可使用 `memora people-merge`、`memora people-remove` 或相应 API 对聚类结果进行人工修正。

### 事件、旅程与叙事

事件发现提供 `time_only`、`time_clip`、`time_clip_gps` 和 `strict_event` 等策略，分别用于比较时间、视觉和位置特征的贡献。Ver4 还支持旅程发现、事件命名、旅程命名和说明生成。

详细数据模型、命令与标注格式见
[`docs/ver4_events_journeys.md`](docs/ver4_events_journeys.md)。

### 相似照片与最佳照片

`group_similar` 综合 pHash 距离、CLIP 余弦相似度和拍摄时间窗口进行分组；每个分组根据清晰度、曝光、人脸质量和构图等质量信号选出代表照片。

```powershell
memora similar --index-path data/index.json --phash-distance 10 `
  --visual-similarity 0.90 --time-window-seconds 30
```

### 向量索引基准

NumPy 精确索引作为召回率基准，可与 FAISS Flat、FAISS HNSW、hnswlib HNSW 和 Qdrant HNSW 比较 Recall@10、平均/P95 延迟和索引内存占用。

```powershell
pip install -e ".[vector]"
python scripts/benchmark.py --index-path data/index.json --queries 100
```

## Immich 集成

Immich 负责上传、相册和时间线，并始终是资产事实来源。Memora 通过 Immich API 读取照片元数据和预览缩略图，在本地建立索引，并在 AI 结果中返回对应的 Immich 资源 UUID；不需要直接访问 Immich 的上传目录或数据库。

```powershell
$env:MEMORA_IMMICH_URL = "http://localhost:2283"
$env:MEMORA_IMMICH_API_KEY = "your-api-key"
$env:MEMORA_ENCODER = "open_clip"
memora immich-status
memora immich-sync --encoder open_clip --index-path data/index.json
```

集成接口、权限要求与请求示例见
[`docs/stage7_immich_integration.md`](docs/stage7_immich_integration.md)。

## Web 控制台

前端基于 Vue 3、TypeScript 与 Vite。先启动 FastAPI，再在第二个终端启动前端：

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://localhost:5173`。开发服务器会将 `/api/*` 请求代理至端口 `8000` 的 FastAPI 服务。

控制台支持创建独立照片工作区、上传照片或文件夹、建立索引、自然语言搜索、人物聚类、事件发现、相似照片分组和最佳照片排序，并可导出 JSON 项目清单、CSV 照片索引或高排名原图 ZIP 包。

## API

启动 `uvicorn memora.api.main:app --reload` 后，可使用以下主要接口：

- `GET /health`
- `POST /index`、`POST /search`
- `POST /people/cluster`、`GET /people`、`POST /people/merge`
- `GET /events`、`POST /journeys/discover`
- `GET /similar-groups`、`GET /quality/{photo_id}`
- `GET /immich/status`、`POST /immich/sync`、`POST /immich/albums`
- `GET/POST /projects`、`POST /projects/{project_id}/analyze`
- `GET /projects/{project_id}/photos`、`POST /projects/{project_id}/search`
- `GET /projects/{project_id}/events`、`GET /projects/{project_id}/similar-groups`
- `GET /projects/{project_id}/best-shots`

## 架构

```text
原始照片
  |-- metadata/exif.py -------- 时间、GPS、相机信息
  |-- encoders/clip_encoder.py - 图像/文本嵌入
  |-- quality/ ------------------ 模糊、曝光、最佳照片
  |-- duplicate/ ---------------- pHash 与视觉相似度
  `-- clustering/ --------------- 事件与人物
                    |
              retrieval/index.py
                    |
                 FastAPI / CLI
```

`HashImageEncoder` 和 NumPy 暴力检索是有意保留的基线，用于在不下载模型的情况下验证算法流程。`OpenCLIPEncoder`、`InsightFaceEncoder` 和 `QdrantStore` 都是可替换的可选适配器。

## 路线图

### v1.0：可用的本地照片智能服务

- 图文语义检索、查询扩展和元数据筛选。
- 人物聚类、事件发现、相似照片分组与最佳照片排序。
- 本地项目工作区、结果导出和 Vue 控制台。
- 不直接访问 Immich 存储或数据库的可选同步集成。

### v1.1：质量、可控性与部署

- 通过标注评估集和可配置阈值，提升检索、事件和人物聚类准确率。
- 提供人物、事件、排序与生成名称的人工反馈工作流。
- 完善增量索引、向量存储配置与 GPU/CPU 部署指引。
- 加强 API 与前端流程的可观测性、异常处理和回归测试。

### v2.0：协作与生产化照片智能

- 引入多用户工作区、访问控制和持久化服务部署。
- 支持可扩展的后台处理、分布式向量存储与大型照片库同步。
- 提供更丰富的相册、旅程和故事生成，并保留可审查的用户控制。
- 扩展集成 API，让照片管理器和其他客户端可直接消费 AI 结果。

路线图是产品阶段规划；包和前端的实际版本分别以 `pyproject.toml` 与 `frontend/package.json` 为准。
