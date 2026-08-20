# Memora AI

English | [简体中文](README_CN.md)

Memora AI is an independent photo-understanding and retrieval service. It is designed to sit beside a self-hosted photo manager such as Immich:

```text
Immich (upload, albums, timeline, UI)
        | REST API: metadata + preview thumbnails
        v
Memora AI (retrieval, people, events, similar shots, best shot)
        | asset UUID + authenticated thumbnail proxy
        v
Immich-facing UI
```

The project focuses on the parts that are useful for an image-algorithm portfolio:

- semantic image/text retrieval;
- time + visual + GPS event discovery;
- near-duplicate and burst-shot grouping;
- image quality and best-shot ranking;
- pluggable OpenCLIP, InsightFace and Qdrant integrations;
- reproducible evaluation and brute-force baselines.

## Open-source dependencies and acknowledgements

Memora AI is built on open-source software. This repository keeps its own
implementation independent, while integrating the following key projects:

| Project | Usage in Memora AI |
| --- | --- |
| [FastAPI](https://fastapi.tiangolo.com/) and [Uvicorn](https://www.uvicorn.org/) | Python HTTP API and local service runtime. |
| [Vue](https://vuejs.org/) and [Vite](https://vite.dev/) | Web dashboard and frontend development tooling. |
| [OpenCLIP](https://github.com/mlfoundations/open_clip) and [PyTorch](https://pytorch.org/) | Optional image-text embedding and semantic retrieval. |
| [InsightFace](https://github.com/deepinsight/insightface) and [ONNX Runtime](https://onnxruntime.ai/) | Optional face detection and face embeddings for people clustering. |
| [FAISS](https://github.com/facebookresearch/faiss), [Qdrant](https://qdrant.tech/) and [hnswlib](https://github.com/nmslib/hnswlib) | Optional vector-index backends and retrieval benchmarks. |
| [OpenCV](https://opencv.org/), [scikit-learn](https://scikit-learn.org/) and [ImageHash](https://github.com/JohannesBuchner/imagehash) | Image analysis, clustering and perceptual-hash duplicate detection. |
| [Immich](https://immich.app/) | Optional self-hosted photo-library integration; Immich remains the system of record for assets. |

Please comply with the licenses, copyright notices and model-weight terms of
each dependency when distributing or deploying this project. In particular,
pretrained model weights, datasets and external services may have terms that
differ from the Python or JavaScript package that loads them. See
[`pyproject.toml`](pyproject.toml) and
[`frontend/package.json`](frontend/package.json) for the complete direct
dependency lists.

## Private test data

This repository does not contain the developer's private test photos. The
local `photos/` source dataset and `photos_prepared/` normalized JPEG dataset
are ignored by Git. Generated indexes, manifests, people clusters and
evaluation JSON files under `data/` are also local-only because they may
contain private paths, metadata or embeddings.

To run the experiments locally, place your own images in `photos/`, prepare
them into `photos_prepared/`, and build the indexes locally. In the future,
production users will provide photos through the application upload/API
interface rather than through files committed to this repository.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,vision]"
memora index .\photos_prepared --index-path .\data\index.json
memora search "a beach" --index-path .\data\index.json
uvicorn memora.api.main:app --reload
```

The default encoder is a deterministic lightweight encoder, so the service works without downloading a model. For real semantic retrieval, install the OpenCLIP runtime separately:

```powershell
pip install -e ".[openclip]"
```

`pip install` installs the runtime and model code. The pretrained weights are
downloaded the first time `OpenCLIPEncoder` loads the selected model. This is
intentional: different models have different weights, and the weights are much
larger than the Python package.

For an NVIDIA GPU, install the PyTorch wheel for the CUDA version supported by
your driver first, then install the project extra. For example, CUDA 12.8:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[openclip]"
```

For CPU-only installation:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[openclip]"
```

Verify the runtime before indexing:

```powershell
python -c "import torch, open_clip; print(torch.__version__); print(torch.cuda.is_available()); print('open_clip ok')"
```

If the source photo directory contains HEIC/HEIF files, install the separate
preprocessing extra. The runtime indexer intentionally handles only the
normalized JPEG dataset:

```powershell
pip install -e ".[preprocess]"
```

Then convert the source dataset without overwriting the originals:

```powershell
python scripts/prepare_dataset.py .\photos --output .\photos_prepared
```

OpenCLIP embeddings are model-specific, so rebuild the index after switching
encoders:

```powershell
memora index .\photos_prepared --index-path .\data\index-openclip.json --encoder open_clip
memora search "海边" --index-path .\data\index-openclip.json --encoder open_clip --strategy query_enhancement
```

Stage 1 includes three query experiments:

- `raw_clip`: encode the original query once;
- `prompt_ensemble`: average five prompt templates around the original query;
- `query_enhancement`: expand bilingual/semantic candidates, then average their text embeddings.

Prepare labeled cases such as `data/retrieval_cases.json`:

```json
[{"query": "海边", "relevant_ids": ["photo-id-1", "photo-id-2"]}]
```

Run Recall@1/5/10 for all three strategies:

```powershell
python scripts/evaluate_retrieval.py data/retrieval_cases.json --index-path data/index-openclip.json
```

The current single-query experiment and its interpretation are recorded in
[`docs/stage1_openclip_experiment.md`](docs/stage1_openclip_experiment.md).

The three-group experiment is recorded in
[`docs/stage1_three_group_experiment.md`](docs/stage1_three_group_experiment.md).

## Stage 2: semantic search with metadata filters

The search service now separates a natural-language query into a semantic
part and metadata constraints. Relative time expressions such as `last year`,
`this year`, `去年` and `今年` become capture-time filters. Explicit date
ranges and GPS bounding boxes can also be supplied directly.

```powershell
memora search "last year beach photos" `
  --index-path data/index-openclip-prepared.json `
  --encoder open_clip `
  --reference-date 2026-08-17
```

GPS filtering uses `MIN_LAT MIN_LON MAX_LAT MAX_LON` and only trusts EXIF GPS
by default:

```powershell
memora search "beach photos" `
  --index-path data/index-openclip-prepared.json `
  --encoder open_clip `
  --bbox 30.0 120.0 31.0 121.0
```

The same request is available through `POST /search` with `captured_from`,
`captured_to`, `bbox`, `reference_date` and
`fallback_if_unavailable`. Strict filtering is the default. The fallback flag
is useful for an old dataset where the requested metadata field is absent in
the entire index; it does not silently keep photos when only some records are
missing metadata.

阶段 2 测试记录见
[`docs/stage2_metadata_filter_test.md`](docs/stage2_metadata_filter_test.md)。

## Stage 3: InsightFace people clustering

People clustering is an optional model path. Install it after the normalized
JPEG dataset and CLIP index are ready:

```powershell
pip install -e ".[people]"
python scripts/cluster_people.py `
  --index-path data/index-openclip-prepared.json `
  --people-path data/people.json `
  --ctx-id 0
```

The pipeline is `InsightFace -> face embedding -> DBSCAN ->
quality-weighted person prototype`. Use `--ctx-id -1` for CPU. The first
InsightFace run may download the `buffalo_l` model files.

Apply manual corrections to the saved index:

```powershell
python scripts/people_feedback.py --people-path data/people.json --merge 3 7
python scripts/people_feedback.py --people-path data/people.json --remove 2 PHOTO_ID
```

Equivalent CLI commands are `memora people-cluster`, `memora people-merge` and
`memora people-remove`. The API exposes `POST /people/cluster`, `GET /people`,
`POST /people/merge` and `POST /people/remove-photo`.

阶段 3 测试记录见
[`docs/stage3_people_clustering_test.md`](docs/stage3_people_clustering_test.md)。

## Stage 4: event discovery ablation

Event discovery exposes the three planned strategies:

- `time_only`: temporal gap baseline;
- `time_clip`: time plus CLIP embedding distance;
- `time_clip_gps`: time plus CLIP plus GPS distance.
- `strict_event`: trusts EXIF capture time, treats filesystem time as upload-batch
  metadata only, and uses conservative CLIP-only matching for photos without
  EXIF.

Run an individual strategy with the CLI or API:

```powershell
memora events --index-path data/index.json --strategy time_clip_gps
# GET /events?strategy=time_only
```

The strict comparison is:

```powershell
memora events --index-path data/index.json --strategy strict_event
```

Its initial CLIP thresholds are intentionally conservative (`0.86` for
photos in the same upload batch and `0.92` across batches or between EXIF and
non-EXIF photos). These are starting points and should be calibrated against
the manually labelled event set.

Ver4 adds `strict_event_people`, journey discovery, EventName, JourneyName
and JourneyNote generation. See
[`docs/ver4_events_journeys.md`](docs/ver4_events_journeys.md) for the data
model, CLI examples and annotation format.

Journey locations are inferred from EXIF GPS by default: recurring clusters
identify the home region, other clusters become destination candidates, and
the optional offline geocoder supplies nearest-locality names. Install it with
`pip install -e ".[geocoding]"`. Manual home/destination arguments remain
available as explicit overrides.

Narrative generation automatically prefers the configured LLM when
`MEMORA_LLM_URL` and `MEMORA_LLM_MODEL` exist. The first endpoint failure opens
a circuit breaker and all remaining names fall back to deterministic templates.
Use `--no-llm` to force template-only generation.

### Configure LLM generation

Memora uses an OpenAI-compatible Chat Completions endpoint. Cloud providers
normally require an API key; a local compatible server may not require one.
Set the variables in the same PowerShell terminal before starting FastAPI:

```powershell
$env:MEMORA_LLM_URL = "https://your-provider.example/v1/chat/completions"
$env:MEMORA_LLM_MODEL = "your-model-name"
$env:MEMORA_LLM_API_KEY = "your-api-key"
python -m uvicorn memora.api.main:app --reload --host 127.0.0.1 --port 8000
```

For a local OpenAI-compatible endpoint, omit the API key when authentication
is disabled:

```powershell
$env:MEMORA_LLM_URL = "http://localhost:YOUR_PORT/v1/chat/completions"
$env:MEMORA_LLM_MODEL = "your-local-model"
Remove-Item Env:MEMORA_LLM_API_KEY -ErrorAction SilentlyContinue
python -m uvicorn memora.api.main:app --reload --host 127.0.0.1 --port 8000
```

Environment variables are read when the backend starts, so restart FastAPI
after changing them. Never commit an API key to the repository or frontend.

Create a JSON object mapping each photo ID to a ground-truth event ID and
evaluate all three strategies:

```powershell
python scripts/evaluate_events.py data/event_labels.json --index-path data/index.json
```

The output contains pairwise Event Precision, Event Recall and Event F1. This
metric is invariant to the numeric IDs assigned to discovered events.

## Stage 5: similar shots and best shot

`group_similar` combines pHash distance, CLIP cosine similarity and a capture
time window. Each returned `SimilarGroup.representative_id` is the quality-
ranked best shot in that group. The quality score contains sharpness,
exposure, face quality and composition signals, and is stored in each indexed
photo's `quality` object.

```powershell
memora similar --index-path data/index.json --phash-distance 10 `
  --visual-similarity 0.90 --time-window-seconds 30
```

## Stage 6: vector index benchmark

The exact NumPy index remains the recall ground truth. The benchmark compares
NumPy exact, FAISS Flat, hnswlib HNSW and Qdrant HNSW:

```powershell
pip install -e ".[vector]"
python scripts/benchmark.py --index-path data/index.json --queries 100
```

Results include Recall@10 against NumPy exact, mean latency, P95 latency and
estimated index memory. To run only the dependency-free baseline:

```powershell
python scripts/benchmark.py --count 1000 --dimension 256 --backend numpy_exact
```

The HNSW benchmark uses FAISS's native `IndexHNSWFlat`, so the normal
`[vector]` installation does not need `hnswlib` or Visual Studio Build Tools.
`hnswlib` is kept only as an optional alternative backend. On Windows, pip
may need to compile it locally and therefore requires MSVC 14.0+:

```powershell
pip install -e ".[hnsw]"
```

Alternatively install `hnswlib` from conda-forge. FAISS Flat, FAISS HNSW and
Qdrant HNSW are included in the normal `[vector]` extra.

The Qdrant benchmark uses an in-memory Qdrant client by default. Pass the
`QdrantHnswIndex` adapter a server URL when deploying it as a service.

## Stage 7: Immich integration

Immich remains the system of record for uploads, albums and the timeline.
Memora reads image metadata and preview thumbnails through the Immich API,
indexes them locally, and returns the original Immich asset UUID in every AI
result. No direct access to Immich's upload-library filesystem or database is
required.

Create an API key in Immich and configure Memora (do not commit the key):

```powershell
$env:MEMORA_IMMICH_URL = "http://localhost:2283"
$env:MEMORA_IMMICH_API_KEY = "your-api-key"
$env:MEMORA_ENCODER = "open_clip"
memora immich-status
memora immich-sync --encoder open_clip --index-path data/index.json
```

Incremental sync reuses an embedding when both the Immich `updatedAt` value
and cached preview are unchanged. Use `--force` after changing the encoder or
model. `--prune-missing` removes missing Immich assets from the Memora index,
but never deletes assets from Immich.

The HTTP integration endpoints are:

- `GET /immich/status` — verify connectivity and report the server version;
- `POST /immich/sync` — incrementally refresh the Memora index;
- `GET /immich/assets/{asset_id}/thumbnail` — authenticated image proxy for a UI;
- `POST /immich/albums` — publish selected search/event/best-shot IDs as an Immich album.

Search results from Immich include `immich_asset_id` and `thumbnail_url`, so a
frontend can render Memora results without receiving the Immich API key. See
[`docs/stage7_immich_integration.md`](docs/stage7_immich_integration.md) for the
deployment contract, permissions and example requests.

### Memora frontend

The independent dashboard is built with Vue 3, TypeScript and Vite. Start the
FastAPI service first, then run the frontend in a second terminal:

#### Two-terminal local test (Windows PowerShell)

Open two PowerShell terminals in the project root (`D:\00A\project\AIbum`).

Terminal 1 — start the FastAPI backend:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn memora.api.main:app --reload --host 127.0.0.1 --port 8000
```

Keep this terminal running. You can verify the backend at
`http://localhost:8000/docs`.

Terminal 2 — start the Vue frontend:

```powershell
cd frontend
npm install
npm run dev
```

After frontend source changes, stop any older Vite process with `Ctrl+C` and
run `npm run dev` again. The dev command forces Vite to refresh its cache and
will fail instead of silently switching to another port when `5173` is already
occupied.

Run `npm install` only the first time, or after `package.json` changes. Open
the frontend at `http://localhost:5173`. The Vite development server proxies
`/api/*` requests to the FastAPI service on port `8000`.

The landing page is a project console: users can select photos or a complete
local folder, preserve its subfolder structure, and create an isolated photo
workspace. Inside a workspace they can run indexing, natural-language search,
InsightFace clustering, event discovery, similar-shot grouping and best-shot
ranking. Results can be exported as a JSON project manifest, a CSV photo index,
or a ZIP containing the highest-ranked original photos.

Each project is stored under `data/projects/<project-id>/` with separate
uploads, indexes, people clusters and generated exports. This directory is
private user data and is ignored by Git.

## API

Start the server with `uvicorn memora.api.main:app --reload` and use:

- `GET /health`
- `POST /index` with `{ "directory": "...", "index_path": "..." }`
- `POST /search` with `{ "query": "...", "top_k": 20 }`
- `POST /people/cluster`
- `GET /people`
- `POST /people/merge`
- `POST /people/remove-photo`
- `POST /people/name`
- `GET /events`
- `POST /journeys/discover`
- `GET /similar-groups`
- `GET /quality/{photo_id}`
- `GET /immich/status`
- `POST /immich/sync`
- `GET /immich/assets/{asset_id}/thumbnail`
- `POST /immich/albums`
- `GET/POST /projects`
- `POST /projects/{project_id}/files`
- `POST /projects/{project_id}/analyze`
- `GET /projects/{project_id}/photos`
- `POST /projects/{project_id}/search`
- `GET /projects/{project_id}/events`
- `GET /projects/{project_id}/similar-groups`
- `GET /projects/{project_id}/best-shots`
- `GET/POST /projects/{project_id}/people[/cluster]`
- `GET /projects/{project_id}/export/{manifest|photos.csv|best-shots.zip}`

## Architecture

```text
raw photo
  |-- metadata/exif.py -------- timestamp, GPS, camera
  |-- encoders/clip_encoder.py - image/text embeddings
  |-- quality/ ------------------ blur, exposure, best shot
  |-- duplicate/ ---------------- pHash + visual similarity
  `-- clustering/ --------------- events and people
                    |
              retrieval/index.py
                    |
                 FastAPI / CLI
```

`HashImageEncoder` and NumPy brute-force search are intentionally included as baselines. They make the algorithm pipeline testable without model downloads. `OpenCLIPEncoder`, `InsightFaceEncoder` and `QdrantStore` are optional adapters rather than hard requirements.

## Roadmap

### v1.0: usable local photo intelligence

- Semantic image-text retrieval with query-expansion and metadata filters.
- People clustering, event discovery, similar-shot grouping and best-shot ranking.
- Local project workspaces, result export and a Vue dashboard.
- Optional Immich synchronization without direct access to Immich storage or database.

### v1.1: quality, control and deployment

- Improve retrieval, event and people-clustering accuracy with labelled evaluation sets and configurable thresholds.
- Add human feedback workflows for correcting people, events, rankings and generated names.
- Improve incremental indexing, vector-store configuration and GPU/CPU deployment guidance.
- Add stronger observability, error handling and regression coverage for API and frontend workflows.

### v2.0: collaborative and production-ready photo intelligence

- Introduce multi-user workspaces, access control and persistent service deployment.
- Support scalable background processing, distributed vector storage and large-library synchronization.
- Provide richer album, journey and story generation with auditable user controls.
- Expand integration APIs so photo managers and other clients can consume AI results directly.

The roadmap describes product milestones. Package and frontend versions remain
tracked in `pyproject.toml` and `frontend/package.json` respectively.
