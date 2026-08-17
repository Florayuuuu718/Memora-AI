# Memora AI

Memora AI is an independent photo-understanding and retrieval service. It is designed to sit beside a self-hosted photo manager such as Immich:

```text
Immich (gallery, upload, albums)  -->  Memora AI (algorithms, retrieval, ranking)
```

The project focuses on the parts that are useful for an image-algorithm portfolio:

- semantic image/text retrieval;
- time + visual + GPS event discovery;
- near-duplicate and burst-shot grouping;
- image quality and best-shot ranking;
- pluggable OpenCLIP, InsightFace and Qdrant integrations;
- reproducible evaluation and brute-force baselines.

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

## API

Start the server with `uvicorn memora.api.main:app --reload` and use:

- `GET /health`
- `POST /index` with `{ "directory": "...", "index_path": "..." }`
- `POST /search` with `{ "query": "...", "top_k": 20 }`
- `POST /people/cluster`
- `GET /people`
- `POST /people/merge`
- `POST /people/remove-photo`
- `GET /events`
- `GET /similar-groups`
- `GET /quality/{photo_id}`

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

## Algorithm roadmap

1. Retrieval: raw embedding -> prompt ensemble -> query expansion -> metadata filtering.
2. People: face embeddings -> DBSCAN -> quality-weighted person prototypes -> human constraints.
3. Events: time only -> time + visual -> time + visual + GPS ablation.
4. Ranking: pHash + CLIP + time for similar shots, then sharpness/exposure/face quality for best shot.

The current repository implements the first usable version of all four paths and leaves model-specific adapters replaceable.
