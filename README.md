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

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,vision]"
memora index .\photos --index-path .\data\index.json
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

OpenCLIP embeddings are model-specific, so rebuild the index after switching
encoders:

```powershell
memora index .\photos --index-path .\data\index-openclip.json --encoder open_clip
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

## API

Start the server with `uvicorn memora.api.main:app --reload` and use:

- `GET /health`
- `POST /index` with `{ "directory": "...", "index_path": "..." }`
- `POST /search` with `{ "query": "...", "top_k": 20 }`
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
