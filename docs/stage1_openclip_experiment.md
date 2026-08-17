# Stage 1 OpenCLIP Retrieval Experiment

## Experiment setup

- Index: `data/index-openclip.json`
- Query: `海边`
- Strict relevant photos: `test (44).jpg`, `test (45).jpg`, `test (46).jpg`
- Boundary photos: `test (37).jpg`, `test (38).jpg`
- Boundary photos are excluded from the strict metrics because their labels are uncertain.
- Search backend: NumPy exact cosine similarity.
- Image encoder: OpenCLIP `ViT-B-32` with `laion2b_s34b_b79k` pretrained weights.

The relevant photo IDs are stored in `data/retrieval_cases.json`.

## Results

| Strategy | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Raw CLIP | 0.333 | 0.333 | 1.000 |
| Prompt Ensemble | 0.333 | 0.667 | 1.000 |
| Query Enhancement | 0.333 | 1.000 | 1.000 |

## Interpretation

The three strategies use the same image embeddings and the same exact vector
search. They differ only in how the text query is encoded:

- `raw_clip` encodes `海边` once;
- `prompt_ensemble` averages five prompt templates around `海边`;
- `query_enhancement` averages `海边`, `a beach`, `people at the beach`,
  `a seaside landscape`, and `ocean and beach`.

For this query, Query Enhancement is the strongest strategy: it retrieves all
three strict positive photos in the top five. Prompt Ensemble retrieves two of
the three in the top five, while Raw CLIP retrieves one.

## Limitations

This is a single-query pilot experiment, not a statistically strong benchmark.
The next evaluation should add multiple labeled queries such as travel,
birthday, hotpot, river, and landscape before making a general claim about
which strategy is best.

## Reproduce

```powershell
python scripts/evaluate_retrieval.py data/retrieval_cases.json `
  --index-path data/index-openclip.json `
  --encoder open_clip
```
