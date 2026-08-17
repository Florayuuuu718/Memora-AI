# Stage 1 Three-Group Retrieval Experiment

## Dataset and labels

The experiment uses the normalized JPEG dataset and OpenCLIP index:

- Images: `photos_prepared/`
- Index: `data/index-openclip-prepared.json`
- Labels: `data/retrieval_cases_prepared.json`
- Search: NumPy exact cosine similarity
- Image encoder: OpenCLIP `ViT-B-32`, `laion2b_s34b_b79k`

The three manually labeled groups are:

| Group | Query | Positives |
| --- | --- | ---: |
| Beach | `海边` | 9 |
| Small animals | `小动物` | 17 |
| Food | `食物` | 12 |

The food group includes fruit. Photos 19, 20 and 21 are recorded as the fruit
subset in the case file.

## Recall results by group

### Beach

| Strategy | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Raw CLIP | 0.000 | 0.111 | 0.111 |
| Prompt Ensemble | 0.000 | 0.222 | 0.333 |
| Query Enhancement | 0.111 | 0.444 | 0.889 |

### Small animals

| Strategy | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Raw CLIP | 0.000 | 0.118 | 0.118 |
| Prompt Ensemble | 0.000 | 0.118 | 0.118 |
| Query Enhancement | 0.059 | 0.118 | 0.176 |

### Food, including fruit

| Strategy | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Raw CLIP | 0.000 | 0.250 | 0.333 |
| Prompt Ensemble | 0.083 | 0.333 | 0.417 |
| Query Enhancement | 0.083 | 0.333 | 0.417 |

## Mean results across the three groups

| Strategy | Recall@1 | Recall@5 | Recall@10 |
| --- | ---: | ---: | ---: |
| Raw CLIP | 0.000 | 0.160 | 0.187 |
| Prompt Ensemble | 0.028 | 0.224 | 0.289 |
| Query Enhancement | 0.084 | 0.298 | 0.494 |

## Metadata coverage

| Group | GPS available | EXIF time available |
| --- | ---: | ---: |
| Beach | 1/9 | 6/9 |
| Small animals | 14/17 | 14/17 |
| Food | 8/12 | 9/12 |

## Conclusions

1. Query Enhancement remains the strongest overall strategy. Its mean Recall@10
   is 0.494, compared with 0.289 for Prompt Ensemble and 0.187 for Raw CLIP.
2. The beach group demonstrates a real semantic ambiguity: visual CLIP
   similarity can confuse a lake, such as Erhai, with a sea or coastline.
3. GPS can help disambiguate this only when the candidate photo actually has
   GPS. The beach group currently has GPS for only one of nine positives, so a
   GPS filter cannot yet be the default condition for this group.
4. The small-animal and food results show that the current bilingual expansion
   dictionary is incomplete. `小动物` and `食物` currently fall back to generic
   expansion instead of receiving dedicated English semantic candidates.
5. The next optimization should be hybrid retrieval: use CLIP for semantic
   recall, then apply time/GPS metadata as an optional hard filter or reranking
   signal when metadata is available.

## Reproduce

```powershell
python scripts/evaluate_retrieval.py data/retrieval_cases_prepared.json `
  --index-path data/index-openclip-prepared.json `
  --encoder open_clip
```
