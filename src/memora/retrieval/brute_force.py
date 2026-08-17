import numpy as np

from memora.models import PhotoRecord, SearchResult


def cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = matrix / np.maximum(norms, 1e-12)
    return safe @ query


def search(records: list[PhotoRecord], query: np.ndarray, top_k: int = 20, min_score: float | None = None) -> list[SearchResult]:
    usable = [(index, record) for index, record in enumerate(records) if record.embedding]
    if not usable:
        return []
    matrix = np.asarray([record.embedding for _, record in usable], dtype=np.float32)
    scores = cosine_scores(query, matrix)
    order = np.argsort(-scores)
    output = []
    for position in order[:max(0, top_k)]:
        score = float(scores[position])
        if min_score is not None and score < min_score:
            continue
        record = usable[int(position)][1]
        output.append(SearchResult(record.id, record.path, score, record.captured_at))
    return output

