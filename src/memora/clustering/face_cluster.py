import numpy as np

from memora.clustering.event_cluster import cluster_embeddings


def quality_weighted_prototype(embeddings: np.ndarray, qualities: np.ndarray | None = None) -> np.ndarray:
    if len(embeddings) == 0:
        return np.zeros(0, dtype=np.float32)
    weights = np.ones(len(embeddings), dtype=np.float32) if qualities is None else np.maximum(qualities, 1e-6)
    result = (embeddings * weights[:, None]).sum(axis=0) / weights.sum()
    return result / max(float(np.linalg.norm(result)), 1e-12)


def cluster_faces(face_embeddings: np.ndarray, eps: float = 0.35, min_samples: int = 2) -> dict[int, list[int]]:
    labels = cluster_embeddings(face_embeddings, eps=eps, min_samples=min_samples)
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        if label >= 0:
            groups.setdefault(int(label), []).append(index)
    return groups
