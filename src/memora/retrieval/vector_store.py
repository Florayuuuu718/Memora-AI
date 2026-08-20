import json
from pathlib import Path
from typing import Any

import numpy as np

from memora.models import PhotoRecord
from memora.retrieval.brute_force import cosine_scores


class NumpyVectorStore:
    """Exact local store used as the transparent retrieval baseline."""

    def __init__(self, records: list[PhotoRecord] | None = None) -> None:
        self.records = records or []

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([record.to_dict() for record in self.records], ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorStore":
        path = Path(path)
        if not path.exists():
            return cls([])
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls([PhotoRecord.from_dict(value) for value in values])

    def matrix(self) -> np.ndarray:
        return np.asarray([record.embedding for record in self.records if record.embedding], dtype=np.float32)


class NumpyExactIndex:
    """Exact cosine index used as the recall and latency ground truth."""

    name = "numpy_exact"

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = _prepare_matrix(matrix)

    @property
    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        scores = cosine_scores(query, self.matrix)
        order = np.argsort(-scores, kind="stable")[: max(0, top_k)]
        return [(int(index), float(scores[index])) for index in order]


def _prepare_matrix(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("vector matrix must be two-dimensional")
    if len(values) == 0:
        raise ValueError("vector matrix must not be empty")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


class FaissFlatIndex:
    """Optional FAISS exact inner-product index (cosine after normalization)."""

    name = "faiss_flat"

    def __init__(self, matrix: np.ndarray) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("FAISS backend requires the 'faiss-cpu' package") from exc
        self.matrix = _prepare_matrix(matrix)
        self._index = faiss.IndexFlatIP(self.matrix.shape[1])
        self._index.add(self.matrix)

    @property
    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        values = _prepare_matrix(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, indexes = self._index.search(values, min(max(0, top_k), len(self.matrix)))
        return [(int(index), float(score)) for index, score in zip(indexes[0], scores[0]) if index >= 0]


class FaissHnswIndex:
    """FAISS HNSW index; avoids a separate native hnswlib install on Windows."""

    name = "hnsw"

    def __init__(self, matrix: np.ndarray, *, ef_construction: int = 200, m: int = 16, ef: int = 64) -> None:
        try:
            import faiss  # type: ignore
        except ImportError as exc:
            raise ImportError("FAISS HNSW backend requires the 'faiss-cpu' package") from exc
        self.matrix = _prepare_matrix(matrix)
        metric = getattr(faiss, "METRIC_INNER_PRODUCT", 0)
        self._index = faiss.IndexHNSWFlat(self.matrix.shape[1], m, metric)
        self._index.hnsw.efConstruction = ef_construction
        self._index.hnsw.efSearch = ef
        self._index.add(self.matrix)

    @property
    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes + len(self.matrix) * 16 * 4)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        values = _prepare_matrix(np.asarray(query, dtype=np.float32).reshape(1, -1))
        scores, indexes = self._index.search(values, min(max(0, top_k), len(self.matrix)))
        return [(int(index), float(score)) for index, score in zip(indexes[0], scores[0]) if index >= 0]


class HnswIndex:
    """Optional hnswlib approximate cosine index."""

    name = "hnsw"

    def __init__(self, matrix: np.ndarray, *, ef_construction: int = 200, m: int = 16, ef: int = 64) -> None:
        try:
            import hnswlib  # type: ignore
        except ImportError as exc:
            raise ImportError("HNSW backend requires the 'hnswlib' package") from exc
        self.matrix = _prepare_matrix(matrix)
        self._index = hnswlib.Index(space="cosine", dim=self.matrix.shape[1])
        self._index.init_index(max_elements=len(self.matrix), ef_construction=ef_construction, M=m)
        self._index.add_items(self.matrix, np.arange(len(self.matrix)))
        self._index.set_ef(max(ef, 1))

    @property
    def memory_bytes(self) -> int:
        # Includes the vector matrix; graph overhead is implementation/version dependent.
        return int(self.matrix.nbytes + len(self.matrix) * 16 * 4)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        values = _prepare_matrix(np.asarray(query, dtype=np.float32).reshape(1, -1))
        indexes, distances = self._index.knn_query(values, k=min(max(0, top_k), len(self.matrix)))
        return [(int(index), float(1.0 - distance)) for index, distance in zip(indexes[0], distances[0])]


class QdrantHnswIndex:
    """Qdrant HNSW adapter, using local in-memory Qdrant by default."""

    name = "qdrant_hnsw"

    def __init__(
        self,
        matrix: np.ndarray,
        *,
        collection: str = "memora_benchmark",
        url: str | None = None,
        m: int = 16,
        ef_construct: int = 200,
    ) -> None:
        try:
            from qdrant_client import QdrantClient, models  # type: ignore
        except ImportError as exc:
            raise ImportError("Qdrant backend requires the 'qdrant-client' package") from exc
        self.matrix = _prepare_matrix(matrix)
        self.collection = collection
        self._models = models
        self._client = QdrantClient(location=":memory:") if not url or url in {":memory:", "memory"} else QdrantClient(url=url)
        if self._client.collection_exists(collection):
            self._client.delete_collection(collection)
        hnsw_config_type = getattr(models, "HnswConfigDiff", None) or getattr(models, "HnswConfig")
        self._client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(size=self.matrix.shape[1], distance=models.Distance.COSINE),
            hnsw_config=hnsw_config_type(m=m, ef_construct=ef_construct),
        )
        self._client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(id=index, vector=vector.tolist())
                for index, vector in enumerate(self.matrix)
            ],
        )

    @property
    def memory_bytes(self) -> int:
        return int(self.matrix.nbytes + len(self.matrix) * 16 * 4)

    def search(self, query: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        vector = _prepare_matrix(np.asarray(query, dtype=np.float32).reshape(1, -1))[0].tolist()
        limit = min(max(0, top_k), len(self.matrix))
        if hasattr(self._client, "query_points"):
            response = self._client.query_points(collection_name=self.collection, query=vector, limit=limit)
            hits = response.points
        else:
            hits = self._client.search(collection_name=self.collection, query_vector=vector, limit=limit)
        return [(int(hit.id), float(hit.score)) for hit in hits]


def create_vector_index(name: str, matrix: np.ndarray, **kwargs: Any):
    """Create one of the four benchmark backends by stable CLI name."""
    normalized = name.lower().replace("-", "_")
    if normalized in {"numpy", "numpy_exact", "numpy_brute_force"}:
        return NumpyExactIndex(matrix)
    if normalized in {"faiss", "faiss_flat"}:
        return FaissFlatIndex(matrix)
    if normalized == "hnsw":
        return FaissHnswIndex(matrix, **kwargs)
    if normalized == "hnswlib":
        return HnswIndex(matrix, **kwargs)
    if normalized in {"qdrant", "qdrant_hnsw"}:
        return QdrantHnswIndex(matrix, **kwargs)
    raise ValueError(f"unknown vector backend: {name}")


# Descriptive aliases kept public for experiment notebooks and external callers.
NumpyBruteForceIndex = NumpyExactIndex
FAISSFlatIndex = FaissFlatIndex
FAISSHNSWIndex = FaissHnswIndex
HNSWIndex = FaissHnswIndex
HnswlibIndex = HnswIndex
QdrantHNSWIndex = QdrantHnswIndex


class QdrantStore:
    """Optional adapter; local NumPy remains the default for reproducible baselines."""

    def __init__(self, collection: str = "memora_photos", url: str = "http://localhost:6333") -> None:
        from qdrant_client import QdrantClient  # type: ignore
        self.client = QdrantClient(url=url)
        self.collection = collection
