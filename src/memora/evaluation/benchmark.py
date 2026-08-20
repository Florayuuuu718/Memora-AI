from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any

import numpy as np

from memora.retrieval.vector_store import create_vector_index


@dataclass
class BenchmarkResult:
    method: str
    count: int
    elapsed_ms: float
    p95_latency_ms: float = 0.0
    recall_at_10: float | None = None
    memory_mb: float = 0.0
    available: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _query_ids(index, queries: np.ndarray, top_k: int) -> tuple[list[list[int]], list[float]]:
    ids: list[list[int]] = []
    latencies: list[float] = []
    for query in queries:
        start = perf_counter()
        hits = index.search(query, top_k=top_k)
        latencies.append((perf_counter() - start) * 1000.0)
        ids.append([item[0] for item in hits])
    return ids, latencies


def _recall_at_k(actual: list[int], expected: list[int], k: int) -> float:
    if not expected:
        return 0.0
    return len(set(actual[:k]) & set(expected[:k])) / len(set(expected[:k]))


def benchmark_exact_search(matrix: np.ndarray, queries: np.ndarray) -> BenchmarkResult:
    index = create_vector_index("numpy_exact", matrix)
    _, latencies = _query_ids(index, np.asarray(queries, dtype=np.float32), top_k=10)
    return BenchmarkResult(
        method="numpy_exact",
        count=len(matrix),
        elapsed_ms=float(np.mean(latencies)) if latencies else 0.0,
        p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
        memory_mb=index.memory_bytes / (1024 * 1024),
    )


def benchmark_vector_indexes(
    matrix: np.ndarray,
    queries: np.ndarray,
    *,
    backends: tuple[str, ...] = ("numpy_exact", "faiss_flat", "hnsw", "qdrant_hnsw"),
    top_k: int = 10,
    **backend_kwargs: Any,
) -> list[BenchmarkResult]:
    """Compare exact and approximate indexes against NumPy ground truth."""
    values = np.asarray(matrix, dtype=np.float32)
    query_values = np.asarray(queries, dtype=np.float32)
    exact = create_vector_index("numpy_exact", values)
    recall_k = 10
    search_k = max(top_k, recall_k)
    expected, _ = _query_ids(exact, query_values, top_k=search_k)
    results: list[BenchmarkResult] = []
    for backend in backends:
        try:
            index = exact if backend in {"numpy", "numpy_exact", "numpy_brute_force"} else create_vector_index(backend, values, **backend_kwargs)
            actual, latencies = _query_ids(index, query_values, top_k=search_k)
            recalls = [_recall_at_k(found, truth, recall_k) for found, truth in zip(actual, expected)]
            results.append(BenchmarkResult(
                method=index.name,
                count=len(values),
                elapsed_ms=float(np.mean(latencies)) if latencies else 0.0,
                p95_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
                recall_at_10=float(np.mean(recalls)) if recalls else 0.0,
                memory_mb=index.memory_bytes / (1024 * 1024),
            ))
        except (ImportError, RuntimeError, ValueError) as exc:
            results.append(BenchmarkResult(
                method=backend,
                count=len(values),
                elapsed_ms=0.0,
                available=False,
                error=str(exc),
            ))
    return results
