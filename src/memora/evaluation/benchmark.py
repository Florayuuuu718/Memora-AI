from dataclasses import dataclass
from time import perf_counter

import numpy as np

from memora.retrieval.brute_force import cosine_scores


@dataclass
class BenchmarkResult:
    method: str
    count: int
    elapsed_ms: float


def benchmark_exact_search(matrix: np.ndarray, queries: np.ndarray) -> BenchmarkResult:
    start = perf_counter()
    for query in queries:
        cosine_scores(query, matrix)
    elapsed_ms = (perf_counter() - start) * 1000.0 / max(len(queries), 1)
    return BenchmarkResult("numpy_exact", len(matrix), elapsed_ms)

