import argparse
import json

import numpy as np

from memora.evaluation.benchmark import benchmark_vector_indexes
from memora.retrieval.vector_store import NumpyVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark NumPy, FAISS, HNSW and Qdrant vector indexes")
    parser.add_argument("--index-path", help="Existing Memora index JSON; otherwise use synthetic vectors")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--dimension", type=int, default=256)
    parser.add_argument("--backend", action="append", dest="backends")
    args = parser.parse_args()
    rng = np.random.default_rng(7)
    if args.index_path:
        matrix = NumpyVectorStore.load(args.index_path).matrix()
        if len(matrix) == 0:
            raise SystemExit("index contains no embeddings")
        queries = rng.normal(size=(args.queries, matrix.shape[1])).astype(np.float32)
    else:
        matrix = rng.normal(size=(args.count, args.dimension)).astype(np.float32)
        queries = rng.normal(size=(args.queries, args.dimension)).astype(np.float32)
    backends = tuple(args.backends or ("numpy_exact", "faiss_flat", "hnsw", "qdrant_hnsw"))
    print(json.dumps([result.to_dict() for result in benchmark_vector_indexes(matrix, queries, backends=backends)], indent=2))


if __name__ == "__main__":
    main()
