import json

import numpy as np

from memora.evaluation.benchmark import benchmark_exact_search


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    result = benchmark_exact_search(rng.normal(size=(1000, 256)).astype(np.float32), rng.normal(size=(20, 256)).astype(np.float32))
    print(json.dumps(result.__dict__, indent=2))

