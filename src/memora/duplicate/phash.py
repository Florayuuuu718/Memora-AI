from pathlib import Path

import numpy as np
from PIL import Image


def phash(path: str | Path, size: int = 8) -> str:
    with Image.open(path).convert("L") as image:
        pixels = np.asarray(image.resize((size * 4, size * 4)), dtype=np.float32)
    # DCT-like low-frequency projection without requiring scipy/opencv.
    basis = np.cos(np.pi * (np.arange(size * 4)[:, None] + 0.5) * np.arange(size * 4)[None, :] / (size * 4))
    low = basis[:size] @ pixels @ basis[:size].T
    median = np.median(low[1:, 1:])
    bits = (low >= median).flatten()
    return "".join("1" if bit else "0" for bit in bits)


def hamming_distance(left: str, right: str) -> int:
    return sum(a != b for a, b in zip(left, right)) + abs(len(left) - len(right))
