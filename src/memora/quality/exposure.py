import numpy as np
from PIL import Image


def exposure_score(path: str) -> float:
    with Image.open(path).convert("L") as image:
        values = np.asarray(image, dtype=np.float32) / 255.0
    under = float((values < 0.05).mean())
    over = float((values > 0.95).mean())
    return max(0.0, 1.0 - 0.5 * (under + over))
