import numpy as np
from PIL import Image, ImageFilter


def laplacian_variance(path: str) -> float:
    try:
        import cv2  # type: ignore
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return float(cv2.Laplacian(image, cv2.CV_64F).var()) if image is not None else 0.0
    except ImportError:
        with Image.open(path).convert("L") as image:
            edges = np.asarray(image, dtype=np.float32) - np.asarray(image.filter(ImageFilter.GaussianBlur(1)), dtype=np.float32)
            return float(edges.var())
