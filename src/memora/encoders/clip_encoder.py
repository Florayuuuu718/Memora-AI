import hashlib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PIL import Image


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


class VisionEncoder(ABC):
    dimension: int

    @abstractmethod
    def encode_image(self, path: str | Path) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def encode_images(self, paths: Iterable[str | Path]) -> np.ndarray:
        return np.vstack([self.encode_image(path) for path in paths])

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.encode_text(text) for text in texts])


class HashImageEncoder(VisionEncoder):
    """Dependency-light baseline: color/spatial statistics plus stable text hashing."""

    dimension = 256

    def _hash_features(self, value: str) -> np.ndarray:
        output = np.zeros(self.dimension, dtype=np.float32)
        tokens = value.lower().replace("_", " ").replace("-", " ").split()
        for token in tokens or [value.lower()]:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(0, 24, 2):
                index = int.from_bytes(digest[offset:offset + 2], "big") % self.dimension
                output[index] += 1.0 if digest[offset] % 2 else -1.0
        return normalize(output)

    def encode_text(self, text: str) -> np.ndarray:
        return self._hash_features(text)

    def encode_image(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            image = image.convert("RGB").resize((32, 32))
            pixels = np.asarray(image, dtype=np.float32) / 255.0
        features = [pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))]
        for row in np.array_split(pixels, 4, axis=0):
            for block in np.array_split(row, 4, axis=1):
                features.append(block.mean(axis=(0, 1)))
        vector = np.concatenate(features)
        digest = hashlib.sha256(Path(path).read_bytes()).digest()
        tiled = np.resize(vector, self.dimension).astype(np.float32)
        for i, byte in enumerate(digest):
            tiled[(i * 17) % self.dimension] += (byte - 127.5) / 127.5
        return normalize(tiled)


class OpenCLIPEncoder(VisionEncoder):
    """Lazy OpenCLIP adapter. Model packages are intentionally optional."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
    ) -> None:
        import open_clip  # type: ignore
        import torch  # type: ignore

        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()
        self.dimension = int(self.model.visual.output_dim)

    def encode_image(self, path: str | Path) -> np.ndarray:
        with Image.open(path) as image:
            tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            vector = self.model.encode_image(tensor).cpu().numpy()[0]
        return normalize(vector)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text])[0]

    def encode_texts(self, texts: Iterable[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        tokens = self.tokenizer(values).to(self.device)
        with self.torch.no_grad():
            vectors = self.model.encode_text(tokens).cpu().numpy()
        return np.asarray([normalize(vector) for vector in vectors], dtype=np.float32)


def create_encoder(kind: str = "lightweight", **kwargs: str) -> VisionEncoder:
    if kind.lower() in {"open_clip", "openclip", "clip"}:
        return OpenCLIPEncoder(
            kwargs.get("model_name", "ViT-B-32"),
            kwargs.get("pretrained", "laion2b_s34b_b79k"),
            kwargs.get("device"),
        )
    return HashImageEncoder()
