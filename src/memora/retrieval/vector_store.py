import json
from pathlib import Path

import numpy as np

from memora.models import PhotoRecord


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


class QdrantStore:
    """Optional adapter; local NumPy remains the default for reproducible baselines."""

    def __init__(self, collection: str = "memora_photos", url: str = "http://localhost:6333") -> None:
        from qdrant_client import QdrantClient  # type: ignore
        self.client = QdrantClient(url=url)
        self.collection = collection

