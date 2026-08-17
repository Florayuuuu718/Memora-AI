from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class PhotoRecord:
    id: str
    path: str
    width: int = 0
    height: int = 0
    captured_at: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    camera: str | None = None
    embedding: list[float] = field(default_factory=list)
    phash: str | None = None
    quality: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PhotoRecord":
        return cls(**value)

    @property
    def timestamp(self) -> datetime | None:
        if not self.captured_at:
            return None
        try:
            return datetime.fromisoformat(self.captured_at.replace("Z", "+00:00"))
        except ValueError:
            return None


@dataclass
class SearchResult:
    photo_id: str
    path: str
    score: float
    captured_at: str | None = None


@dataclass
class EventGroup:
    id: int
    photo_ids: list[str]
    start: str | None
    end: str | None
    centroid_latitude: float | None = None
    centroid_longitude: float | None = None


@dataclass
class SimilarGroup:
    id: int
    photo_ids: list[str]
    representative_id: str | None = None

