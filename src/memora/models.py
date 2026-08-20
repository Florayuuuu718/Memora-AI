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
    captured_at_source: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    gps_source: str | None = None
    camera: str | None = None
    embedding: list[float] = field(default_factory=list)
    phash: str | None = None
    quality: dict[str, float] = field(default_factory=dict)
    source: str = "local"
    source_updated_at: str | None = None
    original_filename: str | None = None

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
    source: str = "local"


@dataclass
class EventGroup:
    id: int
    photo_ids: list[str]
    start: str | None
    end: str | None
    centroid_latitude: float | None = None
    centroid_longitude: float | None = None
    evidence: str | None = None
    person_ids: list[int] = field(default_factory=list)
    activity_tags: list[str] = field(default_factory=list)
    object_tags: list[str] = field(default_factory=list)
    duration_minutes: float | None = None
    name: str | None = None
    summary: str | None = None
    name_source: str | None = None


@dataclass
class JourneyGroup:
    id: int
    event_ids: list[int] = field(default_factory=list)
    loose_photo_ids: list[str] = field(default_factory=list)
    start: str | None = None
    end: str | None = None
    home_name: str | None = None
    destination_names: list[str] = field(default_factory=list)
    person_ids: list[int] = field(default_factory=list)
    activity_tags: list[str] = field(default_factory=list)
    name: str | None = None
    note: str | None = None
    name_source: str | None = None
    confidence: float = 0.0
    stops: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SimilarGroup:
    id: int
    photo_ids: list[str]
    representative_id: str | None = None


@dataclass
class FaceRecord:
    id: str
    photo_id: str
    bbox: list[float] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    det_score: float = 0.0
    appearance_embedding: list[float] = field(default_factory=list)


@dataclass
class PersonGroup:
    id: int
    face_ids: list[str] = field(default_factory=list)
    photo_ids: list[str] = field(default_factory=list)
    prototype: list[float] = field(default_factory=list)
    removed_photo_ids: list[str] = field(default_factory=list)
    name: str | None = None
