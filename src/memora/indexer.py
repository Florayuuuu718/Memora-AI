import hashlib
from pathlib import Path

from memora.duplicate.phash import phash
from memora.encoders.clip_encoder import VisionEncoder
from memora.metadata.exif import read_metadata
from memora.models import PhotoRecord
from memora.quality.best_shot import score_photo


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}


def iter_images(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def photo_id(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def index_directory(directory: str | Path, encoder: VisionEncoder, existing: dict[str, PhotoRecord] | None = None) -> list[PhotoRecord]:
    existing = existing or {}
    records = []
    for path in iter_images(directory):
        metadata = read_metadata(path)
        identifier = photo_id(path)
        record = existing.get(identifier, PhotoRecord(id=identifier, path=str(path.resolve())))
        record.path = str(path.resolve())
        record.width = int(metadata.get("width", 0))
        record.height = int(metadata.get("height", 0))
        record.captured_at = metadata.get("captured_at")
        record.latitude = metadata.get("latitude")
        record.longitude = metadata.get("longitude")
        record.camera = metadata.get("camera")
        record.embedding = encoder.encode_image(path).tolist()
        record.phash = phash(path)
        record.quality = score_photo(str(path))
        records.append(record)
    return records
