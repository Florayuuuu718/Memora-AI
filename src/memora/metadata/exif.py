from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image


_TAGS = {value: key for key, value in ExifTags.TAGS.items()}
_GPS_TAGS = {value: key for key, value in ExifTags.GPSTAGS.items()}


def _ratio(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return float(value.numerator) / float(value.denominator)
        except (AttributeError, ZeroDivisionError):
            return 0.0


def _gps_value(gps: dict[Any, Any], name: str) -> Any:
    return gps.get(_GPS_TAGS.get(name, name))


def _gps_coordinate(value: Any, ref: Any) -> float | None:
    if not value or len(value) != 3:
        return None
    degrees, minutes, seconds = (_ratio(part) for part in value)
    result = degrees + minutes / 60.0 + seconds / 3600.0
    if str(ref).upper() in {"S", "W"}:
        result = -result
    return result


def _timestamp(value: Any) -> str | None:
    if not value:
        return None
    raw = str(value).replace("/", ":")
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt).isoformat()
        except ValueError:
            pass
    return None


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read portable EXIF fields using Pillow, with filesystem mtime fallback."""
    path = Path(path)
    result: dict[str, Any] = {
        "path": str(path.resolve()),
        "captured_at": None,
        "captured_at_source": None,
        "gps_source": None,
    }
    try:
        with Image.open(path) as image:
            result["width"], result["height"] = image.size
            exif = image.getexif()
            tags = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
            result["captured_at"] = _timestamp(tags.get("DateTimeOriginal") or tags.get("DateTime"))
            if result["captured_at"]:
                result["captured_at_source"] = "exif"
            result["camera"] = " ".join(str(tags.get(k, "")).strip() for k in ("Make", "Model")).strip() or None
            gps = exif.get_ifd(_TAGS.get("GPSInfo", 34853)) if exif else {}
            result["latitude"] = _gps_coordinate(_gps_value(gps, "GPSLatitude"), _gps_value(gps, "GPSLatitudeRef"))
            result["longitude"] = _gps_coordinate(_gps_value(gps, "GPSLongitude"), _gps_value(gps, "GPSLongitudeRef"))
            if result["latitude"] is not None and result["longitude"] is not None:
                result["gps_source"] = "exif"
    except (OSError, ValueError, KeyError, AttributeError):
        result["width"] = result["height"] = 0
        result["camera"] = None
    if not result.get("captured_at"):
        result["captured_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()
        result["captured_at_source"] = "filesystem"
    return result
