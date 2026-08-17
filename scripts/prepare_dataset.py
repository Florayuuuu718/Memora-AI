import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image

from memora.metadata.exif import read_metadata

SOURCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff"}
HEIF_EXTENSIONS = {".heic", ".heif"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _register_heif_if_needed(paths: list[Path]) -> None:
    if not any(path.suffix.lower() in HEIF_EXTENSIONS for path in paths):
        return
    try:
        from pillow_heif import register_heif_opener  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "HEIC/HEIF files detected. Install the preprocessing extra first: "
            "pip install -e \".[preprocess]\""
        ) from exc
    register_heif_opener()


def _exif_bytes(image: Image.Image) -> bytes:
    value = image.info.get("exif")
    if isinstance(value, bytes):
        return value
    try:
        return image.getexif().tobytes()
    except (AttributeError, ValueError):
        return b""


def _convert_to_jpeg(source: Path, target: Path, quality: int) -> None:
    temporary = target.with_name(f".{target.name}.tmp")
    try:
        with Image.open(source) as image:
            exif = _exif_bytes(image)
            image.convert("RGB").save(temporary, format="JPEG", quality=quality, exif=exif)
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _metadata_record(source: Path, target: Path, source_root: Path) -> dict[str, Any]:
    source_metadata = read_metadata(source)
    _convert_to_jpeg(source, target, quality=95)
    target_metadata = read_metadata(target)

    has_source_time = source_metadata.get("captured_at_source") == "exif"
    has_source_gps = source_metadata.get("gps_source") == "exif"
    time_preserved = not has_source_time or target_metadata.get("captured_at") == source_metadata.get("captured_at")
    gps_preserved = (
        not has_source_gps
        or (
            target_metadata.get("latitude") is not None
            and target_metadata.get("longitude") is not None
            and abs(target_metadata["latitude"] - source_metadata["latitude"]) < 1e-6
            and abs(target_metadata["longitude"] - source_metadata["longitude"]) < 1e-6
        )
    )
    if not time_preserved or not gps_preserved:
        raise RuntimeError(f"Metadata was not preserved while converting {source}")

    return {
        "source_file": str(source.relative_to(source_root)),
        "output_file": str(target),
        "source_format": source.suffix.lower().lstrip("."),
        "source_sha256": _sha256(source),
        "output_sha256": _sha256(target),
        "captured_at": source_metadata.get("captured_at"),
        "captured_at_source": source_metadata.get("captured_at_source"),
        "latitude": source_metadata.get("latitude"),
        "longitude": source_metadata.get("longitude"),
        "gps_source": source_metadata.get("gps_source"),
        "width": source_metadata.get("width", 0),
        "height": source_metadata.get("height", 0),
        "metadata_preserved": True,
    }


def prepare_dataset(source_dir: str | Path, output_dir: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    source_root = Path(source_dir).resolve()
    output_root = Path(output_dir).resolve()
    manifest = Path(manifest_path)
    if not source_root.is_dir():
        raise ValueError(f"Source directory does not exist: {source_root}")
    if output_root == source_root or output_root.is_relative_to(source_root):
        raise ValueError("Output directory must be separate from and outside the source directory")
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Output directory is not empty: {output_root}")

    sources = sorted(
        (path for path in source_root.rglob("*") if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS),
        key=lambda path: path.relative_to(source_root).as_posix().casefold(),
    )
    _register_heif_if_needed(sources)
    output_root.mkdir(parents=True, exist_ok=True)
    records = []
    for number, source in enumerate(sources, start=1):
        target = output_root / f"{number:06d}.jpg"
        records.append(_metadata_record(source, target, source_root))

    payload = {
        "version": 1,
        "source_directory": str(source_root),
        "output_directory": str(output_root),
        "count": len(records),
        "records": records,
    }
    manifest = Path(manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert source photos to normalized JPEG files")
    parser.add_argument("source_dir")
    parser.add_argument("--output", required=True, help="Separate output directory for normalized JPEGs")
    parser.add_argument("--manifest", default="data/prepared_manifest.json")
    args = parser.parse_args()
    payload = prepare_dataset(args.source_dir, args.output, args.manifest)
    print(json.dumps({"count": payload["count"], "output": args.output, "manifest": args.manifest}, indent=2))


if __name__ == "__main__":
    main()
