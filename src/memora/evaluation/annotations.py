import csv
import json
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence


CONFIDENCE_ORDER = {"low": 0, "medium": 1, "high": 2}


def load_annotations(path: str | Path) -> list[dict[str, Any]]:
    """Load rich CSV/JSON annotations or a legacy photo_id-to-event mapping."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        return [{"photo_id": photo_id, "event_id": event_id, "label_confidence": "high"} for photo_id, event_id in value.items()]
    if isinstance(value, list):
        return [dict(item) for item in value]
    raise ValueError("annotations must be a CSV, JSON list, or JSON photo_id mapping")


def annotation_labels(
    annotations: list[dict[str, Any]],
    field: str,
    *,
    minimum_confidence: str = "low",
    fallback_field: str | None = None,
) -> dict[str, Any]:
    required = CONFIDENCE_ORDER.get(minimum_confidence, 0)
    output = {}
    for item in annotations:
        confidence = str(item.get("label_confidence") or "high").lower()
        value = item.get(field)
        if value in {None, ""} and fallback_field:
            value = item.get(fallback_field)
        photo_id = item.get("photo_id")
        if photo_id and value not in {None, "", "review"} and CONFIDENCE_ORDER.get(confidence, 0) >= required:
            output[str(photo_id)] = value
    return output


def resolve_annotation_labels(
    records: Sequence,
    labels: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve annotation keys written as IDs, filenames, or stems to IDs."""
    by_key: dict[str, str] = {}
    for record in records:
        path = Path(record.path)
        for key in {record.id, path.name, path.stem}:
            by_key[str(key)] = record.id
    return {by_key[str(key)]: value for key, value in labels.items() if str(key) in by_key}
