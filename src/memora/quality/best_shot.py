from collections.abc import Iterable, Mapping

import numpy as np
from PIL import Image

from memora.models import PhotoRecord
from memora.quality.blur import laplacian_variance
from memora.quality.exposure import exposure_score


def face_quality_score(
    faces: Iterable[Mapping[str, object]] | None = None,
    *,
    image_area: float | None = None,
) -> float:
    """Return a normalized face-quality score from optional detector output.

    InsightFace results can be passed as mappings containing ``det_score``,
    ``quality`` and optionally ``bbox``. Photos without detected faces get a
    neutral score so landscapes are not penalized in best-shot ranking.
    """
    values = list(faces or [])
    if not values:
        return 0.5
    scores = []
    for face in values:
        score = float(face.get("quality", face.get("det_score", 0.0)) or 0.0)
        if image_area and face.get("bbox"):
            box = [float(value) for value in face["bbox"]]  # type: ignore[index]
            if len(box) >= 4:
                area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
                score *= min(1.0, (area / image_area) * 8.0)
        scores.append(max(0.0, min(1.0, score)))
    return float(sum(scores) / len(scores))


def composition_score(path: str) -> float:
    """Estimate composition quality without requiring a vision model.

    It uses the center of edge energy as a lightweight subject proxy and
    rewards placement near rule-of-thirds intersections, while keeping the
    result in [0, 1]. It is intentionally a ranking signal, not a semantic
    composition classifier.
    """
    with Image.open(path).convert("L") as image:
        gray = np.asarray(image.resize((128, 128)), dtype=np.float32) / 255.0
    dx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    dy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    energy = dx + dy + 1e-6
    yy, xx = np.indices(energy.shape, dtype=np.float32)
    center_x = float((xx * energy).sum() / energy.sum()) / 127.0
    center_y = float((yy * energy).sum() / energy.sum()) / 127.0
    targets = ((1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3))
    distance = min(np.hypot(center_x - tx, center_y - ty) for tx, ty in targets)
    placement = max(0.0, 1.0 - float(distance) / 0.48)
    edge_coverage = min(1.0, float((energy > np.quantile(energy, 0.75)).mean()) * 4.0)
    return float(0.75 * placement + 0.25 * edge_coverage)


def score_photo(
    path: str,
    *,
    faces: Iterable[Mapping[str, object]] | None = None,
) -> dict[str, float]:
    sharpness = laplacian_variance(path)
    normalized_sharpness = sharpness / (sharpness + 100.0)
    exposure = exposure_score(path)
    with Image.open(path) as image:
        image_area = float(image.width * image.height)
    face_score = face_quality_score(faces, image_area=image_area)
    composition = composition_score(path)
    score = (
        0.35 * normalized_sharpness
        + 0.20 * exposure
        + 0.20 * face_score
        + 0.25 * composition
    )
    return {
        "sharpness": sharpness,
        "sharpness_score": normalized_sharpness,
        "exposure_score": exposure,
        "face_quality_score": face_score,
        "composition_score": composition,
        "score": score,
    }


def best_shot(records: list[PhotoRecord]) -> PhotoRecord | None:
    return max(records, key=lambda record: record.quality.get("score", 0.0), default=None)


def rank_best_shots(records: Iterable[PhotoRecord]) -> list[PhotoRecord]:
    """Return all candidates ranked from best to worst."""
    return sorted(records, key=lambda record: record.quality.get("score", 0.0), reverse=True)
