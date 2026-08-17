from memora.models import PhotoRecord
from memora.quality.blur import laplacian_variance
from memora.quality.exposure import exposure_score


def score_photo(path: str) -> dict[str, float]:
    sharpness = laplacian_variance(path)
    normalized_sharpness = sharpness / (sharpness + 100.0)
    exposure = exposure_score(path)
    score = 0.65 * normalized_sharpness + 0.35 * exposure
    return {"sharpness": sharpness, "sharpness_score": normalized_sharpness, "exposure_score": exposure, "score": score}


def best_shot(records: list[PhotoRecord]) -> PhotoRecord | None:
    return max(records, key=lambda record: record.quality.get("score", 0.0), default=None)
