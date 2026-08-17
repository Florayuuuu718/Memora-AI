from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import numpy as np

from memora.models import EventGroup, PhotoRecord


def _geo_distance_km(a: PhotoRecord, b: PhotoRecord) -> float | None:
    if None in (a.latitude, a.longitude, b.latitude, b.longitude):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(h))


def _time_distance_hours(a: PhotoRecord, b: PhotoRecord) -> float:
    if not a.timestamp or not b.timestamp:
        return 0.0
    return abs((b.timestamp - a.timestamp).total_seconds()) / 3600.0


def _visual_distance(a: PhotoRecord, b: PhotoRecord) -> float:
    if not a.embedding or not b.embedding:
        return 0.0
    va, vb = np.asarray(a.embedding), np.asarray(b.embedding)
    return float(1.0 - np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))


def cluster_events(records: list[PhotoRecord], *, time_weight: float = 0.55, visual_weight: float = 0.30,
                   geo_weight: float = 0.15, max_gap_hours: float = 8.0, threshold: float = 0.48) -> list[EventGroup]:
    ordered = sorted(records, key=lambda item: item.timestamp or datetime.min)
    if not ordered:
        return []
    groups: list[list[PhotoRecord]] = [[ordered[0]]]
    for current in ordered[1:]:
        previous = groups[-1][-1]
        time_component = min(_time_distance_hours(previous, current) / max(max_gap_hours, 1e-6), 1.0)
        visual_component = _visual_distance(previous, current)
        geo = _geo_distance_km(previous, current)
        geo_component = min((geo or 0.0) / 50.0, 1.0)
        distance = time_weight * time_component + visual_weight * visual_component + geo_weight * geo_component
        if time_component > 1.0 or distance > threshold:
            groups.append([current])
        else:
            groups[-1].append(current)
    output = []
    for index, group in enumerate(groups):
        latitudes = [item.latitude for item in group if item.latitude is not None]
        longitudes = [item.longitude for item in group if item.longitude is not None]
        output.append(EventGroup(
            id=index,
            photo_ids=[item.id for item in group],
            start=group[0].captured_at,
            end=group[-1].captured_at,
            centroid_latitude=sum(latitudes) / len(latitudes) if latitudes else None,
            centroid_longitude=sum(longitudes) / len(longitudes) if longitudes else None,
        ))
    return output


def cluster_embeddings(embeddings: np.ndarray, eps: float = 0.35, min_samples: int = 2) -> np.ndarray:
    """DBSCAN wrapper with a small NumPy fallback when scikit-learn is absent."""
    try:
        from sklearn.cluster import DBSCAN  # type: ignore
        return DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit_predict(embeddings)
    except ImportError:
        labels = np.full(len(embeddings), -1, dtype=int)
        next_label = 0
        for index, vector in enumerate(embeddings):
            if labels[index] != -1:
                continue
            distances = 1.0 - embeddings @ vector / np.maximum(np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vector), 1e-12)
            neighbours = np.where(distances <= eps)[0]
            if len(neighbours) < min_samples:
                continue
            labels[neighbours] = next_label
            next_label += 1
        return labels

