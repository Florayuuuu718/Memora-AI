from math import asin, cos, radians, sin, sqrt
from typing import Literal

import numpy as np

from memora.models import EventGroup, PhotoRecord


EventStrategy = Literal["time_only", "time_clip", "time_clip_gps", "strict_event", "strict_event_people"]
TRUSTED_TIME_SOURCES = frozenset({"exif"})


def _geo_distance_km(a: PhotoRecord, b: PhotoRecord) -> float | None:
    if None in (a.latitude, a.longitude, b.latitude, b.longitude):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (a.latitude, a.longitude, b.latitude, b.longitude))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(h))


def _time_distance_hours(a: PhotoRecord, b: PhotoRecord) -> float:
    if not a.timestamp or not b.timestamp:
        return float("inf")
    return abs(b.timestamp.timestamp() - a.timestamp.timestamp()) / 3600.0


def _visual_distance(a: PhotoRecord, b: PhotoRecord) -> float:
    if not a.embedding or not b.embedding:
        return 0.0
    va, vb = np.asarray(a.embedding), np.asarray(b.embedding)
    return float(1.0 - np.dot(va, vb) / max(np.linalg.norm(va) * np.linalg.norm(vb), 1e-12))


def _has_trusted_time(record: PhotoRecord) -> bool:
    return record.captured_at_source in TRUSTED_TIME_SOURCES and record.timestamp is not None


def _same_upload_batch(a: PhotoRecord, b: PhotoRecord, window_seconds: float) -> bool:
    if a.captured_at_source != "filesystem" or b.captured_at_source != "filesystem":
        return False
    if not a.timestamp or not b.timestamp:
        return False
    return abs(a.timestamp.timestamp() - b.timestamp.timestamp()) <= window_seconds


def _strict_pair_matches(
    left: PhotoRecord,
    right: PhotoRecord,
    *,
    time_weight: float,
    visual_weight: float,
    geo_weight: float,
    max_gap_hours: float,
    threshold: float,
    strict_in_batch_similarity: float,
    strict_high_similarity: float,
    upload_batch_window_seconds: float,
) -> bool:
    left_trusted, right_trusted = _has_trusted_time(left), _has_trusted_time(right)
    if left_trusted and right_trusted:
        distance, time_component = _event_distance(
            left,
            right,
            "time_clip_gps",
            time_weight=time_weight,
            visual_weight=visual_weight,
            geo_weight=geo_weight,
            max_gap_hours=max_gap_hours,
        )
        return time_component < 1.0 and distance <= threshold

    if not left.embedding or not right.embedding:
        return False
    similarity = 1.0 - _visual_distance(left, right)
    in_same_batch = _same_upload_batch(left, right, upload_batch_window_seconds)
    required_similarity = strict_in_batch_similarity if in_same_batch else strict_high_similarity
    return similarity >= required_similarity


def _strict_cluster_events(
    records: list[PhotoRecord],
    *,
    time_weight: float,
    visual_weight: float,
    geo_weight: float,
    max_gap_hours: float,
    threshold: float,
    strict_in_batch_similarity: float,
    strict_high_similarity: float,
    upload_batch_window_seconds: float,
) -> list[EventGroup]:
    """Cluster with trusted metadata and conservative CLIP-only fallbacks.

    Filesystem timestamps identify an upload batch only. They never provide
    positive capture-time evidence. A photo without EXIF can join an EXIF
    event only through the high CLIP threshold; photos in the same upload
    batch use the slightly lower, still conservative in-batch threshold.
    """
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            if _strict_pair_matches(
                records[left],
                records[right],
                time_weight=time_weight,
                visual_weight=visual_weight,
                geo_weight=geo_weight,
                max_gap_hours=max_gap_hours,
                threshold=threshold,
                strict_in_batch_similarity=strict_in_batch_similarity,
                strict_high_similarity=strict_high_similarity,
                upload_batch_window_seconds=upload_batch_window_seconds,
            ):
                union(left, right)

    groups: dict[int, list[PhotoRecord]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)
    output = []
    for event_id, group in enumerate(groups.values()):
        trusted_times = [item.timestamp for item in group if _has_trusted_time(item)]
        latitudes = [item.latitude for item in group if item.latitude is not None]
        longitudes = [item.longitude for item in group if item.longitude is not None]
        has_untrusted = any(not _has_trusted_time(item) for item in group)
        output.append(EventGroup(
            id=event_id,
            photo_ids=[item.id for item in group],
            start=min(trusted_times).isoformat() if trusted_times else None,
            end=max(trusted_times).isoformat() if trusted_times else None,
            centroid_latitude=sum(latitudes) / len(latitudes) if latitudes else None,
            centroid_longitude=sum(longitudes) / len(longitudes) if longitudes else None,
            evidence="strict_clip_fallback" if has_untrusted else "strict_exif_time_clip_gps",
        ))
    return output


def _event_distance(
    previous: PhotoRecord,
    current: PhotoRecord,
    strategy: EventStrategy,
    *,
    time_weight: float,
    visual_weight: float,
    geo_weight: float,
    max_gap_hours: float,
) -> tuple[float, float]:
    time_component = min(_time_distance_hours(previous, current) / max(max_gap_hours, 1e-6), 1.0)
    if strategy == "time_only":
        return time_component, time_component

    visual_component = _visual_distance(previous, current)
    distance = time_weight * time_component + visual_weight * visual_component
    if strategy == "time_clip_gps":
        geo = _geo_distance_km(previous, current)
        distance += geo_weight * min((geo or 0.0) / 50.0, 1.0)
    return distance, time_component


def cluster_events(
    records: list[PhotoRecord],
    *,
    strategy: EventStrategy = "time_clip_gps",
    time_weight: float = 0.55,
    visual_weight: float = 0.30,
    geo_weight: float = 0.15,
    max_gap_hours: float = 8.0,
    threshold: float = 0.48,
    strict_in_batch_similarity: float = 0.86,
    strict_high_similarity: float = 0.92,
    upload_batch_window_seconds: float = 15 * 60,
    people_index=None,
) -> list[EventGroup]:
    """Discover events with one of the three ablation strategies.

    ``time_only`` is the temporal baseline, ``time_clip`` adds CLIP
    similarity, and ``time_clip_gps`` adds geographic distance. The
    sequential grouping is deterministic and works without optional ML
    dependencies, making it suitable for the exact experiment baseline.
    """
    if strategy not in {"time_only", "time_clip", "time_clip_gps", "strict_event", "strict_event_people"}:
        raise ValueError(f"unknown event strategy: {strategy}")
    if strategy == "strict_event_people":
        from memora.clustering.event_people import cluster_events_with_people

        return cluster_events_with_people(
            records,
            people_index,
            time_weight=time_weight,
            visual_weight=visual_weight,
            geo_weight=geo_weight,
            max_gap_hours=max_gap_hours,
            threshold=threshold,
            strict_in_batch_similarity=strict_in_batch_similarity,
            strict_high_similarity=strict_high_similarity,
            upload_batch_window_seconds=upload_batch_window_seconds,
        )
    if strategy == "strict_event":
        return _strict_cluster_events(
            records,
            time_weight=time_weight,
            visual_weight=visual_weight,
            geo_weight=geo_weight,
            max_gap_hours=max_gap_hours,
            threshold=threshold,
            strict_in_batch_similarity=strict_in_batch_similarity,
            strict_high_similarity=strict_high_similarity,
            upload_batch_window_seconds=upload_batch_window_seconds,
        )
    ordered = sorted(records, key=lambda item: item.timestamp.timestamp() if item.timestamp else float("inf"))
    if not ordered:
        return []
    groups: list[list[PhotoRecord]] = [[ordered[0]]]
    for current in ordered[1:]:
        previous = groups[-1][-1]
        distance, time_component = _event_distance(
            previous,
            current,
            strategy,
            time_weight=time_weight,
            visual_weight=visual_weight,
            geo_weight=geo_weight,
            max_gap_hours=max_gap_hours,
        )
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
