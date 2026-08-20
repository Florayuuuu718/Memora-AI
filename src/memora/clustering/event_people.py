from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import numpy as np

from memora.clustering.event_cluster import _geo_distance_km, cluster_events
from memora.clustering.people import PeopleIndex
from memora.models import EventGroup, PhotoRecord


def _normalized_mean(vectors: list[list[float]]) -> np.ndarray | None:
    usable = [np.asarray(vector, dtype=np.float32) for vector in vectors if vector]
    if not usable:
        return None
    value = np.mean(usable, axis=0)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else None


def _cosine(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    if left is None or right is None:
        return None
    return float(np.dot(left, right))


def build_people_evidence(
    people: PeopleIndex | None,
) -> tuple[dict[str, set[int]], dict[tuple[str, int], np.ndarray]]:
    photo_people: dict[str, set[int]] = {}
    appearances: dict[tuple[str, int], np.ndarray] = {}
    if people is None:
        return photo_people, appearances
    faces_by_id = {face.id: face for face in people.faces}
    for group in people.groups:
        for photo_id in group.photo_ids:
            photo_people.setdefault(photo_id, set()).add(group.id)
        by_photo: dict[str, list[list[float]]] = {}
        for face_id in group.face_ids:
            face = faces_by_id.get(face_id)
            if face and face.appearance_embedding:
                by_photo.setdefault(face.photo_id, []).append(face.appearance_embedding)
        for photo_id, vectors in by_photo.items():
            descriptor = _normalized_mean(vectors)
            if descriptor is not None:
                appearances[(photo_id, group.id)] = descriptor
    return photo_people, appearances


def _group_people(group: list[PhotoRecord], photo_people: dict[str, set[int]]) -> set[int]:
    output: set[int] = set()
    for record in group:
        output.update(photo_people.get(record.id, set()))
    return output


def _group_centroid(group: list[PhotoRecord]) -> np.ndarray | None:
    return _normalized_mean([record.embedding for record in group])


def _appearance_similarity(
    left: list[PhotoRecord],
    right: list[PhotoRecord],
    shared_people: set[int],
    appearances: dict[tuple[str, int], np.ndarray],
) -> float | None:
    values = []
    for person_id in shared_people:
        left_vectors = [appearances[(record.id, person_id)] for record in left if (record.id, person_id) in appearances]
        right_vectors = [appearances[(record.id, person_id)] for record in right if (record.id, person_id) in appearances]
        left_value = _normalized_mean([vector.tolist() for vector in left_vectors])
        right_value = _normalized_mean([vector.tolist() for vector in right_vectors])
        similarity = _cosine(left_value, right_value)
        if similarity is not None:
            values.append(similarity)
    return float(np.mean(values)) if values else None


def _trusted_times(group: list[PhotoRecord]) -> list[datetime]:
    return [
        record.timestamp
        for record in group
        if record.captured_at_source == "exif" and record.timestamp is not None
    ]


def _has_hard_conflict(
    left: list[PhotoRecord],
    right: list[PhotoRecord],
    *,
    max_gap_hours: float,
    max_geo_km: float,
) -> bool:
    left_times, right_times = _trusted_times(left), _trusted_times(right)
    if left_times and right_times:
        gap = min(abs((a - b).total_seconds()) for a in left_times for b in right_times) / 3600.0
        if gap > max_gap_hours:
            return True
    located_left = [record for record in left if record.latitude is not None and record.longitude is not None]
    located_right = [record for record in right if record.latitude is not None and record.longitude is not None]
    if located_left and located_right:
        distance = min(
            value
            for a in located_left
            for b in located_right
            if (value := _geo_distance_km(a, b)) is not None
        )
        if distance > max_geo_km:
            return True
    return False


def _merge_score(
    left: list[PhotoRecord],
    right: list[PhotoRecord],
    photo_people: dict[str, set[int]],
    appearances: dict[tuple[str, int], np.ndarray],
    *,
    min_visual_similarity: float,
) -> tuple[float, int] | None:
    left_people, right_people = _group_people(left, photo_people), _group_people(right, photo_people)
    shared = left_people & right_people
    if not shared:
        return None
    visual = _cosine(_group_centroid(left), _group_centroid(right))
    if visual is None or visual < min_visual_similarity:
        return None
    containment = len(shared) / max(1, min(len(left_people), len(right_people)))
    appearance = _appearance_similarity(left, right, shared, appearances)
    if appearance is None:
        score = 0.55 * visual + 0.45 * containment
    else:
        score = 0.45 * visual + 0.30 * containment + 0.25 * appearance
    return score, len(shared)


def _to_event_group(
    event_id: int,
    group: list[PhotoRecord],
    photo_people: dict[str, set[int]],
) -> EventGroup:
    trusted = _trusted_times(group)
    latitudes = [item.latitude for item in group if item.latitude is not None]
    longitudes = [item.longitude for item in group if item.longitude is not None]
    duration = (max(trusted) - min(trusted)).total_seconds() / 60.0 if trusted else None
    return EventGroup(
        id=event_id,
        photo_ids=[record.id for record in group],
        start=min(trusted).isoformat() if trusted else None,
        end=max(trusted).isoformat() if trusted else None,
        centroid_latitude=sum(latitudes) / len(latitudes) if latitudes else None,
        centroid_longitude=sum(longitudes) / len(longitudes) if longitudes else None,
        evidence="strict_event_people",
        person_ids=sorted(_group_people(group, photo_people)),
        duration_minutes=duration,
    )


def cluster_events_with_people(
    records: list[PhotoRecord],
    people: PeopleIndex | None,
    *,
    max_gap_hours: float = 8.0,
    max_geo_km: float = 80.0,
    min_visual_similarity: float = 0.72,
    merge_threshold: float = 0.86,
    multi_person_threshold: float = 0.82,
    **strict_kwargs: float,
) -> list[EventGroup]:
    """Refine strict events using identity, co-occurrence and appearance evidence."""
    base = cluster_events(
        records,
        strategy="strict_event",
        max_gap_hours=max_gap_hours,
        **strict_kwargs,
    )
    by_id = {record.id: record for record in records}
    groups = [[by_id[photo_id] for photo_id in event.photo_ids if photo_id in by_id] for event in base]
    photo_people, appearances = build_people_evidence(people)

    while True:
        best: tuple[float, int, int] | None = None
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                if _has_hard_conflict(
                    groups[left], groups[right], max_gap_hours=max_gap_hours, max_geo_km=max_geo_km
                ):
                    continue
                result = _merge_score(
                    groups[left],
                    groups[right],
                    photo_people,
                    appearances,
                    min_visual_similarity=min_visual_similarity,
                )
                if result is None:
                    continue
                score, shared_count = result
                required = multi_person_threshold if shared_count >= 2 else merge_threshold
                if score >= required and (best is None or score > best[0]):
                    best = (score, left, right)
        if best is None:
            break
        _, left, right = best
        groups[left].extend(groups[right])
        groups.pop(right)

    groups.sort(key=lambda group: min((record.timestamp.timestamp() for record in group if record.timestamp), default=float("inf")))
    return [_to_event_group(index, group, photo_people) for index, group in enumerate(groups)]


def enrich_events_with_people(events: list[EventGroup], people: PeopleIndex | None) -> list[EventGroup]:
    """Attach person IDs without changing event membership."""
    photo_people, _ = build_people_evidence(people)
    return [
        replace(
            event,
            person_ids=sorted({person for photo_id in event.photo_ids for person in photo_people.get(photo_id, set())}),
        )
        for event in events
    ]
