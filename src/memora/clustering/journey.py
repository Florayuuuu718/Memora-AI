from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt

import numpy as np

from memora.models import EventGroup, JourneyGroup, PhotoRecord


@dataclass(frozen=True)
class NamedLocation:
    name: str
    latitude: float
    longitude: float
    radius_km: float = 80.0


@dataclass(frozen=True)
class JourneyConfig:
    home: NamedLocation
    destinations: tuple[NamedLocation, ...] = ()
    max_gap_days: float = 5.0
    stop_radius_km: float = 150.0
    attachment_threshold: float = 0.84
    attachment_margin: float = 0.08

    def to_dict(self) -> dict:
        return asdict(self)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(value))


def _event_time(event: EventGroup) -> datetime | None:
    if not event.start:
        return None
    try:
        return datetime.fromisoformat(event.start.replace("Z", "+00:00"))
    except ValueError:
        return None


def _event_state(event: EventGroup, home: NamedLocation) -> str:
    if event.centroid_latitude is None or event.centroid_longitude is None:
        return "unknown"
    distance = _distance_km(
        event.centroid_latitude,
        event.centroid_longitude,
        home.latitude,
        home.longitude,
    )
    return "home" if distance <= home.radius_km else "away"


def _resolve_destination(event: EventGroup, destinations: tuple[NamedLocation, ...]) -> str | None:
    if event.centroid_latitude is None or event.centroid_longitude is None:
        return None
    candidates = [
        (
            _distance_km(
                event.centroid_latitude,
                event.centroid_longitude,
                destination.latitude,
                destination.longitude,
            ),
            destination,
        )
        for destination in destinations
    ]
    if not candidates:
        return None
    distance, destination = min(candidates, key=lambda item: item[0])
    return destination.name if distance <= destination.radius_km else None


def resolve_location_name(event: EventGroup, locations: tuple[NamedLocation, ...]) -> str | None:
    """Resolve an event to the nearest configured city or named place."""
    return _resolve_destination(event, locations)


def _centroid(event: EventGroup, records: dict[str, PhotoRecord]) -> np.ndarray | None:
    vectors = [np.asarray(records[photo_id].embedding, dtype=np.float32) for photo_id in event.photo_ids if photo_id in records and records[photo_id].embedding]
    if not vectors:
        return None
    value = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 1e-12 else None


def _attachment_score(
    candidate: EventGroup,
    members: list[EventGroup],
    records: dict[str, PhotoRecord],
) -> float:
    candidate_vector = _centroid(candidate, records)
    member_vectors = [_centroid(event, records) for event in members]
    usable = [value for value in member_vectors if value is not None]
    visual = max((float(np.dot(candidate_vector, value)) for value in usable), default=0.0) if candidate_vector is not None else 0.0
    member_people = {person for event in members for person in event.person_ids}
    candidate_people = set(candidate.person_ids)
    people = len(candidate_people & member_people) / max(1, len(candidate_people)) if candidate_people else 0.0
    return 0.65 * visual + 0.35 * people


def _build_stops(events: list[EventGroup], config: JourneyConfig) -> list[dict]:
    """Split one continuous trip into geographic city/region stops."""
    ordered = sorted(
        events,
        key=lambda event: (_event_time(event) is None, _event_time(event) or datetime.max),
    )
    working: list[dict] = []
    for event in ordered:
        latitude = event.centroid_latitude
        longitude = event.centroid_longitude
        destination = _resolve_destination(event, config.destinations)
        current = working[-1] if working else None
        start_new = current is None
        if current is not None and latitude is not None and longitude is not None:
            current_latitudes = current["_latitudes"]
            current_longitudes = current["_longitudes"]
            if current_latitudes and current_longitudes:
                distance = _distance_km(
                    latitude,
                    longitude,
                    sum(current_latitudes) / len(current_latitudes),
                    sum(current_longitudes) / len(current_longitudes),
                )
                same_named_destination = bool(
                    destination
                    and current.get("name") == destination
                )
                start_new = not same_named_destination and distance > config.stop_radius_km

        if start_new:
            current = {
                "id": len(working),
                "name": destination,
                "event_ids": [],
                "photo_ids": [],
                "start": None,
                "end": None,
                "centroid_latitude": None,
                "centroid_longitude": None,
                "_latitudes": [],
                "_longitudes": [],
            }
            working.append(current)

        current["event_ids"].append(event.id)
        current["photo_ids"].extend(event.photo_ids)
        if destination and not current.get("name"):
            current["name"] = destination
        if latitude is not None and longitude is not None:
            current["_latitudes"].append(latitude)
            current["_longitudes"].append(longitude)
        timestamp = _event_time(event)
        if timestamp:
            value = timestamp.isoformat()
            current["start"] = min(current["start"], value) if current["start"] else value
            current["end"] = max(current["end"], value) if current["end"] else value

    output = []
    for stop in working:
        latitudes = stop.pop("_latitudes")
        longitudes = stop.pop("_longitudes")
        if latitudes and longitudes:
            stop["centroid_latitude"] = sum(latitudes) / len(latitudes)
            stop["centroid_longitude"] = sum(longitudes) / len(longitudes)
        if not stop.get("name"):
            stop["name"] = f"Stop {stop['id'] + 1}"
        stop["photo_ids"] = sorted(set(stop["photo_ids"]))
        output.append(stop)
    return output


def _build_journey(
    journey_id: int,
    events: list[EventGroup],
    config: JourneyConfig,
) -> JourneyGroup:
    times = [value for event in events if (value := _event_time(event)) is not None]
    destinations = []
    for event in events:
        name = _resolve_destination(event, config.destinations)
        if name and name not in destinations:
            destinations.append(name)
    event_ids = [event.id for event in events if len(event.photo_ids) > 1]
    loose = [event.photo_ids[0] for event in events if len(event.photo_ids) == 1]
    people = sorted({person for event in events for person in event.person_ids})
    anchored = sum(_event_state(event, config.home) == "away" for event in events)
    return JourneyGroup(
        id=journey_id,
        event_ids=event_ids,
        loose_photo_ids=loose,
        start=min(times).isoformat() if times else None,
        end=max(times).isoformat() if times else None,
        home_name=config.home.name,
        destination_names=destinations,
        person_ids=people,
        confidence=anchored / max(1, len(events)),
        stops=_build_stops(events, config),
    )


def discover_journeys(
    records: list[PhotoRecord],
    events: list[EventGroup],
    config: JourneyConfig,
) -> list[JourneyGroup]:
    """Discover continuous away-from-home journeys from event anchors.

    A trusted event inside the home radius ends the current journey. Unknown
    events inside an active travel window are kept as candidates; timeless
    events are attached later only when visual/person evidence is unambiguous.
    """
    timed = sorted((event for event in events if _event_time(event)), key=lambda event: _event_time(event) or datetime.min)
    groups: list[list[EventGroup]] = []
    current: list[EventGroup] = []
    last_time: datetime | None = None
    for event in timed:
        state = _event_state(event, config.home)
        timestamp = _event_time(event)
        gap_days = abs((timestamp - last_time).total_seconds()) / 86400.0 if timestamp and last_time else 0.0
        if state == "home":
            if current:
                groups.append(current)
                current = []
            last_time = timestamp
            continue
        if state == "away":
            if current and gap_days > config.max_gap_days:
                groups.append(current)
                current = []
            current.append(event)
            last_time = timestamp
        elif current and gap_days <= config.max_gap_days:
            current.append(event)
            last_time = timestamp
    if current:
        groups.append(current)

    assigned = {event.id for group in groups for event in group}
    timeless = [event for event in events if event.id not in assigned and _event_time(event) is None]
    records_by_id = {record.id: record for record in records}
    for candidate in timeless:
        scored = sorted(
            ((_attachment_score(candidate, group, records_by_id), index) for index, group in enumerate(groups)),
            reverse=True,
        )
        if not scored:
            continue
        best_score, best_index = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score >= config.attachment_threshold and best_score - second_score >= config.attachment_margin:
            groups[best_index].append(candidate)

    return [_build_journey(index, group, config) for index, group in enumerate(groups)]
