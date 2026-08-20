from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any

import numpy as np

from memora.clustering.journey import JourneyConfig, NamedLocation
from memora.models import PhotoRecord


@dataclass
class LocationCluster:
    id: int
    name: str
    latitude: float
    longitude: float
    photo_ids: list[str]
    photo_count: int
    active_days: int
    active_months: int
    start: str | None
    end: str | None
    radius_km: float
    name_source: str
    geocoder: dict[str, str] | None = None
    is_home: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distance_km(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (*left, *right))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(value))


def _timestamp(record: PhotoRecord) -> datetime | None:
    return record.timestamp


def _reverse_geocode(coordinates: list[tuple[float, float]]) -> list[dict[str, str] | None]:
    try:
        import reverse_geocoder  # type: ignore
    except ImportError:
        return [None] * len(coordinates)
    try:
        with redirect_stdout(io.StringIO()):
            values = reverse_geocoder.search(coordinates, mode=1)
        return [dict(value) for value in values]
    except (OSError, TypeError, ValueError):
        return [None] * len(coordinates)


def _location_name(
    latitude: float,
    longitude: float,
    geocoder: dict[str, str] | None,
) -> tuple[str, str]:
    if geocoder and geocoder.get("name"):
        name = geocoder["name"]
        admin = geocoder.get("admin1")
        country = geocoder.get("cc")
        suffix = admin or country
        return (f"{name}, {suffix}" if suffix and suffix != name else name), "offline_reverse_geocoder"
    return f"GPS {latitude:.3f},{longitude:.3f}", "gps_centroid"


def cluster_gps_locations(
    records: list[PhotoRecord],
    *,
    cluster_radius_km: float = 40.0,
    geocode: bool = True,
) -> list[LocationCluster]:
    """Cluster photo coordinates into recurring city/region locations."""
    usable = [
        record
        for record in records
        if record.latitude is not None and record.longitude is not None
    ]
    if not usable:
        return []
    parent = list(range(len(usable)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(usable)):
        left_coordinate = (float(usable[left].latitude), float(usable[left].longitude))
        for right in range(left):
            right_coordinate = (float(usable[right].latitude), float(usable[right].longitude))
            if _distance_km(left_coordinate, right_coordinate) <= cluster_radius_km:
                union(left, right)

    grouped: dict[int, list[PhotoRecord]] = {}
    for index, record in enumerate(usable):
        grouped.setdefault(find(index), []).append(record)
    values = sorted(grouped.values(), key=len, reverse=True)
    coordinates = [
        (
            float(np.mean([record.latitude for record in group])),
            float(np.mean([record.longitude for record in group])),
        )
        for group in values
    ]
    geocoder_values = _reverse_geocode(coordinates) if geocode else [None] * len(coordinates)
    output = []
    for index, (group, coordinate, geocoder) in enumerate(
        zip(values, coordinates, geocoder_values)
    ):
        times = sorted(value for record in group if (value := _timestamp(record)) is not None)
        days = {value.date().isoformat() for value in times}
        months = {value.strftime("%Y-%m") for value in times}
        distances = [
            _distance_km(
                coordinate,
                (float(record.latitude), float(record.longitude)),
            )
            for record in group
        ]
        radius = max(20.0, min(150.0, float(np.percentile(distances, 90)) + 15.0))
        name, name_source = _location_name(coordinate[0], coordinate[1], geocoder)
        output.append(
            LocationCluster(
                id=index,
                name=name,
                latitude=coordinate[0],
                longitude=coordinate[1],
                photo_ids=sorted(record.id for record in group),
                photo_count=len(group),
                active_days=len(days),
                active_months=len(months),
                start=times[0].isoformat() if times else None,
                end=times[-1].isoformat() if times else None,
                radius_km=radius,
                name_source=name_source,
                geocoder=geocoder,
            )
        )
    return output


def infer_journey_config(
    records: list[PhotoRecord],
    *,
    cluster_radius_km: float = 40.0,
    minimum_destination_photos: int = 2,
    max_gap_days: float = 5.0,
    stop_radius_km: float = 150.0,
    geocode: bool = True,
) -> tuple[JourneyConfig, list[LocationCluster]]:
    """Infer the recurring home region and travel locations from photo GPS."""
    clusters = cluster_gps_locations(
        records,
        cluster_radius_km=cluster_radius_km,
        geocode=geocode,
    )
    if not clusters:
        raise ValueError("Cannot infer journey locations because no GPS records are available")
    home = max(
        clusters,
        key=lambda cluster: (cluster.active_months, cluster.active_days, cluster.photo_count),
    )
    home.is_home = True
    destinations = [
        cluster
        for cluster in clusters
        if cluster.id != home.id and cluster.photo_count >= minimum_destination_photos
    ]
    config = JourneyConfig(
        home=NamedLocation(
            home.name,
            home.latitude,
            home.longitude,
            max(40.0, min(100.0, home.radius_km + 20.0)),
        ),
        destinations=tuple(
            NamedLocation(
                cluster.name,
                cluster.latitude,
                cluster.longitude,
                cluster.radius_km,
            )
            for cluster in destinations
        ),
        max_gap_days=max_gap_days,
        stop_radius_km=stop_radius_km,
    )
    return config, clusters
