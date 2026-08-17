from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from memora.models import PhotoRecord


@dataclass(frozen=True)
class GeoBounds:
    min_latitude: float
    min_longitude: float
    max_latitude: float
    max_longitude: float

    def contains(self, latitude: float, longitude: float) -> bool:
        return (
            self.min_latitude <= latitude <= self.max_latitude
            and self.min_longitude <= longitude <= self.max_longitude
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "min_latitude": self.min_latitude,
            "min_longitude": self.min_longitude,
            "max_latitude": self.max_latitude,
            "max_longitude": self.max_longitude,
        }


@dataclass(frozen=True)
class MetadataFilter:
    captured_from: datetime | None = None
    captured_to: datetime | None = None
    bounds: GeoBounds | None = None
    require_reliable_time: bool = True
    require_reliable_gps: bool = True

    @property
    def requires_time(self) -> bool:
        return self.captured_from is not None or self.captured_to is not None

    @property
    def requires_gps(self) -> bool:
        return self.bounds is not None

    @property
    def is_empty(self) -> bool:
        return not self.requires_time and not self.requires_gps

    def matches(self, record: PhotoRecord) -> bool:
        if self.requires_time:
            timestamp = record.timestamp
            if timestamp is None:
                return False
            if self.require_reliable_time and record.captured_at_source != "exif":
                return False
            timestamp = timestamp.replace(tzinfo=None)
            if self.captured_from is not None and timestamp < self.captured_from:
                return False
            if self.captured_to is not None and timestamp >= self.captured_to:
                return False
        if self.requires_gps:
            if record.latitude is None or record.longitude is None:
                return False
            if self.require_reliable_gps and record.gps_source != "exif":
                return False
            if not self.bounds or not self.bounds.contains(record.latitude, record.longitude):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "captured_from": self.captured_from.isoformat() if self.captured_from else None,
            "captured_to": self.captured_to.isoformat() if self.captured_to else None,
            "bounds": self.bounds.to_dict() if self.bounds else None,
            "require_reliable_time": self.require_reliable_time,
            "require_reliable_gps": self.require_reliable_gps,
        }


@dataclass(frozen=True)
class SearchPlan:
    original_query: str
    semantic_query: str
    metadata_filter: MetadataFilter
    time_expression: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "semantic_query": self.semantic_query,
            "time_expression": self.time_expression,
            "metadata_filter": self.metadata_filter.to_dict(),
        }


def _reference_date(value: date | datetime | None) -> date:
    if value is None:
        return date.today()
    return value.date() if isinstance(value, datetime) else value


def _year_window(year: int) -> tuple[datetime, datetime]:
    return datetime(year, 1, 1), datetime(year + 1, 1, 1)


def _parse_time(query: str, reference: date) -> tuple[datetime | None, datetime | None, str | None]:
    year_match = re.search(r"(20\d{2})\s*\u5e74", query)
    if year_match:
        start, end = _year_window(int(year_match.group(1)))
        return start, end, year_match.group(0)
    if re.search(r"\u53bb\u5e74|\u4e0a\u4e00\u5e74|last\s+year", query, re.IGNORECASE):
        start, end = _year_window(reference.year - 1)
        return start, end, "\u53bb\u5e74"
    if re.search(r"\u4eca\u5e74|\u672c\u5e74|this\s+year", query, re.IGNORECASE):
        start, end = _year_window(reference.year)
        return start, end, "\u4eca\u5e74"
    if re.search(r"\u4e0a\u4e2a\u6708|\u4e0a\u6708|last\s+month", query, re.IGNORECASE):
        first_this_month = date(reference.year, reference.month, 1)
        last_month_end = datetime.combine(first_this_month, datetime.min.time())
        if reference.month == 1:
            start = datetime(reference.year - 1, 12, 1)
        else:
            start = datetime(reference.year, reference.month - 1, 1)
        return start, last_month_end, "\u4e0a\u4e2a\u6708"
    return None, None, None


def _remove_metadata_words(query: str, time_expression: str | None) -> str:
    cleaned = query
    if time_expression:
        cleaned = cleaned.replace(time_expression, " ")
    cleaned = re.sub(
        r"(\u62cd\u6444\u7684|\u62cd\u7684|\u62cd\u6444|\u7167\u7247|\u56fe\u7247|\u76f8\u7247|\u5728)",
        " ",
        cleaned,
    )
    cleaned = re.sub(
        r"\b(photos?|pictures?|images?|taken|captured|from|in)\b",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，,。.")
    return cleaned or query.strip()


def build_search_plan(
    query: str,
    *,
    captured_from: datetime | date | str | None = None,
    captured_to: datetime | date | str | None = None,
    bounds: GeoBounds | None = None,
    reference_date: date | datetime | str | None = None,
    require_reliable_time: bool = True,
    require_reliable_gps: bool = True,
) -> SearchPlan:
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if isinstance(reference_date, str):
        reference_date = date.fromisoformat(reference_date)
    reference = _reference_date(reference_date)
    parsed_from, parsed_to, expression = _parse_time(query, reference)
    start = _parse_datetime(captured_from) if captured_from is not None else parsed_from
    end = _parse_datetime(captured_to) if captured_to is not None else parsed_to
    metadata = MetadataFilter(start, end, bounds, require_reliable_time, require_reliable_gps)
    return SearchPlan(query, _remove_metadata_words(query, expression), metadata, expression)


def _parse_datetime(value: datetime | date | str) -> datetime:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)


def filter_records(
    records: list[PhotoRecord],
    metadata_filter: MetadataFilter,
    *,
    fallback_if_unavailable: bool = False,
) -> tuple[list[PhotoRecord], bool]:
    """Apply strict filters; optionally fall back only when metadata is absent globally."""
    if metadata_filter.is_empty:
        return records, False
    if fallback_if_unavailable:
        if metadata_filter.requires_time and not any(
            item.timestamp is not None
            and (
                not metadata_filter.require_reliable_time
                or item.captured_at_source == "exif"
            )
            for item in records
        ):
            return records, True
        if metadata_filter.requires_gps and not any(
            item.latitude is not None
            and item.longitude is not None
            and (
                not metadata_filter.require_reliable_gps or item.gps_source == "exif"
            )
            for item in records
        ):
            return records, True
    return [item for item in records if metadata_filter.matches(item)], False
