from collections.abc import Mapping, Sequence

from memora.evaluation.events import _pairwise_metrics, _valid_label
from memora.models import EventGroup, JourneyGroup, PhotoRecord


def journey_labels(
    records: Sequence[PhotoRecord],
    journeys: Sequence[JourneyGroup],
    events: Sequence[EventGroup],
) -> list[int]:
    event_photos = {event.id: event.photo_ids for event in events}
    labels: dict[str, int] = {}
    for journey in journeys:
        for event_id in journey.event_ids:
            for photo_id in event_photos.get(event_id, []):
                labels[photo_id] = journey.id
        for photo_id in journey.loose_photo_ids:
            labels[photo_id] = journey.id
    return [labels.get(record.id, -1) for record in records]


def journey_stop_labels(
    records: Sequence[PhotoRecord],
    journeys: Sequence[JourneyGroup],
) -> list[str | int]:
    """Return labels for city/region stops nested inside each journey."""
    labels: dict[str, str] = {}
    for journey in journeys:
        for stop in journey.stops:
            stop_id = f"{journey.id}:{stop.get('id', 0)}"
            for photo_id in stop.get("photo_ids", []):
                labels[str(photo_id)] = stop_id
    return [labels.get(record.id, -1) for record in records]


def evaluate_journeys(
    records: list[PhotoRecord],
    journeys: list[JourneyGroup],
    events: list[EventGroup],
    expected_labels: Mapping[str, object] | Sequence[object],
) -> dict[str, float]:
    if isinstance(expected_labels, Mapping):
        expected = [expected_labels.get(record.id) for record in records]
    else:
        expected = list(expected_labels)
        if len(expected) != len(records):
            raise ValueError("expected_labels must have one label per record")
    predicted = journey_labels(records, journeys, events)
    metrics = _pairwise_metrics(predicted, expected)
    labelled = [index for index, value in enumerate(expected) if _valid_label(value)]
    assigned = sum(predicted[index] >= 0 for index in labelled)
    metrics["coverage"] = assigned / len(labelled) if labelled else 0.0
    return metrics


def evaluate_journey_hierarchy(
    records: list[PhotoRecord],
    journeys: list[JourneyGroup],
    events: list[EventGroup],
    expected_parent_labels: Mapping[str, object] | Sequence[object],
    expected_stop_labels: Mapping[str, object] | Sequence[object],
) -> dict[str, dict[str, float]]:
    """Evaluate both the overall trip and its city/region stop boundaries."""
    return {
        "parent_journey": evaluate_journeys(
            records,
            journeys,
            events,
            expected_parent_labels,
        ),
        "journey_stop": _evaluate_labels(
            records,
            journey_stop_labels(records, journeys),
            expected_stop_labels,
        ),
    }


def _evaluate_labels(
    records: list[PhotoRecord],
    predicted: Sequence,
    expected_labels: Mapping[str, object] | Sequence[object],
) -> dict[str, float]:
    if isinstance(expected_labels, Mapping):
        expected = [expected_labels.get(record.id) for record in records]
    else:
        expected = list(expected_labels)
        if len(expected) != len(records):
            raise ValueError("expected_labels must have one label per record")
    metrics = _pairwise_metrics(predicted, expected)
    labelled = [index for index, value in enumerate(expected) if _valid_label(value)]
    assigned = sum(predicted[index] != -1 for index in labelled)
    metrics["coverage"] = assigned / len(labelled) if labelled else 0.0
    return metrics
