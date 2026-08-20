from collections.abc import Mapping, Sequence

from memora.clustering.event_cluster import EventStrategy, cluster_events
from memora.models import PhotoRecord


EVENT_STRATEGIES: tuple[EventStrategy, ...] = (
    "time_only",
    "time_clip",
    "time_clip_gps",
    "strict_event",
    "strict_event_people",
)


def event_labels(records: Sequence[PhotoRecord], groups) -> list[int]:
    """Convert event groups into labels aligned with ``records``."""
    labels = {-1: -1}
    for group in groups:
        for photo_id in group.photo_ids:
            labels[photo_id] = group.id
    return [labels.get(record.id, -1) for record in records]


def _valid_label(value) -> bool:
    return value not in {None, -1, "", "review"}


def _pairwise_metrics(predicted: Sequence[int], expected: Sequence) -> dict[str, float]:
    true_positive = false_positive = false_negative = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            if not _valid_label(expected[left]) or not _valid_label(expected[right]):
                continue
            same_predicted = (
                _valid_label(predicted[left])
                and _valid_label(predicted[right])
                and predicted[left] == predicted[right]
            )
            same_expected = expected[left] == expected[right]
            true_positive += int(same_predicted and same_expected)
            false_positive += int(same_predicted and not same_expected)
            false_negative += int(not same_predicted and same_expected)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def event_boundary_metrics(
    predicted: Sequence[int],
    strict_expected: Sequence,
    tolerant_expected: Sequence,
) -> dict[str, object]:
    """Report strict and user-tolerant event boundaries side by side.

    ``strict_expected`` preserves the annotator's preferred event split.
    ``tolerant_expected`` may join nearby event families when either boundary
    is acceptable. A tolerated merge is therefore visible, but is not counted
    as a hard false-positive in the tolerant score.
    """
    if not (len(predicted) == len(strict_expected) == len(tolerant_expected)):
        raise ValueError("predicted and expected labels must have equal length")
    tolerated_merge_pairs = 0
    hard_false_positive_pairs = 0
    for left in range(len(predicted)):
        for right in range(left + 1, len(predicted)):
            same_predicted = (
                _valid_label(predicted[left])
                and _valid_label(predicted[right])
                and predicted[left] == predicted[right]
            )
            if not same_predicted:
                continue
            strict_same = (
                _valid_label(strict_expected[left])
                and _valid_label(strict_expected[right])
                and strict_expected[left] == strict_expected[right]
            )
            tolerant_valid = _valid_label(tolerant_expected[left]) and _valid_label(
                tolerant_expected[right]
            )
            tolerant_same = tolerant_valid and tolerant_expected[left] == tolerant_expected[right]
            if not strict_same and tolerant_same:
                tolerated_merge_pairs += 1
            elif tolerant_valid and not tolerant_same:
                hard_false_positive_pairs += 1
    return {
        "strict": _pairwise_metrics(predicted, strict_expected),
        "tolerant": _pairwise_metrics(predicted, tolerant_expected),
        "tolerated_merge_pairs": tolerated_merge_pairs,
        "hard_false_positive_pairs": hard_false_positive_pairs,
    }


def evaluate_event_boundaries(
    records: list[PhotoRecord],
    strict_expected_labels: Mapping[str, object] | Sequence[object],
    tolerant_expected_labels: Mapping[str, object] | Sequence[object],
    *,
    strategy: EventStrategy = "strict_event_people",
    **cluster_kwargs,
) -> dict[str, object]:
    def aligned(values: Mapping[str, object] | Sequence[object]) -> list[object]:
        if isinstance(values, Mapping):
            return [values.get(record.id, -1) for record in records]
        output = list(values)
        if len(output) != len(records):
            raise ValueError("expected_labels must have one label per record")
        return output

    predicted = event_labels(
        records,
        cluster_events(records, strategy=strategy, **cluster_kwargs),
    )
    return event_boundary_metrics(
        predicted,
        aligned(strict_expected_labels),
        aligned(tolerant_expected_labels),
    )


def evaluate_event_strategies(
    records: list[PhotoRecord],
    expected_labels: Mapping[str, int] | Sequence[int],
    *,
    strategies: Sequence[EventStrategy] = EVENT_STRATEGIES,
    **cluster_kwargs,
) -> dict[str, dict[str, float]]:
    """Compare three ablations plus the strict metadata-aware strategy.

    The metric is pairwise event precision/recall/F1: two photos are a
    positive pair when they belong to the same ground-truth event. This is
    robust to arbitrary cluster IDs and is directly comparable to the people
    clustering evaluation already in the project.
    """
    if isinstance(expected_labels, Mapping):
        expected = [expected_labels.get(record.id, -1) for record in records]
    else:
        expected = list(expected_labels)
        if len(expected) != len(records):
            raise ValueError("expected_labels must have one label per record")
    return {
        strategy: _pairwise_metrics(
            event_labels(records, cluster_events(records, strategy=strategy, **cluster_kwargs)),
            expected,
        )
        for strategy in strategies
    }
