import argparse
import json

from memora.evaluation.annotations import annotation_labels, load_annotations, resolve_annotation_labels
from memora.evaluation.events import evaluate_event_boundaries, evaluate_event_strategies
from memora.clustering.people import load_people_index
from memora.retrieval.vector_store import NumpyVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the three event-discovery strategies")
    parser.add_argument("labels", help="CSV/JSON annotations or photo_id-to-event JSON mapping")
    parser.add_argument("--index-path", default="data/index.json")
    parser.add_argument("--max-gap-hours", type=float, default=8.0)
    parser.add_argument("--people-path", default="data/people.json")
    parser.add_argument("--minimum-confidence", choices=["low", "medium", "high"], default="low")
    args = parser.parse_args()
    records = NumpyVectorStore.load(args.index_path).records
    labels = annotation_labels(
        load_annotations(args.labels),
        "event_id",
        minimum_confidence=args.minimum_confidence,
    )
    labels = resolve_annotation_labels(records, labels)
    result = evaluate_event_strategies(
        records,
        labels,
        max_gap_hours=args.max_gap_hours,
        people_index=load_people_index(args.people_path),
    )
    tolerant_labels = annotation_labels(
        load_annotations(args.labels),
        "event_family_id",
        fallback_field="event_id",
        minimum_confidence=args.minimum_confidence,
    )
    tolerant_labels = resolve_annotation_labels(records, tolerant_labels)
    result["strict_event_people_boundary"] = evaluate_event_boundaries(
        records,
        labels,
        tolerant_labels,
        strategy="strict_event_people",
        max_gap_hours=args.max_gap_hours,
        people_index=load_people_index(args.people_path),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
