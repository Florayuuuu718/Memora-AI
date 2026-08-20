import argparse
import json

from memora.clustering.journey import JourneyConfig, NamedLocation, discover_journeys
from memora.clustering.people import load_people_index
from memora.evaluation.annotations import annotation_labels, load_annotations, resolve_annotation_labels
from memora.evaluation.journeys import evaluate_journey_hierarchy, evaluate_journeys
from memora.retrieval.vector_store import NumpyVectorStore
from memora.clustering.event_cluster import cluster_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Ver4 journey discovery")
    parser.add_argument("labels", help="CSV/JSON annotations containing journey_id")
    parser.add_argument("--index-path", default="data/index.json")
    parser.add_argument("--people-path", default="data/people.json")
    parser.add_argument("--home-name", required=True)
    parser.add_argument("--home-lat", required=True, type=float)
    parser.add_argument("--home-lon", required=True, type=float)
    parser.add_argument("--home-radius-km", default=50.0, type=float)
    parser.add_argument("--max-gap-days", default=5.0, type=float)
    parser.add_argument("--stop-radius-km", default=150.0, type=float)
    parser.add_argument(
        "--destination",
        nargs=4,
        action="append",
        metavar=("NAME", "LAT", "LON", "RADIUS_KM"),
    )
    parser.add_argument("--minimum-confidence", choices=["low", "medium", "high"], default="low")
    args = parser.parse_args()
    records = NumpyVectorStore.load(args.index_path).records
    people = load_people_index(args.people_path)
    events = cluster_events(records, strategy="strict_event_people", people_index=people)
    destinations = tuple(
        NamedLocation(name, float(latitude), float(longitude), float(radius))
        for name, latitude, longitude, radius in (args.destination or [])
    )
    config = JourneyConfig(
        home=NamedLocation(args.home_name, args.home_lat, args.home_lon, args.home_radius_km),
        destinations=destinations,
        max_gap_days=args.max_gap_days,
        stop_radius_km=args.stop_radius_km,
    )
    journeys = discover_journeys(records, events, config)
    labels = annotation_labels(
        load_annotations(args.labels),
        "journey_id",
        minimum_confidence=args.minimum_confidence,
    )
    labels = resolve_annotation_labels(records, labels)
    result = evaluate_journeys(records, journeys, events, labels)
    parent_labels = annotation_labels(
        load_annotations(args.labels),
        "journey_parent_id",
        fallback_field="journey_id",
        minimum_confidence=args.minimum_confidence,
    )
    stop_labels = annotation_labels(
        load_annotations(args.labels),
        "journey_stop_id",
        fallback_field="journey_id",
        minimum_confidence=args.minimum_confidence,
    )
    hierarchy = evaluate_journey_hierarchy(
        records,
        journeys,
        events,
        resolve_annotation_labels(records, parent_labels),
        resolve_annotation_labels(records, stop_labels),
    )
    print(json.dumps({"flat_journey": result, "hierarchy": hierarchy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
