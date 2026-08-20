"""Generate and persist Ver4 event/journey names without an LLM."""

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from memora.clustering.journey import JourneyConfig, NamedLocation
from memora.clustering.location_inference import infer_journey_config
from memora.config import settings
from memora.encoders.clip_encoder import create_encoder
from memora.generation.narrative import ChatCompletionsBackend
from memora.service import MemoraService


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate template Event/Journey narratives")
    parser.add_argument("--index-path", default="data/index-openclip-prepared-v4-enriched.json")
    parser.add_argument("--people-path", default="data/people-v4.json")
    parser.add_argument("--output", default="data/ver4_named_events_journeys.json")
    parser.add_argument("--encoder", choices=["open_clip", "lightweight"], default="open_clip")
    parser.add_argument("--home-name")
    parser.add_argument("--home-lat", type=float)
    parser.add_argument("--home-lon", type=float)
    parser.add_argument("--home-radius-km", type=float)
    parser.add_argument("--max-gap-days", type=float, default=5.0)
    parser.add_argument("--stop-radius-km", type=float, default=150.0)
    parser.add_argument("--location-cluster-radius-km", type=float, default=40.0)
    parser.add_argument("--minimum-destination-photos", type=int, default=2)
    parser.add_argument("--no-geocoding", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--llm-timeout", type=float, default=5.0)
    parser.add_argument(
        "--destination",
        nargs=4,
        action="append",
        metavar=("NAME", "LAT", "LON", "RADIUS_KM"),
    )
    args = parser.parse_args()

    service = MemoraService(create_encoder(args.encoder), args.index_path)
    manual_home_values = (args.home_lat, args.home_lon)
    if any(value is not None for value in manual_home_values) and not all(
        value is not None for value in manual_home_values
    ):
        raise SystemExit("--home-lat and --home-lon must be provided together")
    if all(value is not None for value in manual_home_values):
        destinations = tuple(
            NamedLocation(name, float(latitude), float(longitude), float(radius))
            for name, latitude, longitude, radius in (args.destination or [])
        )
        config = JourneyConfig(
            home=NamedLocation(
                args.home_name or "Home",
                args.home_lat,
                args.home_lon,
                args.home_radius_km or 80.0,
            ),
            destinations=destinations,
            max_gap_days=args.max_gap_days,
            stop_radius_km=args.stop_radius_km,
        )
        location_mode = "manual_override"
        location_clusters = []
    else:
        config, location_clusters = infer_journey_config(
            service.records,
            cluster_radius_km=args.location_cluster_radius_km,
            minimum_destination_photos=args.minimum_destination_photos,
            max_gap_days=args.max_gap_days,
            stop_radius_km=args.stop_radius_km,
            geocode=not args.no_geocoding,
        )
        location_mode = "auto_gps"

    llm_configured = bool(settings.llm_url and settings.llm_model)
    backend = None
    if not args.no_llm and llm_configured:
        backend = ChatCompletionsBackend(
            settings.llm_url or "",
            settings.llm_model or "",
            settings.llm_api_key,
            timeout=args.llm_timeout,
        )
    events, journeys = service.journeys(
        config,
        people_path=args.people_path,
        strategy="strict_event_people",
        backend=backend,
    )

    filename_by_id = {record.id: Path(record.path).name for record in service.records}
    event_payload = []
    for event in events:
        value = asdict(event)
        value["photo_files"] = [filename_by_id.get(photo_id, photo_id) for photo_id in event.photo_ids]
        event_payload.append(value)

    event_photos = {event.id: set(event.photo_ids) for event in events}
    journey_payload = []
    for journey in journeys:
        value = asdict(journey)
        photo_ids = set(journey.loose_photo_ids)
        for event_id in journey.event_ids:
            photo_ids.update(event_photos.get(event_id, set()))
        value["photo_files"] = sorted(filename_by_id.get(photo_id, photo_id) for photo_id in photo_ids)
        for stop in value["stops"]:
            stop["photo_files"] = [
                filename_by_id.get(photo_id, photo_id) for photo_id in stop.get("photo_ids", [])
            ]
        journey_payload.append(value)

    llm_event_count = sum(event.name_source == "llm" for event in events)
    llm_journey_count = sum(journey.name_source == "llm" for journey in journeys)
    llm_used = llm_event_count + llm_journey_count > 0
    if llm_used:
        generation_mode = "llm_preferred_with_template_fallback"
    elif args.no_llm:
        generation_mode = "template_llm_disabled"
    elif llm_configured:
        generation_mode = "template_fallback_llm_unavailable"
    else:
        generation_mode = "template_llm_not_configured"
    payload = {
        "version": "ver4-narrative-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generation_mode": generation_mode,
        "llm_configured": llm_configured,
        "llm_used": llm_used,
        "llm_event_count": llm_event_count,
        "llm_journey_count": llm_journey_count,
        "llm_error": backend.last_error if backend else None,
        "index_path": args.index_path,
        "people_path": args.people_path,
        "location_mode": location_mode,
        "location_clusters": [cluster.to_dict() for cluster in location_clusters],
        "config": config.to_dict(),
        "event_count": len(events),
        "journey_count": len(journeys),
        "events": event_payload,
        "journeys": journey_payload,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "generation_mode": payload["generation_mode"],
                "llm_configured": llm_configured,
                "llm_used": llm_used,
                "llm_error": payload["llm_error"],
                "location_mode": location_mode,
                "inferred_home": config.home.name,
                "inferred_destination_count": len(config.destinations),
                "event_count": len(events),
                "named_event_count": sum(bool(event.name) for event in events),
                "journey_count": len(journeys),
                "named_journey_count": sum(bool(journey.name) for journey in journeys),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
