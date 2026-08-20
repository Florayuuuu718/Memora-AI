import argparse
import json

from memora.clustering.journey import JourneyConfig, NamedLocation
from memora.clustering.location_inference import infer_journey_config
from memora.clustering.people import (
    apply_feedback,
    cluster_people,
    load_people_index,
    save_people_index,
    set_person_name,
)
from memora.config import settings
from memora.encoders.clip_encoder import create_encoder
from memora.encoders.face_encoder import InsightFaceEncoder
from memora.generation.narrative import ChatCompletionsBackend
from memora.integrations.immich import ImmichClient, ImmichError, sync_immich_assets
from memora.retrieval.metadata_filter import GeoBounds
from memora.retrieval.vector_store import NumpyVectorStore
from memora.service import MemoraService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="memora", description="Memora AI photo understanding CLI")
    parser.add_argument("--index-path", default=settings.index_path)
    parser.add_argument("--encoder", default=settings.encoder, choices=["lightweight", "open_clip"])
    subparsers = parser.add_subparsers(dest="command", required=True)
    index = subparsers.add_parser("index")
    index.add_argument("directory")
    index.add_argument("--index-path", default=argparse.SUPPRESS)
    index.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    search = subparsers.add_parser("search")
    search.add_argument("query")
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--strategy", choices=["raw_clip", "prompt_ensemble", "query_enhancement"], default="query_enhancement")
    search.add_argument("--captured-from")
    search.add_argument("--captured-to")
    search.add_argument("--bbox", nargs=4, type=float, metavar=("MIN_LAT", "MIN_LON", "MAX_LAT", "MAX_LON"))
    search.add_argument("--fallback-if-unavailable", action="store_true")
    search.add_argument("--reference-date", help="Reference date for relative terms such as 去年")
    search.add_argument("--index-path", default=argparse.SUPPRESS)
    search.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    events_command = subparsers.add_parser("events")
    events_command.add_argument("--strategy", choices=["time_only", "time_clip", "time_clip_gps", "strict_event", "strict_event_people"], default="time_clip_gps")
    events_command.add_argument("--people-path", default="data/people.json")
    events_command.add_argument("--generate-names", action="store_true")
    event_llm = events_command.add_mutually_exclusive_group()
    event_llm.add_argument("--use-llm", dest="use_llm", action="store_true")
    event_llm.add_argument("--no-llm", dest="use_llm", action="store_false")
    events_command.set_defaults(use_llm=None)
    similar_command = subparsers.add_parser("similar")
    similar_command.add_argument("--phash-distance", type=int, default=10)
    similar_command.add_argument("--visual-similarity", type=float, default=0.90)
    similar_command.add_argument("--time-window-seconds", type=float, default=30.0)
    journey = subparsers.add_parser("journeys")
    journey.add_argument("--home-name")
    journey.add_argument("--home-lat", type=float)
    journey.add_argument("--home-lon", type=float)
    journey.add_argument("--home-radius-km", type=float)
    journey.add_argument("--destination", nargs=4, action="append", metavar=("NAME", "LAT", "LON", "RADIUS_KM"))
    journey.add_argument("--max-gap-days", type=float, default=5.0)
    journey.add_argument("--stop-radius-km", type=float, default=150.0)
    journey.add_argument("--people-path", default="data/people.json")
    journey.add_argument("--strategy", choices=["strict_event", "strict_event_people"], default="strict_event_people")
    journey_llm = journey.add_mutually_exclusive_group()
    journey_llm.add_argument("--use-llm", dest="use_llm", action="store_true")
    journey_llm.add_argument("--no-llm", dest="use_llm", action="store_false")
    journey.set_defaults(use_llm=None)
    for name in ("events", "similar", "journeys"):
        command = subparsers.choices[name]
        command.add_argument("--index-path", default=argparse.SUPPRESS)
        command.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    people_cluster = subparsers.add_parser("people-cluster")
    people_cluster.add_argument("--index-path", default=argparse.SUPPRESS)
    people_cluster.add_argument("--people-path", default="data/people.json")
    people_cluster.add_argument("--model-name", default="buffalo_l")
    people_cluster.add_argument("--ctx-id", type=int, default=0, help="0 for GPU, -1 for CPU")
    people_cluster.add_argument("--eps", type=float, default=0.35)
    people_cluster.add_argument("--min-samples", type=int, default=2)
    people_merge = subparsers.add_parser("people-merge")
    people_merge.add_argument("--people-path", default="data/people.json")
    people_merge.add_argument("--merge", nargs=2, type=int, action="append", required=True)
    people_remove = subparsers.add_parser("people-remove")
    people_remove.add_argument("--people-path", default="data/people.json")
    people_remove.add_argument("--person-id", type=int, required=True)
    people_remove.add_argument("--photo-id", required=True)
    people_name = subparsers.add_parser("people-name")
    people_name.add_argument("--people-path", default="data/people.json")
    people_name.add_argument("--person-id", type=int, required=True)
    people_name.add_argument("--name", required=True)
    immich_status = subparsers.add_parser("immich-status")
    immich_status.add_argument("--index-path", default=argparse.SUPPRESS)
    immich_status.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    immich_sync = subparsers.add_parser("immich-sync")
    immich_sync.add_argument("--page-size", type=int, default=250)
    immich_sync.add_argument("--force", action="store_true")
    immich_sync.add_argument("--prune-missing", action="store_true")
    immich_sync.add_argument("--index-path", default=argparse.SUPPRESS)
    immich_sync.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    return parser


def _llm_backend(enabled: bool | None):
    if enabled is False:
        return None
    if not settings.llm_url or not settings.llm_model:
        return None
    return ChatCompletionsBackend(
        settings.llm_url,
        settings.llm_model,
        settings.llm_api_key,
    )


def _immich_client() -> ImmichClient:
    if not settings.immich_url or not settings.immich_api_key:
        raise SystemExit("Set MEMORA_IMMICH_URL and MEMORA_IMMICH_API_KEY first")
    return ImmichClient(
        settings.immich_url,
        settings.immich_api_key,
        timeout=settings.immich_timeout_seconds,
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "people-cluster":
        records = NumpyVectorStore.load(args.index_path).records
        encoder = InsightFaceEncoder(model_name=args.model_name, ctx_id=args.ctx_id)
        people = cluster_people(
            records,
            encoder,
            model_name=args.model_name,
            eps=args.eps,
            min_samples=args.min_samples,
        )
        save_people_index(people, args.people_path)
        print(json.dumps({
            "photo_count": len({face.photo_id for face in people.faces}),
            "face_count": len(people.faces),
            "group_count": len(people.groups),
            "noise_face_count": len(people.noise_face_ids),
            "people_path": args.people_path,
        }, ensure_ascii=False, indent=2))
        return
    if args.command in {"people-merge", "people-remove", "people-name"}:
        people = load_people_index(args.people_path)
        if args.command == "people-merge":
            people = apply_feedback(people, merges=args.merge)
        elif args.command == "people-remove":
            people = apply_feedback(
                people,
                removed_photos=[{"person_id": args.person_id, "photo_id": args.photo_id}],
            )
        else:
            people = set_person_name(people, args.person_id, args.name)
        save_people_index(people, args.people_path)
        print(json.dumps(people.to_dict(), ensure_ascii=False, indent=2))
        return
    if args.command == "immich-status":
        try:
            output = _immich_client().status()
        except ImmichError as exc:
            raise SystemExit(f"Immich connection failed: {exc}") from exc
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    if args.command == "immich-sync":
        client = _immich_client()
        try:
            client.status()
            service = MemoraService(create_encoder(args.encoder), args.index_path)
            output = sync_immich_assets(
                client,
                service.encoder,
                service.store,
                service.index_path,
                settings.immich_cache_path,
                page_size=args.page_size,
                force=args.force,
                prune_missing=args.prune_missing,
            ).to_dict()
        except ImmichError as exc:
            raise SystemExit(f"Immich sync failed: {exc}") from exc
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    service = MemoraService(create_encoder(args.encoder), args.index_path)
    if args.command == "index":
        records = service.index(args.directory)
        print(json.dumps({"count": len(records), "index_path": args.index_path}, ensure_ascii=False, indent=2))
    elif args.command == "search":
        bounds = GeoBounds(*args.bbox) if args.bbox else None
        print(json.dumps(service.search_details(args.query, args.top_k, strategy=args.strategy, captured_from=args.captured_from, captured_to=args.captured_to, bounds=bounds, fallback_if_unavailable=args.fallback_if_unavailable, reference_date=args.reference_date), ensure_ascii=False, indent=2))
    elif args.command == "events":
        if args.generate_names:
            try:
                inferred_config, _ = infer_journey_config(service.records)
                locations = (inferred_config.home,) + inferred_config.destinations
            except ValueError:
                locations = ()
            events = service.named_events(
                strategy=args.strategy,
                people_path=args.people_path,
                backend=_llm_backend(args.use_llm),
                locations=locations,
            )
        else:
            events = service.events(strategy=args.strategy, people_path=args.people_path)
        print(json.dumps([event.__dict__ for event in events], ensure_ascii=False, indent=2))
    elif args.command == "similar":
        print(json.dumps([group.__dict__ for group in service.similar_groups(phash_distance=args.phash_distance, visual_similarity=args.visual_similarity, time_window_seconds=args.time_window_seconds)], ensure_ascii=False, indent=2))
    elif args.command == "journeys":
        if (args.home_lat is None) != (args.home_lon is None):
            raise SystemExit("--home-lat and --home-lon must be provided together")
        if args.home_lat is None:
            config, location_clusters = infer_journey_config(
                service.records,
                max_gap_days=args.max_gap_days,
                stop_radius_km=args.stop_radius_km,
            )
            location_mode = "auto_gps"
        else:
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
            location_clusters = []
            location_mode = "manual_override"
        events, journeys = service.journeys(
            config,
            people_path=args.people_path,
            strategy=args.strategy,
            backend=_llm_backend(args.use_llm),
        )
        print(json.dumps({
            "config": config.to_dict(),
            "location_mode": location_mode,
            "location_clusters": [cluster.to_dict() for cluster in location_clusters],
            "event_count": len(events),
            "journeys": [journey.__dict__ for journey in journeys],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
