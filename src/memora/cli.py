import argparse
import json

from memora.clustering.people import apply_feedback, cluster_people, load_people_index, save_people_index
from memora.config import settings
from memora.encoders.clip_encoder import create_encoder
from memora.encoders.face_encoder import InsightFaceEncoder
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
    subparsers.add_parser("events")
    subparsers.add_parser("similar")
    for name in ("events", "similar"):
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
    return parser


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
    if args.command in {"people-merge", "people-remove"}:
        people = load_people_index(args.people_path)
        if args.command == "people-merge":
            people = apply_feedback(people, merges=args.merge)
        else:
            people = apply_feedback(
                people,
                removed_photos=[{"person_id": args.person_id, "photo_id": args.photo_id}],
            )
        save_people_index(people, args.people_path)
        print(json.dumps(people.to_dict(), ensure_ascii=False, indent=2))
        return
    service = MemoraService(create_encoder(args.encoder), args.index_path)
    if args.command == "index":
        records = service.index(args.directory)
        print(json.dumps({"count": len(records), "index_path": args.index_path}, ensure_ascii=False, indent=2))
    elif args.command == "search":
        bounds = GeoBounds(*args.bbox) if args.bbox else None
        print(json.dumps(service.search_details(args.query, args.top_k, strategy=args.strategy, captured_from=args.captured_from, captured_to=args.captured_to, bounds=bounds, fallback_if_unavailable=args.fallback_if_unavailable, reference_date=args.reference_date), ensure_ascii=False, indent=2))
    elif args.command == "events":
        print(json.dumps([event.__dict__ for event in service.events()], ensure_ascii=False, indent=2))
    elif args.command == "similar":
        print(json.dumps([group.__dict__ for group in service.similar_groups()], ensure_ascii=False, indent=2))
