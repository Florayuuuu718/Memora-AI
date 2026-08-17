import argparse
import json

from memora.config import settings
from memora.encoders.clip_encoder import create_encoder
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
    search.add_argument("--index-path", default=argparse.SUPPRESS)
    search.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    subparsers.add_parser("events")
    subparsers.add_parser("similar")
    for name in ("events", "similar"):
        command = subparsers.choices[name]
        command.add_argument("--index-path", default=argparse.SUPPRESS)
        command.add_argument("--encoder", choices=["lightweight", "open_clip"], default=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    service = MemoraService(create_encoder(args.encoder), args.index_path)
    if args.command == "index":
        records = service.index(args.directory)
        print(json.dumps({"count": len(records), "index_path": args.index_path}, ensure_ascii=False, indent=2))
    elif args.command == "search":
        print(json.dumps([result.__dict__ for result in service.search(args.query, args.top_k, strategy=args.strategy)], ensure_ascii=False, indent=2))
    elif args.command == "events":
        print(json.dumps([event.__dict__ for event in service.events()], ensure_ascii=False, indent=2))
    elif args.command == "similar":
        print(json.dumps([group.__dict__ for group in service.similar_groups()], ensure_ascii=False, indent=2))
