import argparse
import json

from memora.encoders.clip_encoder import create_encoder
from memora.service import MemoraService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query")
    parser.add_argument("--index-path", default="data/index.json")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--encoder", default="lightweight", choices=["lightweight", "open_clip"])
    parser.add_argument("--strategy", default="query_enhancement", choices=["raw_clip", "prompt_ensemble", "query_enhancement"])
    args = parser.parse_args()
    service = MemoraService(create_encoder(args.encoder), args.index_path)
    print(json.dumps([result.__dict__ for result in service.search(args.query, args.top_k, strategy=args.strategy)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
