import argparse
import json

from memora.encoders.clip_encoder import create_encoder
from memora.service import MemoraService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--index-path", default="data/index.json")
    parser.add_argument("--encoder", default="lightweight")
    args = parser.parse_args()
    service = MemoraService(create_encoder(args.encoder), args.index_path)
    print(json.dumps({"count": len(service.index(args.directory))}, indent=2))


if __name__ == "__main__":
    main()

