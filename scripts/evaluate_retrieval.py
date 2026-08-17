import argparse
import json
from pathlib import Path

from memora.encoders.clip_encoder import create_encoder
from memora.evaluation.retrieval import RetrievalCase, evaluate_strategies
from memora.retrieval.vector_store import NumpyVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the three OpenCLIP query strategies")
    parser.add_argument("cases", help="JSON file containing [{\"query\": ..., \"relevant_ids\": [...]}]")
    parser.add_argument("--index-path", default="data/index.json")
    parser.add_argument("--encoder", default="open_clip", choices=["lightweight", "open_clip"])
    args = parser.parse_args()

    raw_cases = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    cases = [RetrievalCase(item["query"], frozenset(item["relevant_ids"])) for item in raw_cases]
    records = NumpyVectorStore.load(args.index_path).records
    metrics = evaluate_strategies(records, create_encoder(args.encoder), cases)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
