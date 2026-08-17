"""Detect faces and build DBSCAN-based person groups."""

import argparse
import json

from memora.clustering.people import cluster_people, save_people_index
from memora.encoders.face_encoder import InsightFaceEncoder
from memora.retrieval.vector_store import NumpyVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster faces with InsightFace + DBSCAN")
    parser.add_argument("--index-path", default="data/index-openclip-prepared.json")
    parser.add_argument("--people-path", default="data/people.json")
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--ctx-id", type=int, default=0, help="0 for GPU, -1 for CPU")
    parser.add_argument("--eps", type=float, default=0.35)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()

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
    print(
        json.dumps(
            {
                "photo_count": len({face.photo_id for face in people.faces}),
                "face_count": len(people.faces),
                "group_count": len(people.groups),
                "noise_face_count": len(people.noise_face_ids),
                "people_path": args.people_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
