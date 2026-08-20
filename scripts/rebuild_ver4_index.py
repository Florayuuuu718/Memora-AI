"""Enrich an OpenCLIP index with InsightFace and quality signals.

The regular indexer deliberately has no InsightFace dependency. This script is
the reproducible second pass used for Ver4/Ver5 experiments: it detects faces
once, keeps the appearance descriptors in the people index, and writes the
corresponding face-quality and composition scores back to the photo index.
"""

import argparse
import json

from memora.clustering.people import cluster_people, save_people_index
from memora.encoders.face_encoder import InsightFaceEncoder
from memora.quality.best_shot import score_photo
from memora.retrieval.vector_store import NumpyVectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich an index with people and photo-quality signals")
    parser.add_argument("--index-path", required=True, help="Input OpenCLIP index")
    parser.add_argument("--output-index-path", required=True, help="Output enriched index")
    parser.add_argument("--people-path", required=True, help="Output people index")
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--ctx-id", type=int, default=0, help="0 for GPU, -1 for CPU")
    parser.add_argument("--eps", type=float, default=0.35)
    parser.add_argument("--min-samples", type=int, default=2)
    args = parser.parse_args()

    store = NumpyVectorStore.load(args.index_path)
    encoder = InsightFaceEncoder(model_name=args.model_name, ctx_id=args.ctx_id)
    people = cluster_people(
        store.records,
        encoder,
        model_name=args.model_name,
        eps=args.eps,
        min_samples=args.min_samples,
    )

    faces_by_photo: dict[str, list[dict[str, object]]] = {}
    for face in people.faces:
        faces_by_photo.setdefault(face.photo_id, []).append(
            {"bbox": face.bbox, "det_score": face.det_score}
        )
    for record in store.records:
        record.quality = score_photo(record.path, faces=faces_by_photo.get(record.id, []))

    output_store = NumpyVectorStore(store.records)
    output_store.save(args.output_index_path)
    save_people_index(people, args.people_path)
    quality_fields = sorted({key for record in store.records for key in record.quality})
    print(
        json.dumps(
            {
                "input_index_path": args.index_path,
                "output_index_path": args.output_index_path,
                "people_path": args.people_path,
                "photo_count": len(store.records),
                "face_count": len(people.faces),
                "people_group_count": len(people.groups),
                "quality_fields": quality_fields,
                "photos_with_faces": len(faces_by_photo),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
