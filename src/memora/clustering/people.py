from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from memora.clustering.event_cluster import cluster_embeddings
from memora.clustering.face_cluster import quality_weighted_prototype
from memora.encoders.face_encoder import InsightFaceEncoder
from memora.models import FaceRecord, PersonGroup, PhotoRecord


@dataclass
class PeopleIndex:
    faces: list[FaceRecord] = field(default_factory=list)
    groups: list[PersonGroup] = field(default_factory=list)
    noise_face_ids: list[str] = field(default_factory=list)
    model_name: str = "buffalo_l"
    feedback: dict[str, Any] = field(default_factory=lambda: {"merges": [], "removed_photos": []})

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model_name": self.model_name,
            "faces": [asdict(face) for face in self.faces],
            "groups": [asdict(group) for group in self.groups],
            "noise_face_ids": self.noise_face_ids,
            "feedback": self.feedback,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PeopleIndex":
        return cls(
            faces=[FaceRecord(**item) for item in value.get("faces", [])],
            groups=[PersonGroup(**item) for item in value.get("groups", [])],
            noise_face_ids=list(value.get("noise_face_ids", [])),
            model_name=value.get("model_name", "buffalo_l"),
            feedback=value.get("feedback", {"merges": [], "removed_photos": []}),
        )


def _read_bgr(path: str) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install the people extra before running face clustering") from exc
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    return image


def extract_faces(records: list[PhotoRecord], encoder: InsightFaceEncoder) -> list[FaceRecord]:
    output: list[FaceRecord] = []
    for record in records:
        detections = encoder.detect(_read_bgr(record.path))
        for index, detection in enumerate(detections):
            embedding = np.asarray(detection.get("embedding", []), dtype=np.float32)
            norm = float(np.linalg.norm(embedding))
            if embedding.size == 0 or norm <= 1e-12:
                continue
            embedding = embedding / norm
            output.append(
                FaceRecord(
                    id=f"{record.id}:face-{index}",
                    photo_id=record.id,
                    bbox=[float(value) for value in detection.get("bbox", [])],
                    embedding=embedding.tolist(),
                    det_score=float(detection.get("det_score", 0.0)),
                )
            )
    return output


def _build_groups(faces: list[FaceRecord], eps: float, min_samples: int) -> tuple[list[PersonGroup], list[str]]:
    if not faces:
        return [], []
    matrix = np.asarray([face.embedding for face in faces], dtype=np.float32)
    labels = cluster_embeddings(matrix, eps=eps, min_samples=min_samples)
    groups: list[PersonGroup] = []
    noise: list[str] = []
    for label in sorted(set(int(value) for value in labels)):
        indexes = [index for index, value in enumerate(labels) if int(value) == label]
        if label < 0:
            noise.extend(faces[index].id for index in indexes)
            continue
        selected = [faces[index] for index in indexes]
        embeddings = np.asarray([face.embedding for face in selected], dtype=np.float32)
        qualities = np.asarray([max(face.det_score, 1e-6) for face in selected], dtype=np.float32)
        photo_ids = sorted({face.photo_id for face in selected})
        groups.append(
            PersonGroup(
                id=label,
                face_ids=[face.id for face in selected],
                photo_ids=photo_ids,
                prototype=quality_weighted_prototype(embeddings, qualities).tolist(),
            )
        )
    return groups, noise


def cluster_people(
    records: list[PhotoRecord],
    encoder: InsightFaceEncoder,
    *,
    model_name: str = "buffalo_l",
    eps: float = 0.35,
    min_samples: int = 2,
) -> PeopleIndex:
    faces = extract_faces(records, encoder)
    groups, noise = _build_groups(faces, eps, min_samples)
    return PeopleIndex(faces=faces, groups=groups, noise_face_ids=noise, model_name=model_name)


def _recompute_prototype(group: PersonGroup, faces_by_id: dict[str, FaceRecord]) -> None:
    selected = [faces_by_id[face_id] for face_id in group.face_ids if face_id in faces_by_id]
    if not selected:
        group.prototype = []
        return
    embeddings = np.asarray([face.embedding for face in selected], dtype=np.float32)
    qualities = np.asarray([max(face.det_score, 1e-6) for face in selected], dtype=np.float32)
    group.prototype = quality_weighted_prototype(embeddings, qualities).tolist()


def apply_feedback(
    index: PeopleIndex,
    *,
    merges: list[list[int]] | None = None,
    removed_photos: list[dict[str, Any]] | None = None,
) -> PeopleIndex:
    groups = {
        group.id: PersonGroup(
            id=group.id,
            face_ids=list(group.face_ids),
            photo_ids=list(group.photo_ids),
            prototype=list(group.prototype),
            removed_photo_ids=list(group.removed_photo_ids),
        )
        for group in index.groups
    }
    faces_by_id = {face.id: face for face in index.faces}
    applied_merges = list(index.feedback.get("merges", []))
    applied_removals = list(index.feedback.get("removed_photos", []))

    for merge in merges or []:
        if len(merge) < 2:
            raise ValueError("A merge operation needs at least two person IDs")
        existing = [person_id for person_id in merge if person_id in groups]
        if len(existing) != len(merge):
            raise ValueError(f"Unknown person ID in merge: {merge}")
        target_id = min(existing)
        target = groups[target_id]
        for person_id in existing:
            if person_id == target_id:
                continue
            source = groups.pop(person_id)
            target.face_ids = sorted(set(target.face_ids + source.face_ids))
            target.photo_ids = sorted(set(target.photo_ids + source.photo_ids))
            target.removed_photo_ids = sorted(set(target.removed_photo_ids + source.removed_photo_ids))
        _recompute_prototype(target, faces_by_id)
        applied_merges.append(existing)

    for removal in removed_photos or []:
        person_id = int(removal["person_id"])
        photo_id = str(removal["photo_id"])
        if person_id not in groups:
            raise ValueError(f"Unknown person ID: {person_id}")
        group = groups[person_id]
        group.photo_ids = [value for value in group.photo_ids if value != photo_id]
        group.face_ids = [
            face_id
            for face_id in group.face_ids
            if faces_by_id.get(face_id) is None or faces_by_id[face_id].photo_id != photo_id
        ]
        _recompute_prototype(group, faces_by_id)
        if photo_id not in group.removed_photo_ids:
            group.removed_photo_ids.append(photo_id)
        applied_removals.append({"person_id": person_id, "photo_id": photo_id})

    return PeopleIndex(
        faces=index.faces,
        groups=sorted(groups.values(), key=lambda group: group.id),
        noise_face_ids=index.noise_face_ids,
        model_name=index.model_name,
        feedback={"merges": applied_merges, "removed_photos": applied_removals},
    )


def save_people_index(index: PeopleIndex, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def load_people_index(path: str | Path) -> PeopleIndex:
    path = Path(path)
    if not path.exists():
        return PeopleIndex()
    return PeopleIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))
