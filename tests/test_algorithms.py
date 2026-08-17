from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from memora.clustering.event_cluster import cluster_events
from memora.clustering.people import PeopleIndex, _build_groups, apply_feedback
from memora.duplicate.phash import hamming_distance, phash
from memora.encoders.clip_encoder import HashImageEncoder
from memora.evaluation.clustering import pairwise_f1
from memora.evaluation.retrieval import RetrievalCase, evaluate_strategies, recall_at_k
from memora.models import FaceRecord, PersonGroup, PhotoRecord
from memora.quality.best_shot import score_photo
from memora.retrieval.brute_force import search
from memora.retrieval.metadata_filter import GeoBounds, build_search_plan, filter_records
from memora.retrieval.query_expansion import expand_query, query_texts
from scripts.prepare_dataset import prepare_dataset


def make_record(identifier: str, timestamp: datetime, vector: list[float]) -> PhotoRecord:
    return PhotoRecord(id=identifier, path=f"{identifier}.jpg", captured_at=timestamp.isoformat(), embedding=vector)


def test_hash_encoder_is_normalized(tmp_path):
    path = tmp_path / "one.png"
    Image.new("RGB", (40, 40), (255, 0, 0)).save(path)
    vector = HashImageEncoder().encode_image(path)
    assert vector.shape == (256,)
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_brute_force_returns_highest_score_first():
    records = [make_record("a", datetime.now(), [1, 0]), make_record("b", datetime.now(), [0, 1])]
    results = search(records, np.asarray([1, 0], dtype=np.float32), top_k=2)
    assert [result.photo_id for result in results] == ["a", "b"]


def test_event_clustering_splits_large_time_gap():
    now = datetime.now()
    records = [make_record("a", now, [1, 0]), make_record("b", now + timedelta(minutes=10), [1, 0]), make_record("c", now + timedelta(days=1), [1, 0])]
    events = cluster_events(records, geo_weight=0.0)
    assert len(events) == 2
    assert events[0].photo_ids == ["a", "b"]


def test_phash_and_quality(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (80, 80), "white").save(first)
    Image.new("RGB", (80, 80), "white").save(second)
    assert hamming_distance(phash(first), phash(second)) == 0
    assert score_photo(str(first))["score"] >= 0


def test_pairwise_f1_is_perfect_for_matching_clusters():
    metrics = pairwise_f1([0, 0, 1, 1], [4, 4, 9, 9])
    assert metrics["f1"] == 1.0


def test_query_strategies_are_distinct_and_support_chinese_expansion():
    assert "a beach" in expand_query("海边")
    assert len(query_texts("海边", "raw_clip")) == 1
    assert len(query_texts("海边", "prompt_ensemble")) == 5
    assert len(query_texts("海边", "query_enhancement")) == 5


def test_recall_and_strategy_evaluation():
    records = [make_record("a", datetime.now(), [1, 0]), make_record("b", datetime.now(), [1, 0]), make_record("c", datetime.now(), [0, 1])]
    assert recall_at_k(search(records, np.asarray([1, 0], dtype=np.float32), top_k=2), {"a", "b"}, 1) == 0.5
    vector_a = [1.0] + [0.0] * 255
    vector_c = [0.0, 1.0] + [0.0] * 254
    eval_records = [make_record("a", datetime.now(), vector_a), make_record("b", datetime.now(), vector_a), make_record("c", datetime.now(), vector_c)]
    metrics = evaluate_strategies(eval_records, HashImageEncoder(), [RetrievalCase("dog", frozenset({"a"}))], ks=(1, 5, 10))
    assert set(metrics) == {"raw_clip", "prompt_ensemble", "query_enhancement"}
    assert set(metrics["raw_clip"]) == {"recall@1", "recall@5", "recall@10"}


def test_prepare_dataset_converts_jpeg_and_writes_manifest(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "prepared"
    source.mkdir()
    Image.new("RGB", (20, 20), "blue").save(source / "original.jpg")

    payload = prepare_dataset(source, output, tmp_path / "manifest.json")

    assert payload["count"] == 1
    assert (output / "000001.jpg").exists()
    assert payload["records"][0]["metadata_preserved"] is True


def test_metadata_plan_extracts_year_and_filters_untrusted_time():
    records = [
        PhotoRecord("exif", "exif.jpg", captured_at="2025-06-01T12:00:00", captured_at_source="exif"),
        PhotoRecord("filesystem", "filesystem.jpg", captured_at="2025-06-01T12:00:00", captured_at_source="filesystem"),
    ]
    plan = build_search_plan("去年在海边拍的照片", reference_date="2026-08-17")
    assert plan.semantic_query == "海边"
    filtered, fallback = filter_records(records, plan.metadata_filter)
    assert [record.id for record in filtered] == ["exif"]
    assert fallback is False


def test_metadata_filter_supports_gps_bbox_and_missing_metadata_fallback():
    records = [
        PhotoRecord("inside", "inside.jpg", latitude=30.5, longitude=120.5, gps_source="exif"),
        PhotoRecord("outside", "outside.jpg", latitude=31.5, longitude=121.5, gps_source="exif"),
    ]
    metadata_filter = build_search_plan("海边", bounds=GeoBounds(30.0, 120.0, 31.0, 121.0)).metadata_filter
    filtered, fallback = filter_records(records, metadata_filter)
    assert [record.id for record in filtered] == ["inside"]
    assert fallback is False

    no_gps = [PhotoRecord("unknown", "unknown.jpg")]
    filtered, fallback = filter_records(no_gps, metadata_filter, fallback_if_unavailable=True)
    assert [record.id for record in filtered] == ["unknown"]
    assert fallback is True


def test_people_clustering_builds_prototypes_and_noise():
    faces = [
        FaceRecord("a", "photo-a", embedding=[1.0, 0.0], det_score=0.99),
        FaceRecord("b", "photo-b", embedding=[0.99, 0.01], det_score=0.90),
        FaceRecord("c", "photo-c", embedding=[0.0, 1.0], det_score=0.95),
        FaceRecord("d", "photo-d", embedding=[0.01, 0.99], det_score=0.80),
        FaceRecord("noise", "photo-noise", embedding=[-1.0, 0.0], det_score=0.50),
    ]
    groups, noise = _build_groups(faces, eps=0.05, min_samples=2)
    assert len(groups) == 2
    assert noise == ["noise"]
    assert all(np.isclose(np.linalg.norm(group.prototype), 1.0) for group in groups)


def test_people_feedback_merges_groups_and_removes_photo():
    index = PeopleIndex(
        faces=[
            FaceRecord("f3", "photo-3", embedding=[1.0, 0.0], det_score=1.0),
            FaceRecord("f7", "photo-7", embedding=[0.0, 1.0], det_score=1.0),
        ],
        groups=[
            PersonGroup(3, ["f3"], ["photo-3"], [1.0, 0.0]),
            PersonGroup(7, ["f7"], ["photo-7"], [0.0, 1.0]),
        ],
    )
    updated = apply_feedback(
        index,
        merges=[[3, 7]],
        removed_photos=[{"person_id": 3, "photo_id": "photo-7"}],
    )
    assert [group.id for group in updated.groups] == [3]
    assert updated.groups[0].photo_ids == ["photo-3"]
    assert updated.groups[0].face_ids == ["f3"]
    assert updated.groups[0].removed_photo_ids == ["photo-7"]
