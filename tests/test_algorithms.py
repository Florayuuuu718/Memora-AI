from datetime import datetime, timedelta

import numpy as np
import pytest
from PIL import Image

from memora.clustering.event_cluster import cluster_events
from memora.clustering.event_people import cluster_events_with_people
from memora.clustering.journey import JourneyConfig, NamedLocation, discover_journeys
from memora.clustering.location_inference import infer_journey_config
from memora.clustering.people import PeopleIndex, _build_groups, apply_feedback
from memora.duplicate.phash import hamming_distance, phash
from memora.duplicate.similar_group import best_shot_by_group, group_similar
from memora.encoders.clip_encoder import HashImageEncoder
from memora.evaluation.benchmark import benchmark_vector_indexes
from memora.evaluation.clustering import pairwise_f1
from memora.evaluation.events import event_boundary_metrics, evaluate_event_strategies
from memora.evaluation.annotations import annotation_labels, load_annotations
from memora.evaluation.journeys import evaluate_journey_hierarchy, evaluate_journeys
from memora.evaluation.retrieval import RetrievalCase, evaluate_strategies, recall_at_k
from memora.models import FaceRecord, PersonGroup, PhotoRecord
from memora.models import EventGroup
from memora.generation.narrative import (
    ChatCompletionsBackend,
    NarrativeBackendUnavailable,
    generate_event_narrative,
    generate_journey_narrative,
)
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


def test_event_strategies_are_evaluable():
    now = datetime.now()
    records = [
        make_record("a", now, [1, 0]),
        make_record("b", now + timedelta(minutes=10), [1, 0]),
        make_record("c", now + timedelta(days=1), [0, 1]),
    ]
    metrics = evaluate_event_strategies(records, {"a": 0, "b": 0, "c": 1}, geo_weight=0.0)
    assert set(metrics) == {
        "time_only",
        "time_clip",
        "time_clip_gps",
        "strict_event",
        "strict_event_people",
    }
    assert all(set(value) == {"precision", "recall", "f1"} for value in metrics.values())


def test_strict_events_do_not_merge_upload_batch_by_filesystem_time():
    now = datetime.now()
    records = [
        PhotoRecord("a", "a.jpg", captured_at=now.isoformat(), captured_at_source="filesystem", embedding=[1.0, 0.0]),
        PhotoRecord("b", "b.jpg", captured_at=(now + timedelta(seconds=2)).isoformat(), captured_at_source="filesystem", embedding=[1.0, 0.0]),
        PhotoRecord("c", "c.jpg", captured_at=(now + timedelta(seconds=4)).isoformat(), captured_at_source="filesystem", embedding=[0.0, 1.0]),
    ]
    events = cluster_events(records, strategy="strict_event")
    assert sorted(len(event.photo_ids) for event in events) == [1, 2]
    assert all(event.start is None for event in events)
    assert all(event.evidence == "strict_clip_fallback" for event in events)


def test_people_evidence_can_link_related_upload_photos():
    now = datetime.now()
    records = [
        PhotoRecord("a", "a.jpg", captured_at=now.isoformat(), captured_at_source="filesystem", embedding=[1.0, 0.0]),
        PhotoRecord("b", "b.jpg", captured_at=now.isoformat(), captured_at_source="filesystem", embedding=[0.8, 0.6]),
    ]
    people = PeopleIndex(groups=[PersonGroup(0, ["fa", "fb"], ["a", "b"], [1.0, 0.0])])
    assert len(cluster_events(records, strategy="strict_event")) == 2
    events = cluster_events_with_people(records, people)
    assert len(events) == 1
    assert events[0].person_ids == [0]


def test_journey_discovery_and_template_narratives():
    records = [
        PhotoRecord("a", "a.jpg", embedding=[1.0, 0.0]),
        PhotoRecord("b", "b.jpg", embedding=[1.0, 0.0]),
        PhotoRecord("c", "c.jpg", embedding=[1.0, 0.0]),
    ]
    events = [
        EventGroup(0, ["a", "b"], "2025-03-01T10:00:00", "2025-03-01T12:00:00", 10.0, 10.0, person_ids=[0], activity_tags=["城市漫步"]),
        EventGroup(1, ["c"], "2025-03-02T10:00:00", "2025-03-02T10:00:00", 10.1, 10.1, person_ids=[0]),
    ]
    people = PeopleIndex(groups=[PersonGroup(0, [], ["a", "b", "c"], [], name="小林")])
    named_event = generate_event_narrative(events[0], {record.id: record for record in records}, people, place_name="测试城")
    events[0] = named_event
    config = JourneyConfig(
        home=NamedLocation("家", 0.0, 0.0, 50.0),
        destinations=(NamedLocation("测试城", 10.0, 10.0, 100.0),),
    )
    journeys = discover_journeys(records, events, config)
    assert len(journeys) == 1
    assert journeys[0].event_ids == [0]
    assert journeys[0].loose_photo_ids == ["c"]
    named_journey = generate_journey_narrative(journeys[0], {event.id: event for event in events}, people)
    assert named_event.name == "测试城城市漫步"
    assert named_journey.name == "测试城之旅"
    assert "小林" in named_journey.note
    metrics = evaluate_journeys(records, journeys, events, {"a": "J1", "b": "J1", "c": "J1"})
    assert metrics["f1"] == 1.0
    assert metrics["coverage"] == 1.0


def test_rich_annotation_confidence_filter(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "photo_id,event_id,journey_id,label_confidence\n"
        "a,E1,J1,high\n"
        "b,E1,J1,medium\n"
        "c,review,J1,low\n",
        encoding="utf-8",
    )
    annotations = load_annotations(path)
    assert annotation_labels(annotations, "event_id", minimum_confidence="medium") == {"a": "E1", "b": "E1"}


def test_annotation_fallback_supports_tolerant_boundaries(tmp_path):
    path = tmp_path / "labels.csv"
    path.write_text(
        "photo_id,event_id,event_family_id,label_confidence\n"
        "a,E1,,high\n"
        "b,E2,FAMILY,high\n"
        "c,E3,FAMILY,high\n",
        encoding="utf-8",
    )
    annotations = load_annotations(path)
    assert annotation_labels(
        annotations,
        "event_family_id",
        fallback_field="event_id",
    ) == {"a": "E1", "b": "FAMILY", "c": "FAMILY"}


def test_event_boundary_metrics_separate_tolerated_merges():
    result = event_boundary_metrics(
        predicted=[0, 0, 0],
        strict_expected=["E1", "E2", "E3"],
        tolerant_expected=["FAMILY", "FAMILY", "E3"],
    )
    assert result["tolerated_merge_pairs"] == 1
    assert result["hard_false_positive_pairs"] == 2
    assert result["tolerant"]["precision"] == 1 / 3


def test_journey_splits_one_large_trip_into_geographic_stops():
    records = [
        PhotoRecord("hk", "hk.jpg", captured_at="2024-07-26T10:00:00"),
        PhotoRecord("bos", "bos.jpg", captured_at="2024-07-28T10:00:00"),
    ]
    events = [
        EventGroup(0, ["hk"], "2024-07-26T10:00:00", "2024-07-26T10:00:00", 22.3, 114.2),
        EventGroup(1, ["bos"], "2024-07-28T10:00:00", "2024-07-28T10:00:00", 42.36, -71.09),
    ]
    config = JourneyConfig(
        home=NamedLocation("Home", 30.66, 104.06, 80),
        destinations=(
            NamedLocation("Hong Kong", 22.3, 114.2, 100),
            NamedLocation("Boston", 42.36, -71.09, 100),
        ),
    )
    journeys = discover_journeys(records, events, config)
    assert len(journeys) == 1
    assert [stop["name"] for stop in journeys[0].stops] == ["Hong Kong", "Boston"]
    metrics = evaluate_journey_hierarchy(
        records,
        journeys,
        events,
        {"hk": "BIG", "bos": "BIG"},
        {"hk": "HK", "bos": "BOS"},
    )
    assert metrics["parent_journey"]["f1"] == 1.0
    assert metrics["journey_stop"]["f1"] == 0.0


def test_location_config_is_inferred_from_recurring_gps():
    records = [
        PhotoRecord("h1", "h1.jpg", captured_at="2025-01-01T10:00:00", latitude=30.0, longitude=104.0),
        PhotoRecord("h2", "h2.jpg", captured_at="2025-02-01T10:00:00", latitude=30.01, longitude=104.01),
        PhotoRecord("h3", "h3.jpg", captured_at="2025-03-01T10:00:00", latitude=30.02, longitude=104.02),
        PhotoRecord("t1", "t1.jpg", captured_at="2025-02-10T10:00:00", latitude=22.3, longitude=114.2),
        PhotoRecord("t2", "t2.jpg", captured_at="2025-02-11T10:00:00", latitude=22.31, longitude=114.21),
    ]
    config, clusters = infer_journey_config(
        records,
        cluster_radius_km=20,
        minimum_destination_photos=2,
        geocode=False,
    )
    assert len(clusters) == 2
    assert clusters[0].is_home is True
    assert np.isclose(config.home.latitude, 30.01)
    assert len(config.destinations) == 1
    assert config.destinations[0].name.startswith("GPS 22.305")


def test_llm_backend_circuit_breaker_falls_back_after_one_failure(monkeypatch):
    calls = 0

    def unavailable(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise OSError("offline")

    monkeypatch.setattr("memora.generation.narrative.urlopen", unavailable)
    backend = ChatCompletionsBackend("http://localhost:1", "test", timeout=0.1)
    with pytest.raises(NarrativeBackendUnavailable):
        backend.generate_json("event", {})
    with pytest.raises(NarrativeBackendUnavailable):
        backend.generate_json("event", {})
    assert calls == 1
    assert backend.available is False


def test_llm_backend_is_used_when_endpoint_is_available(monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return (
                b'{"choices":[{"message":{"content":"{\\"name\\":\\"LLM Name\\",'
                b'\\"summary\\":\\"LLM Summary\\"}"}}]}'
            )

    monkeypatch.setattr("memora.generation.narrative.urlopen", lambda *args, **kwargs: Response())
    backend = ChatCompletionsBackend("http://localhost", "test")
    result = backend.generate_json("event", {})
    assert result == {"name": "LLM Name", "summary": "LLM Summary"}
    assert backend.available is True


def test_phash_and_quality(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("RGB", (80, 80), "white").save(first)
    Image.new("RGB", (80, 80), "white").save(second)
    assert hamming_distance(phash(first), phash(second)) == 0
    assert score_photo(str(first))["score"] >= 0


def test_similar_group_uses_time_gate_and_best_quality():
    now = datetime.now()
    first = make_record("first", now, [1, 0])
    second = make_record("second", now + timedelta(seconds=5), [1, 0])
    second.phash = first.phash = "0" * 64
    first.quality = {"score": 0.2}
    second.quality = {"score": 0.9}
    groups = group_similar([first, second])
    assert groups[0].representative_id == "second"
    assert best_shot_by_group([first, second], groups)[0].id == "second"


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


def test_vector_benchmark_has_exact_recall_baseline():
    rng = np.random.default_rng(3)
    matrix = rng.normal(size=(30, 8)).astype(np.float32)
    queries = rng.normal(size=(4, 8)).astype(np.float32)
    results = benchmark_vector_indexes(matrix, queries, backends=("numpy_exact",))
    assert results[0].recall_at_10 == 1.0
    assert results[0].p95_latency_ms >= 0
    assert results[0].memory_mb > 0


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
