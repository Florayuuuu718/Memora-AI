from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from memora.clustering.event_cluster import cluster_events
from memora.duplicate.phash import hamming_distance, phash
from memora.encoders.clip_encoder import HashImageEncoder
from memora.evaluation.clustering import pairwise_f1
from memora.evaluation.retrieval import RetrievalCase, evaluate_strategies, recall_at_k
from memora.models import PhotoRecord
from memora.quality.best_shot import score_photo
from memora.retrieval.brute_force import search
from memora.retrieval.query_expansion import expand_query, query_texts


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
