from collections.abc import Iterable
from dataclasses import dataclass

from memora.encoders.clip_encoder import VisionEncoder
from memora.models import PhotoRecord, SearchResult
from memora.retrieval.brute_force import search
from memora.retrieval.query_expansion import QUERY_STRATEGIES, QueryStrategy, encode_query


def recall_at_k(results: Iterable[SearchResult], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = {item.photo_id for item in list(results)[:k]}
    return len(retrieved & relevant_ids) / len(relevant_ids)


def mean_recall_at_k(cases: Iterable[tuple[Iterable[SearchResult], set[str]]], k: int) -> float:
    values = [recall_at_k(results, relevant, k) for results, relevant in cases]
    return sum(values) / len(values) if values else 0.0


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: frozenset[str]


def evaluate_strategies(
    records: list[PhotoRecord],
    encoder: VisionEncoder,
    cases: Iterable[RetrievalCase],
    ks: Iterable[int] = (1, 5, 10),
    strategies: Iterable[QueryStrategy] = QUERY_STRATEGIES,
) -> dict[str, dict[str, float]]:
    """Compare raw CLIP, prompt ensemble and query enhancement."""
    cases = list(cases)
    ks = tuple(sorted(set(ks)))
    top_k = max(ks, default=10)
    output: dict[str, dict[str, float]] = {}
    for strategy in strategies:
        ranked = {
            case.query: search(records, encode_query(encoder, case.query, strategy), top_k=top_k)
            for case in cases
        }
        values: dict[str, float] = {}
        for k in ks:
            scores = []
            for case in cases:
                scores.append(recall_at_k(ranked[case.query], set(case.relevant_ids), k))
            values[f"recall@{k}"] = sum(scores) / len(scores) if scores else 0.0
        output[strategy] = values
    return output
