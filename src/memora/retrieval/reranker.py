from collections.abc import Iterable

from memora.models import SearchResult


def rerank(results: Iterable[SearchResult], *, recency_bonus: float = 0.0) -> list[SearchResult]:
    """Small extension point for metadata-aware ranking."""
    ranked = list(results)
    if recency_bonus:
        ranked.sort(key=lambda item: item.score + recency_bonus * bool(item.captured_at), reverse=True)
    else:
        ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked

