from memora.duplicate.phash import hamming_distance
from memora.models import PhotoRecord, SimilarGroup
from memora.retrieval.brute_force import cosine_scores
import numpy as np


def group_similar(records: list[PhotoRecord], *, phash_distance: int = 10, visual_similarity: float = 0.90, time_window_seconds: float = 30.0) -> list[SimilarGroup]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            same_hash = records[left].phash and records[right].phash and hamming_distance(records[left].phash, records[right].phash) <= phash_distance
            same_visual = False
            if records[left].embedding and records[right].embedding:
                same_visual = float(cosine_scores(np.asarray(records[left].embedding), np.asarray([records[right].embedding]))[0]) >= visual_similarity
            close_time = True
            if records[left].timestamp and records[right].timestamp:
                close_time = abs((records[left].timestamp - records[right].timestamp).total_seconds()) <= time_window_seconds
            if same_hash or (same_visual and close_time):
                union(left, right)
    clusters: dict[int, list[PhotoRecord]] = {}
    for index, record in enumerate(records):
        clusters.setdefault(find(index), []).append(record)
    output = []
    group_id = 0
    for group in clusters.values():
        if len(group) < 2:
            continue
        representative = max(group, key=lambda item: item.quality.get("score", 0.0))
        output.append(SimilarGroup(group_id, [item.id for item in group], representative.id))
        group_id += 1
    return output

