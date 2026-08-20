from pathlib import Path
from typing import Any

from memora.clustering.event_cluster import cluster_events
from memora.clustering.journey import JourneyConfig, NamedLocation, discover_journeys, resolve_location_name
from memora.clustering.people import PeopleIndex, load_people_index
from memora.duplicate.similar_group import group_similar
from memora.encoders.clip_encoder import VisionEncoder
from memora.generation.narrative import NarrativeBackend, generate_event_narrative, generate_journey_narrative
from memora.indexer import index_directory
from memora.models import EventGroup, JourneyGroup, PhotoRecord, SearchResult, SimilarGroup
from memora.retrieval.brute_force import search
from memora.retrieval.metadata_filter import GeoBounds, build_search_plan, filter_records
from memora.retrieval.query_expansion import QueryStrategy, encode_query
from memora.retrieval.vector_store import NumpyVectorStore


class MemoraService:
    def __init__(self, encoder: VisionEncoder, index_path: str | Path = "data/index.json") -> None:
        self.encoder = encoder
        self.index_path = Path(index_path)
        self.store = NumpyVectorStore.load(self.index_path)

    @property
    def records(self) -> list[PhotoRecord]:
        return self.store.records

    def index(self, directory: str | Path) -> list[PhotoRecord]:
        existing = {record.id: record for record in self.records}
        self.store.records = index_directory(directory, self.encoder, existing)
        self.store.save(self.index_path)
        return self.records

    def search(
        self,
        query: str,
        top_k: int = 20,
        min_score: float | None = None,
        strategy: QueryStrategy = "query_enhancement",
        captured_from: str | None = None,
        captured_to: str | None = None,
        bounds: GeoBounds | None = None,
        fallback_if_unavailable: bool = False,
        reference_date: str | None = None,
    ) -> list[SearchResult]:
        plan = build_search_plan(
            query,
            captured_from=captured_from,
            captured_to=captured_to,
            bounds=bounds,
            reference_date=reference_date,
        )
        candidates, _ = filter_records(self.records, plan.metadata_filter, fallback_if_unavailable=fallback_if_unavailable)
        query_vector = encode_query(self.encoder, plan.semantic_query, strategy)
        return search(candidates, query_vector, top_k=top_k, min_score=min_score)

    def search_details(self, query: str, top_k: int = 20, min_score: float | None = None, strategy: QueryStrategy = "query_enhancement", *, captured_from: str | None = None, captured_to: str | None = None, bounds: GeoBounds | None = None, fallback_if_unavailable: bool = False, reference_date: str | None = None) -> dict:
        plan = build_search_plan(query, captured_from=captured_from, captured_to=captured_to, bounds=bounds, reference_date=reference_date)
        candidates, fallback = filter_records(self.records, plan.metadata_filter, fallback_if_unavailable=fallback_if_unavailable)
        results = search(candidates, encode_query(self.encoder, plan.semantic_query, strategy), top_k=top_k, min_score=min_score)
        return {
            "plan": plan.to_dict(),
            "total_count": len(self.records),
            "candidate_count": len(candidates),
            "metadata_fallback": fallback,
            "results": [result.__dict__ for result in results],
        }

    def events(
        self,
        *,
        people_path: str | Path | None = None,
        people_index: PeopleIndex | None = None,
        **kwargs: Any,
    ) -> list[EventGroup]:
        people = people_index or (load_people_index(people_path) if people_path else None)
        return cluster_events(self.records, people_index=people, **kwargs)

    def named_events(
        self,
        *,
        people_path: str | Path | None = None,
        people_index: PeopleIndex | None = None,
        strategy: str = "strict_event_people",
        backend: NarrativeBackend | None = None,
        place_names: dict[int, str] | None = None,
        locations: tuple[NamedLocation, ...] = (),
        **kwargs: Any,
    ) -> list[EventGroup]:
        people = people_index or (load_people_index(people_path) if people_path else None)
        events = self.events(strategy=strategy, people_index=people, **kwargs)
        records = {record.id: record for record in self.records}
        resolved_places = {
            event.id: resolve_location_name(event, locations)
            for event in events
            if resolve_location_name(event, locations)
        }
        resolved_places.update(place_names or {})
        return [
            generate_event_narrative(
                event,
                records,
                people,
                self.encoder,
                backend,
                place_name=resolved_places.get(event.id),
            )
            for event in events
        ]

    def journeys(
        self,
        config: JourneyConfig,
        *,
        people_path: str | Path | None = None,
        people_index: PeopleIndex | None = None,
        strategy: str = "strict_event_people",
        backend: NarrativeBackend | None = None,
        generate_narratives: bool = True,
        **kwargs: Any,
    ) -> tuple[list[EventGroup], list[JourneyGroup]]:
        people = people_index or (load_people_index(people_path) if people_path else None)
        locations = (config.home,) + config.destinations
        events = self.named_events(
            people_index=people,
            strategy=strategy,
            backend=backend,
            locations=locations,
            **kwargs,
        )
        journeys = discover_journeys(self.records, events, config)
        if generate_narratives:
            by_id = {event.id: event for event in events}
            journeys = [generate_journey_narrative(journey, by_id, people, backend) for journey in journeys]
        return events, journeys

    def similar_groups(self, **kwargs: float) -> list[SimilarGroup]:
        return group_similar(self.records, **kwargs)
