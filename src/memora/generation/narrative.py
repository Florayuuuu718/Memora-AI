from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol
from urllib.request import Request, urlopen

import numpy as np

from memora.clustering.people import PeopleIndex
from memora.encoders.clip_encoder import VisionEncoder
from memora.models import EventGroup, JourneyGroup, PhotoRecord

ACTIVITY_PROMPTS: dict[str, str] = {
    "海边游玩": "people spending time at a beach by the ocean",
    "聚餐": "people having a meal together at a restaurant",
    "徒步": "people hiking outdoors on a trail",
    "城市漫步": "people walking through city streets",
    "公园游览": "people visiting a park or garden",
    "博物馆参观": "people visiting a museum or exhibition",
    "看日落": "watching a sunset outdoors",
    "生日聚会": "a birthday party with a cake",
    "雪景游览": "people visiting snowy mountains or a winter landscape",
    "交通途中": "a travel scene in a train car airport or vehicle",
}

OBJECT_PROMPTS: dict[str, str] = {
    "蛋糕": "a cake in a photo",
    "餐桌": "a dining table with food",
    "海洋": "the ocean or sea",
    "雪山": "a snowy mountain",
    "汽车": "a car or road vehicle",
    "火车": "a train or railway carriage",
    "鲜花": "flowers or a bouquet",
    "历史建筑": "historic architecture or an old building",
    "宠物": "a pet dog or cat",
}

_TEXT_VECTOR_CACHE: dict[tuple[int, tuple[str, ...]], np.ndarray] = {}


class NarrativeBackend(Protocol):
    def generate_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]: ...


class NarrativeBackendUnavailable(OSError):
    """Raised after a configured LLM endpoint fails its first request."""


class ChatCompletionsBackend:
    """OpenAI-compatible backend with a one-failure circuit breaker."""

    def __init__(self, url: str, model: str, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.available = True
        self.last_error: str | None = None

    def generate_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if not self.available:
            raise NarrativeBackendUnavailable(self.last_error or "LLM endpoint is unavailable")
        system = (
            "You organize a private personal photo library by naming people profiles, events, and journeys. "
            "Use only facts in the JSON payload. Never invent a person's real identity, relationships, exact places, "
            "or activities that are absent. Match the language used by existing names when possible. Return JSON only."
        )
        body = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps({"task": task, "facts": payload}, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2,
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.url, data=body, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            content = value["choices"][0]["message"]["content"]
            generated = json.loads(content)
            if not isinstance(generated, Mapping):
                raise TypeError("LLM response content must be a JSON object")
            required = {
                "event_name_and_summary": ("name", "summary"),
                "journey_name_and_note": ("name", "note"),
                "person_profile": ("name", "note"),
            }.get(task, ())
            if any(not generated.get(field) for field in required):
                raise ValueError(f"LLM response is missing required fields: {required}")
            return dict(generated)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            self.available = False
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise NarrativeBackendUnavailable(self.last_error) from exc


def _person_names(people: PeopleIndex | None) -> dict[int, str]:
    if people is None:
        return {}
    return {group.id: group.name for group in people.groups if group.name}


def _infer_tags(
    event: EventGroup,
    records: Mapping[str, PhotoRecord],
    encoder: VisionEncoder | None,
    prompts: Mapping[str, str],
    *,
    top_n: int = 2,
    min_score: float = 0.18,
) -> list[str]:
    if encoder is None:
        return []
    vectors = [records[photo_id].embedding for photo_id in event.photo_ids if photo_id in records and records[photo_id].embedding]
    if not vectors:
        return []
    centroid = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    labels = list(prompts)
    cache_key = (id(encoder), tuple(prompts[label] for label in labels))
    text_vectors = _TEXT_VECTOR_CACHE.get(cache_key)
    if text_vectors is None:
        text_vectors = encoder.encode_texts(prompts[label] for label in labels)
        _TEXT_VECTOR_CACHE[cache_key] = text_vectors
    if text_vectors.ndim != 2 or text_vectors.shape[1] != centroid.shape[0]:
        return []
    scores = text_vectors @ centroid
    order = np.argsort(-scores)
    return [labels[int(index)] for index in order[:top_n] if float(scores[index]) >= min_score]


def infer_activity_tags(
    event: EventGroup,
    records: Mapping[str, PhotoRecord],
    encoder: VisionEncoder | None,
    *,
    top_n: int = 2,
    min_score: float = 0.18,
) -> list[str]:
    return list(event.activity_tags) or _infer_tags(
        event,
        records,
        encoder,
        ACTIVITY_PROMPTS,
        top_n=top_n,
        min_score=min_score,
    )


def infer_object_tags(
    event: EventGroup,
    records: Mapping[str, PhotoRecord],
    encoder: VisionEncoder | None,
    *,
    top_n: int = 3,
    min_score: float = 0.18,
) -> list[str]:
    return list(event.object_tags) or _infer_tags(
        event,
        records,
        encoder,
        OBJECT_PROMPTS,
        top_n=top_n,
        min_score=min_score,
    )


def _duration_minutes(event: EventGroup) -> float | None:
    if event.duration_minutes is not None:
        return event.duration_minutes
    if not event.start or not event.end:
        return None
    try:
        start = datetime.fromisoformat(event.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(event.end.replace("Z", "+00:00"))
        return max(0.0, (end.timestamp() - start.timestamp()) / 60.0)
    except ValueError:
        return None


def build_event_facts(
    event: EventGroup,
    records: Mapping[str, PhotoRecord],
    people: PeopleIndex | None = None,
    encoder: VisionEncoder | None = None,
    *,
    place_name: str | None = None,
) -> dict[str, Any]:
    names = _person_names(people)
    tags = infer_activity_tags(event, records, encoder)
    objects = infer_object_tags(event, records, encoder)
    return {
        "event_id": event.id,
        "start": event.start,
        "end": event.end,
        "duration_minutes": _duration_minutes(event),
        "place": place_name,
        "people": [names[person_id] for person_id in event.person_ids if person_id in names],
        "unnamed_people_count": sum(person_id not in names for person_id in event.person_ids),
        "activity_candidates": tags,
        "object_candidates": objects,
        "photo_count": len(event.photo_ids),
        "evidence": event.evidence,
    }


def _people_phrase(facts: Mapping[str, Any]) -> str | None:
    names = list(facts.get("people") or [])
    unnamed = int(facts.get("unnamed_people_count") or 0)
    if names:
        return f"和{'、'.join(names[:2])}" if len(names) <= 2 else "和朋友们"
    if unnamed == 1:
        return "和一位朋友"
    if unnamed > 1:
        return "和朋友们"
    return None


def template_event_narrative(facts: Mapping[str, Any]) -> dict[str, str]:
    place = facts.get("place")
    activities = list(facts.get("activity_candidates") or [])
    activity = activities[0] if activities else None
    objects = list(facts.get("object_candidates") or [])
    object_name = objects[0] if objects else None
    people = _people_phrase(facts)
    date = str(facts.get("start") or "")[:10]
    if place and activity:
        name = f"{place}{activity}"
    elif activity:
        name = f"{date}·{activity}" if date else activity
    elif place:
        name = f"{place}记忆"
    elif people:
        name = f"{date}·与朋友的时光" if date else "与朋友的时光"
    elif object_name:
        name = f"{date}·{object_name}记录" if date else f"{object_name}记录"
    elif date:
        name = f"{date}·照片记忆"
    else:
        name = "一段未命名的记忆"
    summary_parts = []
    if date:
        summary_parts.append(date)
    if place:
        summary_parts.append(f"地点为{place}")
    if people:
        summary_parts.append(people)
    if activity:
        summary_parts.append(f"可能记录了{activity}")
    if objects:
        summary_parts.append(f"画面中可能包含{'、'.join(objects[:2])}")
    summary = "，".join(summary_parts) + "。" if summary_parts else "这组照片暂时缺少足够的事件信息。"
    return {"name": name, "summary": summary}


def generate_event_narrative(
    event: EventGroup,
    records: Mapping[str, PhotoRecord],
    people: PeopleIndex | None = None,
    encoder: VisionEncoder | None = None,
    backend: NarrativeBackend | None = None,
    *,
    place_name: str | None = None,
) -> EventGroup:
    facts = build_event_facts(event, records, people, encoder, place_name=place_name)
    fallback = template_event_narrative(facts)
    source = "template"
    value = fallback
    if backend is not None:
        try:
            generated = backend.generate_json("event_name_and_summary", facts)
            if generated.get("name") and generated.get("summary"):
                value = {"name": str(generated["name"]), "summary": str(generated["summary"])}
                source = "llm"
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            value = fallback
    return replace(
        event,
        activity_tags=list(facts["activity_candidates"]),
        object_tags=list(facts["object_candidates"]),
        duration_minutes=facts["duration_minutes"],
        name=value["name"],
        summary=value["summary"],
        name_source=source,
    )


def build_journey_facts(
    journey: JourneyGroup,
    events: Mapping[int, EventGroup],
    people: PeopleIndex | None = None,
) -> dict[str, Any]:
    names = _person_names(people)
    selected = [events[event_id] for event_id in journey.event_ids if event_id in events]
    activities = []
    for event in selected:
        for tag in event.activity_tags:
            if tag not in activities:
                activities.append(tag)
    return {
        "journey_id": journey.id,
        "start": journey.start,
        "end": journey.end,
        "home": journey.home_name,
        "destinations": journey.destination_names,
        "companions": [names[person_id] for person_id in journey.person_ids if person_id in names],
        "unnamed_companion_count": sum(person_id not in names for person_id in journey.person_ids),
        "events": [{"name": event.name, "summary": event.summary} for event in selected],
        "activities": activities,
        "loose_photo_count": len(journey.loose_photo_ids),
    }


def template_journey_narrative(facts: Mapping[str, Any]) -> dict[str, str]:
    destinations = list(facts.get("destinations") or [])
    start = str(facts.get("start") or "")[:10]
    activities = "、".join(list(facts.get("activities") or [])[:4])
    if len(destinations) == 1:
        name = f"{destinations[0]}之旅"
    elif len(destinations) > 1:
        name = f"{'与'.join(destinations[:2])}之旅"
    elif start and activities:
        name = f"{start[:7]}·{activities.split('、')[0]}之旅"
    elif start:
        name = f"{start[:7]}的旅行"
    else:
        name = "一段旅行"
    companions = list(facts.get("companions") or [])
    unnamed = int(facts.get("unnamed_companion_count") or 0)
    who = f"和{'、'.join(companions[:2])}" if companions else ("和朋友们" if unnamed else "")
    places = "、".join(destinations) if destinations else "外地"
    event_names = [
        str(event["name"])
        for event in facts.get("events") or []
        if isinstance(event, Mapping) and event.get("name")
    ]
    note = f"{start + '，' if start else ''}{who + '一起' if who else ''}去了{places}。"
    if activities:
        note += f"旅途中记录了{activities}。"
    elif event_names:
        note += f"主要记录包括{'、'.join(event_names[:3])}。"
    if facts.get("loose_photo_count"):
        note += f"另外保留了{facts['loose_photo_count']}张旅途中的零散照片。"
    return {"name": name, "note": note}


def generate_journey_narrative(
    journey: JourneyGroup,
    events: Mapping[int, EventGroup],
    people: PeopleIndex | None = None,
    backend: NarrativeBackend | None = None,
) -> JourneyGroup:
    facts = build_journey_facts(journey, events, people)
    fallback = template_journey_narrative(facts)
    source = "template"
    value = fallback
    if backend is not None:
        try:
            generated = backend.generate_json("journey_name_and_note", facts)
            if generated.get("name") and generated.get("note"):
                value = {"name": str(generated["name"]), "note": str(generated["note"])}
                source = "llm"
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            value = fallback
    return replace(
        journey,
        activity_tags=list(facts["activities"]),
        name=value["name"],
        note=value["note"],
        name_source=source,
    )
