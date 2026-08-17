from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memora.clustering.people import (
    apply_feedback,
    cluster_people,
    load_people_index,
    save_people_index,
)
from memora.config import settings
from memora.encoders.clip_encoder import create_encoder
from memora.encoders.face_encoder import InsightFaceEncoder
from memora.retrieval.metadata_filter import GeoBounds
from memora.service import MemoraService


app = FastAPI(title="Memora AI", version="0.1.0", description="Independent photo understanding service")
service = MemoraService(create_encoder(settings.encoder, model_name=settings.clip_model, pretrained=settings.clip_pretrained), settings.index_path)


class IndexRequest(BaseModel):
    directory: str
    index_path: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=20, ge=1, le=200)
    min_score: float | None = None
    strategy: str = Field(default="query_enhancement", pattern="^(raw_clip|prompt_ensemble|query_enhancement)$")
    captured_from: str | None = None
    captured_to: str | None = None
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    fallback_if_unavailable: bool = False
    reference_date: str | None = None


class PeopleClusterRequest(BaseModel):
    people_path: str = "data/people.json"
    model_name: str = "buffalo_l"
    ctx_id: int = Field(default=0, ge=-1, le=0)
    eps: float = Field(default=0.35, gt=0)
    min_samples: int = Field(default=2, ge=2)


class PeopleMergeRequest(BaseModel):
    people_path: str = "data/people.json"
    person_ids: list[int] = Field(min_length=2)


class PeopleRemovePhotoRequest(BaseModel):
    people_path: str = "data/people.json"
    person_id: int
    photo_id: str = Field(min_length=1)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "photos": len(service.records), "encoder": service.encoder.__class__.__name__}


@app.post("/index")
def index(request: IndexRequest) -> dict:
    directory = Path(request.directory)
    if not directory.is_dir():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {directory}")
    if request.index_path:
        service.index_path = Path(request.index_path)
    records = service.index(directory)
    return {"count": len(records), "index_path": str(service.index_path)}


@app.post("/search")
def search(request: SearchRequest) -> dict:
    bounds = GeoBounds(*request.bbox) if request.bbox else None
    response = service.search_details(
        request.query,
        request.top_k,
        request.min_score,
        request.strategy,
        captured_from=request.captured_from,
        captured_to=request.captured_to,
        bounds=bounds,
        fallback_if_unavailable=request.fallback_if_unavailable,
        reference_date=request.reference_date,
    )
    return response


@app.get("/events")
def events() -> dict:
    return {"events": [event.__dict__ for event in service.events()]}


@app.get("/similar-groups")
def similar_groups() -> dict:
    return {"groups": [group.__dict__ for group in service.similar_groups()]}


@app.get("/quality/{photo_id}")
def quality(photo_id: str) -> dict:
    for record in service.records:
        if record.id == photo_id:
            return {"photo_id": photo_id, "quality": record.quality}
    raise HTTPException(status_code=404, detail="Photo not found")


@app.post("/people/cluster")
def cluster_people_endpoint(request: PeopleClusterRequest) -> dict:
    try:
        encoder = InsightFaceEncoder(model_name=request.model_name, ctx_id=request.ctx_id)
        people = cluster_people(
            service.records,
            encoder,
            model_name=request.model_name,
            eps=request.eps,
            min_samples=request.min_samples,
        )
        save_people_index(people, request.people_path)
    except (ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "photo_count": len({face.photo_id for face in people.faces}),
        "face_count": len(people.faces),
        "group_count": len(people.groups),
        "noise_face_count": len(people.noise_face_ids),
        "people_path": request.people_path,
    }


@app.get("/people")
def people(people_path: str = "data/people.json") -> dict:
    return load_people_index(people_path).to_dict()


@app.post("/people/merge")
def merge_people(request: PeopleMergeRequest) -> dict:
    try:
        people_index = load_people_index(request.people_path)
        people_index = apply_feedback(people_index, merges=[request.person_ids])
        save_people_index(people_index, request.people_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return people_index.to_dict()


@app.post("/people/remove-photo")
def remove_person_photo(request: PeopleRemovePhotoRequest) -> dict:
    try:
        people_index = load_people_index(request.people_path)
        people_index = apply_feedback(
            people_index,
            removed_photos=[{"person_id": request.person_id, "photo_id": request.photo_id}],
        )
        save_people_index(people_index, request.people_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return people_index.to_dict()
