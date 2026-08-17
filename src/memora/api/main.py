from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from memora.config import settings
from memora.encoders.clip_encoder import create_encoder
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
    return {"query": request.query, "strategy": request.strategy, "results": [result.__dict__ for result in service.search(request.query, request.top_k, request.min_score, request.strategy)]}


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
