import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    index_path: str = os.getenv("MEMORA_INDEX_PATH", "data/index.json")
    encoder: str = os.getenv("MEMORA_ENCODER", "lightweight")
    clip_model: str = os.getenv("MEMORA_CLIP_MODEL", "ViT-B-32")
    clip_pretrained: str = os.getenv("MEMORA_CLIP_PRETRAINED", "laion2b_s34b_b79k")
    llm_url: str | None = os.getenv("MEMORA_LLM_URL")
    llm_model: str | None = os.getenv("MEMORA_LLM_MODEL")
    llm_api_key: str | None = os.getenv("MEMORA_LLM_API_KEY")
    immich_url: str | None = os.getenv("MEMORA_IMMICH_URL")
    immich_api_key: str | None = os.getenv("MEMORA_IMMICH_API_KEY")
    immich_cache_path: str = os.getenv("MEMORA_IMMICH_CACHE_PATH", "data/immich-cache")
    immich_timeout_seconds: float = float(os.getenv("MEMORA_IMMICH_TIMEOUT_SECONDS", "30"))
    projects_path: str = os.getenv("MEMORA_PROJECTS_PATH", "data/projects")


settings = Settings()
