from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    index_path: str = os.getenv("MEMORA_INDEX_PATH", "data/index.json")
    encoder: str = os.getenv("MEMORA_ENCODER", "lightweight")
    clip_model: str = os.getenv("MEMORA_CLIP_MODEL", "ViT-B-32")
    clip_pretrained: str = os.getenv("MEMORA_CLIP_PRETRAINED", "laion2b_s34b_b79k")


settings = Settings()

