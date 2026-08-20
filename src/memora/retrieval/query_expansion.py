import re
from collections.abc import Iterable
from typing import Literal

import numpy as np

from memora.encoders.clip_encoder import VisionEncoder, normalize

QueryStrategy = Literal["raw_clip", "prompt_ensemble", "query_enhancement"]
QUERY_STRATEGIES: tuple[QueryStrategy, ...] = ("raw_clip", "prompt_ensemble", "query_enhancement")

PROMPT_TEMPLATES: tuple[str, ...] = (
    "a photo of {}",
    "a photograph of {}",
    "an image showing {}",
    "a picture of {}",
    "a photo depicting {}",
)

_EXPANSIONS = {
    "海边": ["a beach", "people at the beach", "a seaside landscape", "ocean and beach"],
    "beach": ["a beach", "a seaside landscape", "ocean and beach"],
    "火锅": ["hotpot dinner with friends", "people eating hotpot at night", "group dining in a restaurant"],
    "hotpot": ["hotpot dinner", "people eating hotpot", "a restaurant meal"],
    "旅行": ["a travel photo", "vacation", "a tourist attraction", "a trip landscape"],
    "travel": ["a travel photo", "vacation", "a tourist attraction", "a trip landscape"],
    "生日": ["a birthday party", "a birthday dinner", "people celebrating"],
    "birthday": ["a birthday party", "a birthday dinner", "people celebrating"],
    "狗": ["a dog", "a puppy", "a pet dog"],
    "dog": ["a dog", "a puppy", "a pet dog"],
    "猫": ["a cat", "a kitten", "a pet cat"],
    "cat": ["a cat", "a kitten", "a pet cat"],
    "小动物": ["a small animal", "a cute pet", "a cat or dog", "a small furry animal"],
    "small animals": ["a small animal", "a cute pet", "a cat or dog"],
    "食物": ["food", "a meal", "a plate of food", "fruit and food"],
    "美食": ["delicious food", "a restaurant meal", "a plate of food"],
    "food": ["food", "a meal", "a plate of food", "fruit and food"],
}


def expand_query(query: str) -> list[str]:
    query = query.strip()
    folded = query.casefold()
    expansions = list(_EXPANSIONS.get(folded, _EXPANSIONS.get(query, [])))
    if not expansions:
        for key, candidates in _EXPANSIONS.items():
            if key.casefold() in folded:
                expansions.extend(candidates)
    if not expansions:
        words = [word for word in re.split(r"\s+", query) if word]
        expansions = [query, f"a photo of {query}"]
        if len(words) > 1:
            expansions.append("a photo showing " + " ".join(words))
    return list(dict.fromkeys([query, *expansions]))


def query_texts(query: str, strategy: QueryStrategy = "query_enhancement") -> list[str]:
    """Build the text prompts used by one retrieval experiment."""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")
    if strategy == "raw_clip":
        return [query]
    if strategy == "query_enhancement":
        return expand_query(query)
    candidates: Iterable[str] = [query]
    return list(dict.fromkeys(template.format(item) for item in candidates for template in PROMPT_TEMPLATES))


def encode_query(encoder: VisionEncoder, query: str, strategy: QueryStrategy = "query_enhancement") -> np.ndarray:
    """Encode and L2-normalize a query using one of the three baselines."""
    texts = query_texts(query, strategy)
    vectors = encoder.encode_texts(texts)
    return normalize(np.mean(vectors, axis=0))
