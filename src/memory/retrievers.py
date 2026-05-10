from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Protocol

from .memory_card import MemoryCard
from src.utils.text_utils import bag_of_words_score, cosine_similarity, deterministic_embedding, keyword_overlap


class Retriever(Protocol):
    def retrieve(self, query: str, memories: Iterable[MemoryCard], k: int = 5) -> list[MemoryCard]:
        ...


class RandomRetriever:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def retrieve(self, query: str, memories: Iterable[MemoryCard], k: int = 5) -> list[MemoryCard]:
        items = list(memories)
        self.rng.shuffle(items)
        return items[:k]


class RecencyRetriever:
    def retrieve(self, query: str, memories: Iterable[MemoryCard], k: int = 5) -> list[MemoryCard]:
        return sorted(memories, key=lambda memory: memory.timestamp, reverse=True)[:k]


class KeywordRetriever:
    def retrieve(self, query: str, memories: Iterable[MemoryCard], k: int = 5) -> list[MemoryCard]:
        return sorted(memories, key=lambda memory: bag_of_words_score(query, memory.content), reverse=True)[:k]


class EmbeddingRetriever:
    def __init__(self, client=None, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def _embedding(self, text: str) -> list[float]:
        if self.client is not None:
            try:
                return self.client.embed([text], model=self.model)[0]
            except Exception:
                return deterministic_embedding(text)
        return deterministic_embedding(text)

    def retrieve(self, query: str, memories: Iterable[MemoryCard], k: int = 5) -> list[MemoryCard]:
        query_embedding = self._embedding(query)
        scored = []
        for memory in memories:
            embedding = memory.embedding or self._embedding(memory.content)
            scored.append((cosine_similarity(query_embedding, embedding), memory))
        return [memory for _, memory in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]


@dataclass
class HybridRetriever:
    alpha: float = 0.7
    beta: float = 0.2
    gamma: float = 0.1
    client: object | None = None
    model: str = "text-embedding-3-small"

    def __post_init__(self) -> None:
        self.embedding_retriever = EmbeddingRetriever(self.client, self.model)

    def retrieve(self, query: str, memories: Iterable[MemoryCard], k: int = 5) -> list[MemoryCard]:
        items = list(memories)
        if not items:
            return []
        query_embedding = self.embedding_retriever._embedding(query)
        max_timestamp = max(memory.timestamp for memory in items) or 1
        min_timestamp = min(memory.timestamp for memory in items)
        span = max(1, max_timestamp - min_timestamp)
        scored = []
        for memory in items:
            embedding = memory.embedding or self.embedding_retriever._embedding(memory.content)
            emb_score = (cosine_similarity(query_embedding, embedding) + 1.0) / 2.0
            recency = (memory.timestamp - min_timestamp) / span
            lexical = keyword_overlap(query, memory.content)
            score = self.alpha * emb_score + self.beta * recency + self.gamma * lexical
            scored.append((score, memory))
        return [memory for _, memory in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]


def make_retriever(name: str, **kwargs) -> Retriever:
    name = name.lower()
    if name == "random":
        return RandomRetriever(kwargs.get("seed", 42))
    if name == "recency":
        return RecencyRetriever()
    if name in {"keyword", "bm25"}:
        return KeywordRetriever()
    if name in {"embedding", "vector"}:
        return EmbeddingRetriever(kwargs.get("client"), kwargs.get("model", "text-embedding-3-small"))
    if name == "hybrid":
        return HybridRetriever(
            alpha=kwargs.get("alpha", 0.7),
            beta=kwargs.get("beta", 0.2),
            gamma=kwargs.get("gamma", 0.1),
            client=kwargs.get("client"),
            model=kwargs.get("model", "text-embedding-3-small"),
        )
    raise ValueError(f"Unknown retriever: {name}")
