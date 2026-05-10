from __future__ import annotations

from src.memory.memory_card import MemoryCard
from src.memory.retrievers import EmbeddingRetriever, KeywordRetriever, RecencyRetriever


def memories():
    return [
        MemoryCard(memory_id="m1", content="The user prefers concise emails.", memory_type="preference", label="useful", scope="email", timestamp=1),
        MemoryCard(memory_id="m2", content="The user likes vegetarian food.", memory_type="preference", label="irrelevant", scope="food", timestamp=2),
    ]


def test_keyword_retriever():
    result = KeywordRetriever().retrieve("write a concise email", memories(), k=1)
    assert result[0].memory_id == "m1"


def test_recency_retriever():
    result = RecencyRetriever().retrieve("anything", memories(), k=1)
    assert result[0].memory_id == "m2"


def test_embedding_retriever_mocked_embeddings():
    class Client:
        def embed(self, texts, model):
            return [[1.0, 0.0] if "email" in text or "concise" in text else [0.0, 1.0] for text in texts]

    result = EmbeddingRetriever(client=Client()).retrieve("concise email", memories(), k=1)
    assert result[0].memory_id == "m1"
