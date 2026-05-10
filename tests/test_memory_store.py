from __future__ import annotations

from src.memory.memory_card import MemoryCard
from src.memory.memory_store import InMemoryStore, JsonMemoryStore


def make_memory(memory_id: str, timestamp: int, content: str = "The user prefers concise email replies.") -> MemoryCard:
    return MemoryCard(
        memory_id=memory_id,
        content=content,
        memory_type="preference",
        label="useful",
        scope="email",
        timestamp=timestamp,
    )


def test_in_memory_store_retrieve_and_update():
    store = InMemoryStore([make_memory("m1", 1), make_memory("m2", 2, "Vegetarian dinner preference.")])
    assert store.retrieve("concise email", k=1)[0].memory_id == "m1"
    assert store.retrieve_by_recency(k=1)[0].memory_id == "m2"
    store.update_memory_score("m1", 0.4)
    assert store.retrieve_all()[0].causal_utility == 0.4


def test_json_store_persists(tmp_path):
    path = tmp_path / "memories.json"
    store = JsonMemoryStore(path)
    store.add_memory(make_memory("m1", 1))
    loaded = JsonMemoryStore(path)
    assert loaded.retrieve_all()[0].memory_id == "m1"
