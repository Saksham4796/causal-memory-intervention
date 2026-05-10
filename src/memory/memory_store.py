from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .memory_card import MemoryCard
from src.utils.io import ensure_dir, read_json, write_json
from src.utils.text_utils import keyword_overlap


class MemoryStore:
    def add_memory(self, memory: MemoryCard) -> None:
        raise NotImplementedError

    def add_memories(self, memories: Iterable[MemoryCard]) -> None:
        for memory in memories:
            self.add_memory(memory)

    def retrieve(self, query: str, k: int = 5) -> list[MemoryCard]:
        raise NotImplementedError

    def retrieve_by_recency(self, k: int = 5) -> list[MemoryCard]:
        raise NotImplementedError

    def retrieve_by_type(self, memory_type: str) -> list[MemoryCard]:
        raise NotImplementedError

    def retrieve_all(self) -> list[MemoryCard]:
        raise NotImplementedError

    def update_memory_score(self, memory_id: str, causal_utility: float) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError


class InMemoryStore(MemoryStore):
    def __init__(self, memories: Iterable[MemoryCard] | None = None):
        self._memories: list[MemoryCard] = list(memories or [])

    def add_memory(self, memory: MemoryCard) -> None:
        self._memories = [m for m in self._memories if m.memory_id != memory.memory_id]
        self._memories.append(memory)

    def retrieve(self, query: str, k: int = 5) -> list[MemoryCard]:
        return sorted(self._memories, key=lambda memory: keyword_overlap(query, memory.content), reverse=True)[:k]

    def retrieve_by_recency(self, k: int = 5) -> list[MemoryCard]:
        return sorted(self._memories, key=lambda memory: memory.timestamp, reverse=True)[:k]

    def retrieve_by_type(self, memory_type: str) -> list[MemoryCard]:
        return [memory for memory in self._memories if memory.memory_type == memory_type]

    def retrieve_all(self) -> list[MemoryCard]:
        return list(self._memories)

    def update_memory_score(self, memory_id: str, causal_utility: float) -> None:
        for memory in self._memories:
            if memory.memory_id == memory_id:
                memory.update_causal_utility(causal_utility)
                return
        raise KeyError(memory_id)

    def clear(self) -> None:
        self._memories.clear()


class JsonMemoryStore(InMemoryStore):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        memories: list[MemoryCard] = []
        if self.path.exists():
            data = read_json(self.path)
            memories = [MemoryCard(**record) for record in data]
        super().__init__(memories)

    def _save(self) -> None:
        ensure_dir(self.path.parent)
        write_json([memory.to_json_dict() for memory in self._memories], self.path)

    def add_memory(self, memory: MemoryCard) -> None:
        super().add_memory(memory)
        self._save()

    def update_memory_score(self, memory_id: str, causal_utility: float) -> None:
        super().update_memory_score(memory_id, causal_utility)
        self._save()

    def clear(self) -> None:
        super().clear()
        self._save()
