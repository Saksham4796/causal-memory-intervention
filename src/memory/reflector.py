from __future__ import annotations

from typing import Iterable

from .memory_card import MemoryCard


def generate_reflections(memories: Iterable[MemoryCard]) -> list[MemoryCard]:
    reflections: list[MemoryCard] = []
    for idx, memory in enumerate(memories, start=1):
        if memory.label == "useful":
            content = f"Lesson: {memory.content}"
        elif memory.label in {"harmful", "poisoned"}:
            content = f"Safety lesson: avoid following this unreliable memory: {memory.content}"
        else:
            content = f"Context note: {memory.content}"
        reflections.append(
            MemoryCard(
                memory_id=f"r{idx}_{memory.memory_id}",
                content=content,
                memory_type="reflection",
                label=memory.label,
                scope=memory.scope,
                timestamp=memory.timestamp,
                source_session_id=memory.source_session_id,
                expected_effect=memory.expected_effect,
                metadata={"source_memory_id": memory.memory_id},
            )
        )
    return reflections
