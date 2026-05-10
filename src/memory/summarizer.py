from __future__ import annotations

from typing import Iterable

from .memory_card import MemoryCard
from src.utils.text_utils import summarize_memories


def summarize_memory_bank(memories: Iterable[MemoryCard]) -> str:
    return summarize_memories(memories)
