from __future__ import annotations

from typing import Iterable

import networkx as nx

from .memory_card import MemoryCard
from src.utils.text_utils import keyword_overlap, tokenize


class GraphMemory:
    def __init__(self, memories: Iterable[MemoryCard] = ()):
        self.graph = nx.Graph()
        self.memories = list(memories)
        self._build()

    def _build(self) -> None:
        self.graph.clear()
        self.graph.add_node("user", kind="user")
        for memory in self.memories:
            self.graph.add_node(memory.memory_id, kind="memory", memory=memory)
            self.graph.add_edge("user", memory.memory_id, relation="has_memory")
            if memory.scope:
                scope_node = f"scope:{memory.scope.lower()}"
                self.graph.add_node(scope_node, kind="scope")
                self.graph.add_edge(memory.memory_id, scope_node, relation="applies_to")
            for token in tokenize(memory.content):
                if len(token) > 3:
                    concept_node = f"concept:{token}"
                    self.graph.add_node(concept_node, kind="concept")
                    self.graph.add_edge(memory.memory_id, concept_node, relation="related_to")

    def retrieve(self, query: str, k: int = 5) -> list[MemoryCard]:
        query_tokens = set(tokenize(query))
        scored = []
        for memory in self.memories:
            connected = 0
            for neighbor in self.graph.neighbors(memory.memory_id):
                if neighbor.startswith("concept:") and neighbor.split(":", 1)[1] in query_tokens:
                    connected += 1
                if neighbor.startswith("scope:") and keyword_overlap(query, neighbor) > 0:
                    connected += 1
            score = connected + keyword_overlap(query, memory.content)
            scored.append((score, memory))
        return [memory for _, memory in sorted(scored, key=lambda item: item[0], reverse=True)[:k]]
