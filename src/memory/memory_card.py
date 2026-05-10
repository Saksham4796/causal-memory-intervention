from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class MemoryCard(BaseModel):
    memory_id: str
    content: str
    memory_type: str
    label: str
    scope: Optional[str] = None
    timestamp: int
    source_session_id: Optional[str] = None
    expected_effect: Optional[str] = None
    embedding: Optional[list[float]] = None
    causal_utility: Optional[float] = None
    last_verified: Optional[int] = None
    poison_risk: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_benchmark_memory(cls, memory: Any) -> "MemoryCard":
        if isinstance(memory, dict):
            data = memory
            memory_type = data.get("memory_type") or data.get("type", "unknown")
            return cls(
                memory_id=data["memory_id"],
                content=data["content"],
                memory_type=memory_type,
                label=data.get("label", "unknown"),
                scope=data.get("scope"),
                timestamp=data.get("timestamp", 0),
                source_session_id=data.get("source_session_id"),
                expected_effect=data.get("expected_effect"),
                embedding=data.get("embedding"),
                causal_utility=data.get("causal_utility"),
                last_verified=data.get("last_verified"),
                poison_risk=data.get("poison_risk"),
            )
        return cls(
            memory_id=memory.memory_id,
            content=memory.content,
            memory_type=getattr(memory, "memory_type", getattr(memory, "type", "unknown")),
            label=memory.label,
            scope=memory.scope,
            timestamp=memory.timestamp,
            source_session_id=getattr(memory, "source_session_id", None),
            expected_effect=getattr(memory, "expected_effect", None),
        )

    def to_json_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()

    def with_embedding(self, embedding: list[float]) -> "MemoryCard":
        data = self.to_json_dict()
        data["embedding"] = embedding
        return MemoryCard(**data)

    def update_causal_utility(self, value: float) -> None:
        self.causal_utility = value

    def is_newer_than(self, other: "MemoryCard") -> bool:
        return self.timestamp > other.timestamp

    def matches_scope(self, query: str) -> bool:
        if not self.scope:
            return True
        query_l = query.lower()
        return any(token in query_l for token in self.scope.lower().replace("_", " ").split())

    def is_useful(self) -> bool:
        return self.label == "useful"

    def is_harmful(self) -> bool:
        return self.label in {"harmful", "poisoned"} or self.memory_type == "poisoned"

    def is_irrelevant(self) -> bool:
        return self.label in {"irrelevant", "outdated"}
