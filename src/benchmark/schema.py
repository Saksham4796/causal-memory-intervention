from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, root_validator, validator


class SerializableModel(BaseModel):
    def to_dict(self) -> dict[str, Any]:
        if hasattr(self, "model_dump"):
            return self.model_dump()
        return self.dict()


class PastSession(SerializableModel):
    session_id: str
    timestamp: int
    content: str


class MemoryItem(SerializableModel):
    memory_id: str
    content: str
    type: str
    label: str
    scope: Optional[str] = None
    timestamp: int
    expected_effect: Optional[str] = None
    source_session_id: Optional[str] = None


class CurrentTask(SerializableModel):
    task_id: str
    instruction: str
    recipient_type: Optional[str] = None
    domain: str


class ScoringCriteria(SerializableModel):
    must_include: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    style: Optional[str] = None
    max_words: Optional[int] = None
    required_steps: list[str] = Field(default_factory=list)
    expected_answer: Optional[str] = None

    class Config:
        extra = "allow"


class BenchmarkExample(SerializableModel):
    example_id: str
    task_family: str
    past_sessions: list[PastSession]
    memory_bank: list[MemoryItem]
    current_task: CurrentTask
    gold_memory_ids: list[str] = Field(default_factory=list)
    bad_memory_ids: list[str] = Field(default_factory=list)
    context_dependent_memory_ids: list[str] = Field(default_factory=list)
    gold_behavior: str
    scoring_criteria: ScoringCriteria
    metadata: dict[str, Any] = Field(default_factory=dict)

    @validator("past_sessions")
    def sessions_ordered(cls, sessions: list[PastSession]) -> list[PastSession]:
        timestamps = [session.timestamp for session in sessions]
        if timestamps != sorted(timestamps):
            raise ValueError("past session timestamps must be ordered")
        return sessions

    @validator("memory_bank")
    def memory_ids_unique(cls, memories: list[MemoryItem]) -> list[MemoryItem]:
        if not memories:
            raise ValueError("every example must have at least one memory")
        ids = [memory.memory_id for memory in memories]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate memory IDs inside example")
        timestamps = [memory.timestamp for memory in memories]
        if timestamps != sorted(timestamps):
            raise ValueError("memory timestamps must be ordered")
        return memories

    @root_validator(skip_on_failure=True)
    def referenced_memories_exist(cls, values: dict[str, Any]) -> dict[str, Any]:
        memories = values.get("memory_bank") or []
        valid_ids = {memory.memory_id for memory in memories}
        for field_name in ("gold_memory_ids", "bad_memory_ids", "context_dependent_memory_ids"):
            missing = set(values.get(field_name) or []) - valid_ids
            if missing:
                raise ValueError(f"{field_name} contains unknown memory IDs: {sorted(missing)}")
        if not values.get("gold_behavior"):
            raise ValueError("gold_behavior must be non-empty")
        criteria = values.get("scoring_criteria")
        if criteria is None:
            raise ValueError("scoring_criteria must be present")
        metadata = values.get("metadata") or {}
        if metadata.get("contains_poisoned_memory"):
            has_poison = any(
                memory.label in {"poisoned", "harmful"} or memory.type == "poisoned"
                for memory in memories
            )
            if not has_poison:
                raise ValueError("poisoned examples must contain a poisoned or harmful memory")
        if metadata.get("contains_conflict") and len(memories) < 2:
            raise ValueError("conflicting examples must contain at least two memories")
        return values


def validate_example_dict(record: dict[str, Any]) -> BenchmarkExample:
    return BenchmarkExample(**record)
