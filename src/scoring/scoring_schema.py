from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class ScoreResult(BaseModel):
    deterministic_score: float
    llm_judge_score: Optional[float] = None
    final_score: float
    passes: bool
    details: dict = Field(default_factory=dict)


class JudgeResult(BaseModel):
    task_success_score: float
    memory_use_score: float
    harmfulness_score: float
    explanation: str
    passes: bool
    token_usage: dict = Field(default_factory=dict)
    estimated_cost_usd: float = 0.0
