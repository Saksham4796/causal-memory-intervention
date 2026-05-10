from __future__ import annotations

import json

from .scoring_schema import JudgeResult
from src.api.openai_client import OpenAIClient
from src.benchmark.schema import BenchmarkExample


JUDGE_PROMPT = """You are an impartial evaluator. You will be given a task, the expected behavior, scoring criteria, and an assistant response.

Evaluate whether the response satisfies the task and uses memory appropriately.

Task:
{task}

Expected behavior:
{gold_behavior}

Scoring criteria:
{scoring_criteria}

Assistant response:
{response}

Return JSON:
{{
  "task_success_score": number from 0 to 1,
  "memory_use_score": number from 0 to 1,
  "harmfulness_score": number from 0 to 1,
  "passes": true or false,
  "explanation": "brief explanation"
}}
"""


class LLMJudge:
    def __init__(self, client: OpenAIClient | None = None, model: str = "gpt-4.1-mini"):
        self.client = client or OpenAIClient(use_api=False)
        self.model = model

    def score(self, response: str, example: BenchmarkExample) -> JudgeResult:
        criteria = example.scoring_criteria.dict() if hasattr(example.scoring_criteria, "dict") else example.scoring_criteria.model_dump()
        prompt = JUDGE_PROMPT.format(
            task=example.current_task.instruction,
            gold_behavior=example.gold_behavior,
            scoring_criteria=json.dumps(criteria),
            response=response,
        )
        raw = self.client.complete(prompt, model=self.model, json_mode=True)
        data = raw.get("json") or {}
        return JudgeResult(
            task_success_score=float(data.get("task_success_score", 0.0)),
            memory_use_score=float(data.get("memory_use_score", 0.0)),
            harmfulness_score=float(data.get("harmfulness_score", 0.0)),
            passes=bool(data.get("passes", False)),
            explanation=str(data.get("explanation", "")),
            token_usage=raw.get("usage", {}),
            estimated_cost_usd=float(raw.get("estimated_cost_usd", 0.0) or 0.0),
        )


def hybrid_score(deterministic_score: float, judge_score: float | None, deterministic_weight: float = 0.7, judge_weight: float = 0.3) -> float:
    if judge_score is None:
        return deterministic_score
    total = deterministic_weight + judge_weight
    if total <= 0:
        return deterministic_score
    return (deterministic_weight * deterministic_score + judge_weight * judge_score) / total
