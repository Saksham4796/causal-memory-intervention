from __future__ import annotations

import time
from typing import Any, Iterable

from pydantic import BaseModel, Field

from src.api.openai_client import OpenAIClient
from src.benchmark.schema import BenchmarkExample
from src.memory.memory_card import MemoryCard
from src.scoring.deterministic_scorers import score_response
from src.utils.text_utils import format_memories, generate_local_answer


class AgentOutput(BaseModel):
    example_id: str
    agent_name: str
    final_answer: str
    selected_memory_ids: list[str] = Field(default_factory=list)
    retrieved_memory_ids: list[str] = Field(default_factory=list)
    rejected_memory_ids: list[str] = Field(default_factory=list)
    raw_model_outputs: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)
    latency_seconds: float = 0.0
    estimated_cost_usd: float = 0.0

    def to_prediction(self, task_family: str, scores: dict[str, Any]) -> dict[str, Any]:
        data = self.model_dump() if hasattr(self, "model_dump") else self.dict()
        data["task_family"] = task_family
        data["scores"] = scores
        data["response"] = data["final_answer"]
        data["cost_usd"] = data["estimated_cost_usd"]
        return data


class BaseAgent:
    agent_name = "base"

    def __init__(self, config: dict[str, Any] | None = None, client: OpenAIClient | None = None):
        self.config = config or {}
        self.client = client or OpenAIClient(use_api=False)
        self.openai_config = self.config.get("openai", {})
        self.experiment_config = self.config.get("experiment", {})
        self.model = self.openai_config.get("agent_model", "gpt-4.1-mini")
        self.temperature = float(self.openai_config.get("temperature", 0.0))
        self.max_output_tokens = int(self.openai_config.get("max_output_tokens", 600))
        self.deterministic_only = bool(self.experiment_config.get("deterministic_only", True))

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        raise NotImplementedError

    def memories_from_example(self, example: BenchmarkExample) -> list[MemoryCard]:
        return [MemoryCard.from_benchmark_memory(memory) for memory in example.memory_bank]

    def _answer_with_memories(self, example: BenchmarkExample, memories: Iterable[MemoryCard], prompt_kind: str = "agent") -> dict[str, Any]:
        memories = list(memories)
        prompt = self._render_prompt(example, memories, prompt_kind=prompt_kind)
        started = time.time()
        if self.deterministic_only or not self.client.use_api:
            text = generate_local_answer(example.current_task, memories)
            latency = time.time() - started
            usage = {
                "input_tokens": max(1, len(prompt.split())),
                "output_tokens": max(1, len(text.split())),
                "total_tokens": max(1, len(prompt.split()) + len(text.split())),
            }
            return {"text": text, "prompt": prompt, "usage": usage, "cost": 0.0, "latency": latency}
        result = self.client.complete(
            prompt,
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_output_tokens,
        )
        return {
            "text": result.get("text", ""),
            "prompt": prompt,
            "usage": result.get("usage", {}),
            "cost": result.get("estimated_cost_usd", 0.0),
            "latency": result.get("latency_seconds", time.time() - started),
        }

    def _render_prompt(self, example: BenchmarkExample, memories: Iterable[MemoryCard], prompt_kind: str = "agent") -> str:
        task = example.current_task.instruction
        if prompt_kind == "no_memory":
            return f"You are a careful assistant. Complete the current task.\n\nCurrent task:\n{task}\n\nReturn only the final response."
        if prompt_kind == "full_history":
            sessions = "\n".join(f"- {session.timestamp}: {session.content}" for session in example.past_sessions)
            return f"You are an assistant with access to the user's past sessions.\n\nPast sessions:\n{sessions}\n\nCurrent task:\n{task}"
        return (
            "You are a careful assistant. Complete the current task using only memories that are relevant and reliable.\n\n"
            f"Retrieved memories:\n{format_memories(memories)}\n\n"
            f"Current task:\n{task}\n\nReturn only the final response."
        )

    @staticmethod
    def _sum_usage(*usages: dict[str, int]) -> dict[str, int]:
        keys = {"input_tokens", "output_tokens", "total_tokens"}
        return {key: sum(int(usage.get(key, 0) or 0) for usage in usages) for key in keys}

    def _make_output(
        self,
        example: BenchmarkExample,
        answer_result: dict[str, Any],
        selected: list[MemoryCard],
        retrieved: list[MemoryCard] | None = None,
        rejected: list[MemoryCard] | None = None,
        raw: dict[str, Any] | None = None,
        extra_usage: list[dict[str, int]] | None = None,
        extra_cost: float = 0.0,
        extra_latency: float = 0.0,
    ) -> AgentOutput:
        usage = self._sum_usage(answer_result.get("usage", {}), *(extra_usage or []))
        return AgentOutput(
            example_id=example.example_id,
            agent_name=self.agent_name,
            final_answer=answer_result.get("text", ""),
            selected_memory_ids=[memory.memory_id for memory in selected],
            retrieved_memory_ids=[memory.memory_id for memory in (retrieved or selected)],
            rejected_memory_ids=[memory.memory_id for memory in (rejected or [])],
            raw_model_outputs=raw or {"prompt": answer_result.get("prompt", "")},
            token_usage=usage,
            latency_seconds=float(answer_result.get("latency", 0.0) or 0.0) + extra_latency,
            estimated_cost_usd=float(answer_result.get("cost", 0.0) or 0.0) + extra_cost,
        )

    def score_text(self, text: str, example: BenchmarkExample) -> float:
        return score_response(text, example).final_score
