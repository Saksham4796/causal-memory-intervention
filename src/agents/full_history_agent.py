from __future__ import annotations

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample


class FullHistoryAgent(BaseAgent):
    agent_name = "full_history"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        memories = self.memories_from_example(example)
        result = self._answer_with_memories(example, memories, prompt_kind="full_history")
        return self._make_output(example, result, selected=memories, retrieved=memories, raw={"prompt": result["prompt"]})
