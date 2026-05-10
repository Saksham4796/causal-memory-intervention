from __future__ import annotations

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample


class NoMemoryAgent(BaseAgent):
    agent_name = "no_memory"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        result = self._answer_with_memories(example, [], prompt_kind="no_memory")
        return self._make_output(example, result, selected=[], retrieved=[], raw={"prompt": result["prompt"]})
