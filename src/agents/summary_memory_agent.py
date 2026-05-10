from __future__ import annotations

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample
from src.memory.memory_card import MemoryCard
from src.memory.summarizer import summarize_memory_bank


class SummaryMemoryAgent(BaseAgent):
    agent_name = "summary_memory"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        memories = self.memories_from_example(example)
        summary = summarize_memory_bank(memories)
        summary_memory = MemoryCard(
            memory_id="summary",
            content=summary,
            memory_type="summary",
            label="summary",
            scope="all",
            timestamp=max(memory.timestamp for memory in memories),
        )
        result = self._answer_with_memories(example, [summary_memory])
        return self._make_output(
            example,
            result,
            selected=memories,
            retrieved=memories,
            raw={"prompt": result["prompt"], "summary": summary},
        )
