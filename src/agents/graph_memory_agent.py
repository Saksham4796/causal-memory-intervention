from __future__ import annotations

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample
from src.memory.graph_memory import GraphMemory


class GraphMemoryAgent(BaseAgent):
    agent_name = "graph_memory"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        memories = self.memories_from_example(example)
        top_k = int(self.config.get("retrieval", {}).get("top_k", 5))
        retrieved = GraphMemory(memories).retrieve(example.current_task.instruction, k=top_k)
        result = self._answer_with_memories(example, retrieved)
        rejected = [memory for memory in memories if memory.memory_id not in {m.memory_id for m in retrieved}]
        return self._make_output(example, result, selected=retrieved, retrieved=retrieved, rejected=rejected, raw={"prompt": result["prompt"]})
