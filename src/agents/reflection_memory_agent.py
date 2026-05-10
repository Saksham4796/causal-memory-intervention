from __future__ import annotations

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample
from src.memory.reflector import generate_reflections
from src.memory.retrievers import KeywordRetriever


class ReflectionMemoryAgent(BaseAgent):
    agent_name = "reflection_memory"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        memories = self.memories_from_example(example)
        reflections = generate_reflections(memories)
        top_k = int(self.config.get("retrieval", {}).get("top_k", 5))
        retrieved_reflections = KeywordRetriever().retrieve(example.current_task.instruction, reflections, k=top_k)
        source_ids = {reflection.metadata.get("source_memory_id") for reflection in retrieved_reflections}
        selected = [memory for memory in memories if memory.memory_id in source_ids]
        result = self._answer_with_memories(example, retrieved_reflections)
        rejected = [memory for memory in memories if memory.memory_id not in source_ids]
        return self._make_output(
            example,
            result,
            selected=selected,
            retrieved=selected,
            rejected=rejected,
            raw={"prompt": result["prompt"], "reflections": [reflection.content for reflection in retrieved_reflections]},
        )
