from __future__ import annotations

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample
from src.memory.retrievers import EmbeddingRetriever


class VectorMemoryAgent(BaseAgent):
    agent_name = "vector_memory"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        memories = self.memories_from_example(example)
        top_k = int(self.config.get("retrieval", {}).get("top_k", 5))
        model = self.openai_config.get("embedding_model", "text-embedding-3-small")
        retrieved = EmbeddingRetriever(client=self.client, model=model).retrieve(example.current_task.instruction, memories, k=top_k)
        result = self._answer_with_memories(example, retrieved)
        rejected = [memory for memory in memories if memory.memory_id not in {m.memory_id for m in retrieved}]
        return self._make_output(example, result, selected=retrieved, retrieved=retrieved, rejected=rejected, raw={"prompt": result["prompt"]})
