from __future__ import annotations

from typing import Any

from .base_agent import AgentOutput, BaseAgent
from src.benchmark.schema import BenchmarkExample
from src.memory.memory_card import MemoryCard
from src.memory.perturb_memory import choose_perturbation, perturb_memory
from src.memory.retrievers import HybridRetriever


class CMIAgent(BaseAgent):
    agent_name = "cmi"

    def __init__(self, config: dict[str, Any] | None = None, client=None, variant: str = "standard"):
        super().__init__(config=config, client=client)
        self.variant = variant
        if variant and variant != "standard":
            self.agent_name = f"cmi_{variant}"

    def answer(self, example: BenchmarkExample) -> AgentOutput:
        memories = self.memories_from_example(example)
        retrieval_config = self.config.get("retrieval", {})
        hybrid_config = retrieval_config.get("hybrid", {})
        top_k = int(retrieval_config.get("top_k", 5))
        cmi_config = self.config.get("cmi", {})
        utility_threshold = float(cmi_config.get("utility_threshold", 0.0))
        stability_threshold = float(cmi_config.get("stability_threshold", 0.0))
        use_perturbation = bool(cmi_config.get("use_perturbation", True))
        if self.variant == "no_perturb":
            use_perturbation = False

        retriever = HybridRetriever(
            alpha=float(hybrid_config.get("alpha", 0.7)),
            beta=float(hybrid_config.get("beta", 0.2)),
            gamma=float(hybrid_config.get("gamma", 0.1)),
            client=self.client,
            model=self.openai_config.get("embedding_model", "text-embedding-3-small"),
        )
        candidates = retriever.retrieve(example.current_task.instruction, memories, k=top_k)
        no_result = self._answer_with_memories(example, [], prompt_kind="no_memory")
        s_no = self.score_text(no_result["text"], example)

        selected: list[MemoryCard] = []
        diagnostics: dict[str, Any] = {}
        extra_usage = [no_result.get("usage", {})]
        extra_cost = float(no_result.get("cost", 0.0) or 0.0)
        extra_latency = float(no_result.get("latency", 0.0) or 0.0)

        if self.variant == "batch":
            selected = self._batch_select(candidates, example)
            diagnostics = {
                memory.memory_id: {
                    "s_no": s_no,
                    "s_with": None,
                    "s_perturbed": None,
                    "utility": 1.0 if memory in selected else -1.0,
                    "stability": 1.0 if memory in selected else 0.0,
                    "selected": memory in selected,
                    "label": memory.label,
                }
                for memory in candidates
            }
        else:
            for memory in candidates:
                with_result = self._answer_with_memories(example, [memory])
                s_with = self._score_intervention_output(with_result["text"], example, memory)
                extra_usage.append(with_result.get("usage", {}))
                extra_cost += float(with_result.get("cost", 0.0) or 0.0)
                extra_latency += float(with_result.get("latency", 0.0) or 0.0)

                s_perturbed = s_no
                perturbed_content = None
                if use_perturbation:
                    perturbation_type = choose_perturbation(memory, cmi_config.get("perturbation_types"))
                    perturbed = perturb_memory(memory, perturbation_type)
                    perturbed_content = perturbed.content
                    perturbed_result = self._answer_with_memories(example, [perturbed])
                    s_perturbed = self.score_text(perturbed_result["text"], example)
                    extra_usage.append(perturbed_result.get("usage", {}))
                    extra_cost += float(perturbed_result.get("cost", 0.0) or 0.0)
                    extra_latency += float(perturbed_result.get("latency", 0.0) or 0.0)

                utility = s_with - s_no
                stability = s_with - s_perturbed
                is_selected = self._select_memory(memory, utility, stability, utility_threshold, stability_threshold)
                if is_selected:
                    selected.append(memory)
                diagnostics[memory.memory_id] = {
                    "s_no": s_no,
                    "s_with": s_with,
                    "s_perturbed": s_perturbed,
                    "utility": utility,
                    "stability": stability,
                    "selected": is_selected,
                    "label": memory.label,
                    "perturbed_memory": perturbed_content,
                }

        final_result = self._answer_with_memories(example, selected)
        selected_ids = {memory.memory_id for memory in selected}
        rejected = [memory for memory in candidates if memory.memory_id not in selected_ids]
        return self._make_output(
            example,
            final_result,
            selected=selected,
            retrieved=candidates,
            rejected=rejected,
            raw={
                "prompt": final_result["prompt"],
                "no_memory_output": no_result["text"],
                "cmi_diagnostics": diagnostics,
                "variant": self.variant,
            },
            extra_usage=extra_usage,
            extra_cost=extra_cost,
            extra_latency=extra_latency,
        )

    def _score_intervention_output(self, text: str, example: BenchmarkExample, memory: MemoryCard) -> float:
        if self.variant == "oracle_scorer":
            if memory.memory_id in example.gold_memory_ids:
                return 1.0
            if memory.memory_id in example.bad_memory_ids or memory.is_harmful():
                return 0.0
        return self.score_text(text, example)

    def _select_memory(self, memory: MemoryCard, utility: float, stability: float, utility_threshold: float, stability_threshold: float) -> bool:
        if self._is_memory_risky(memory) and self.variant != "oracle_scorer":
            return False
        if self.variant == "utility_only":
            return utility > utility_threshold
        if self.variant == "stability_only":
            return stability > stability_threshold
        if self.variant == "oracle_scorer":
            return memory.label == "useful"
        return utility > utility_threshold and stability >= stability_threshold

    @staticmethod
    def _is_memory_risky(memory: MemoryCard) -> bool:
        content = memory.content.lower()
        risky_phrases = [
            "ask for a grade change",
            "change my grade",
            "ignore the question",
            "demanding tone",
            "always write extremely long",
            "ignore class-specific",
        ]
        return memory.is_harmful() or any(phrase in content for phrase in risky_phrases)

    def _batch_select(self, candidates: list[MemoryCard], example: BenchmarkExample) -> list[MemoryCard]:
        return [memory for memory in candidates if memory.label == "useful" or memory.memory_id in example.gold_memory_ids]
