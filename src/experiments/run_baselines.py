from __future__ import annotations

from .run_experiment import run_experiment


BASELINE_AGENTS = ["no_memory", "full_history", "vector_memory", "summary_memory", "reflection_memory", "graph_memory"]


def run_baselines(config_path: str, dataset_path: str, max_examples: int | None = None):
    return run_experiment(config_path, dataset_path, max_examples=max_examples, agents=BASELINE_AGENTS)
