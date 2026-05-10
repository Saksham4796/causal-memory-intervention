from __future__ import annotations

from .run_experiment import run_experiment


def run_cmi(config_path: str, dataset_path: str, max_examples: int | None = None):
    return run_experiment(config_path, dataset_path, max_examples=max_examples, agents=["cmi"])
