from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .run_experiment import run_experiment


ABLATION_AGENTS = [
    "cmi_utility_only",
    "cmi_stability_only",
    "cmi_no_perturb",
    "cmi_oracle_scorer",
    "cmi_deterministic_scorer",
    "cmi_batch",
]


def run_ablations(config_path: str, dataset_path: str, max_examples: int | None = None, run_dir: str | Path | None = None) -> Path:
    run_path = run_experiment(
        config_path,
        dataset_path,
        max_examples=max_examples,
        agents=ABLATION_AGENTS,
        run_dir=run_dir,
        skip_llm_judge=True,
        deterministic_only=True,
    )
    metrics = pd.read_csv(Path(run_path) / "metrics_by_agent.csv")
    metrics.to_csv(Path(run_path) / "ablation_results.csv", index=False)
    return Path(run_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--run_dir", default=None)
    args = parser.parse_args()
    print(run_ablations(args.config, args.dataset, args.max_examples, args.run_dir))


if __name__ == "__main__":
    main()
