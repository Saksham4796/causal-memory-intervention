from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.analysis.make_artifacts import make_all_artifacts
from src.benchmark.generate_causalmembench import generate_examples, write_dataset
from src.benchmark.validate_dataset import validate_file
from src.experiments.run_ablations import run_ablations
from src.experiments.run_experiment import run_experiment


def run_setup() -> None:
    subprocess.run([sys.executable, "scripts/00_setup.py"], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--skip_setup", action="store_true")
    args = parser.parse_args()

    if not args.skip_setup:
        run_setup()

    dataset_path = Path(f"data/generated/causalmembench_{args.size}.jsonl")
    examples = generate_examples(args.size)
    write_dataset(examples, dataset_path)
    report = validate_file(dataset_path, dataset_path.with_suffix(".validation.json"))
    if report["num_errors"]:
        raise SystemExit(f"Dataset validation failed: {report['num_errors']} errors")

    max_examples = args.max_examples if args.max_examples is not None else args.size
    main_run = run_experiment(args.config, dataset_path, max_examples=max_examples, skip_llm_judge=True, deterministic_only=True)
    ablation_run = run_ablations(args.config, dataset_path, max_examples=max_examples)
    ablation_results = Path(ablation_run) / "ablation_results.csv"
    if ablation_results.exists():
        shutil.copyfile(ablation_results, Path(main_run) / "ablation_results.csv")
    make_all_artifacts(main_run)
    print(main_run)


if __name__ == "__main__":
    main()
