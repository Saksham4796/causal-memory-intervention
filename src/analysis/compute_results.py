from __future__ import annotations

import argparse
from pathlib import Path

from src.scoring.aggregate_metrics import aggregate_metrics
from src.utils.io import read_jsonl


def compute_results(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    aggregate_metrics(predictions, output_dir=run_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    compute_results(args.run_dir)
    print(f"Analyzed {args.run_dir}")


if __name__ == "__main__":
    main()
