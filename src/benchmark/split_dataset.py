from __future__ import annotations

import argparse
import random
from pathlib import Path

from .load_dataset import load_examples
from src.utils.io import write_jsonl


def split_examples(examples, train: float = 0.2, dev: float = 0.2, seed: int = 42):
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    train_end = int(n * train)
    dev_end = train_end + int(n * dev)
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    examples = load_examples(args.input)
    splits = split_examples(examples, seed=args.seed)
    output_dir = Path(args.output_dir)
    for name, split in splits.items():
        write_jsonl(split, output_dir / f"{name}.jsonl")
        print(f"Wrote {len(split)} {name} examples")


if __name__ == "__main__":
    main()
