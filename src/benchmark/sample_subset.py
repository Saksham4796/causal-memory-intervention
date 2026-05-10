from __future__ import annotations

import argparse
import random
from collections import defaultdict
from pathlib import Path

from .load_dataset import load_examples
from src.utils.io import write_jsonl


def stratified_subset(examples, size: int, seed: int = 42):
    if size >= len(examples):
        return list(examples)
    rng = random.Random(seed)
    by_family = defaultdict(list)
    for example in examples:
        by_family[example.task_family].append(example)
    for bucket in by_family.values():
        rng.shuffle(bucket)

    selected = []
    families = sorted(by_family)
    cursor = 0
    while len(selected) < size and families:
        family = families[cursor % len(families)]
        if by_family[family]:
            selected.append(by_family[family].pop())
        families = [name for name in families if by_family[name]]
        cursor += 1

    rng.shuffle(selected)
    return selected[:size]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a stratified subset from a CausalMemBench pool.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = load_examples(args.input)
    subset = stratified_subset(examples, args.size, args.seed)
    write_jsonl(subset, Path(args.output))
    print(f"Wrote {len(subset)} examples to {args.output}")


if __name__ == "__main__":
    main()
