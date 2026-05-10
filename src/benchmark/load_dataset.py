from __future__ import annotations

from pathlib import Path

from .schema import BenchmarkExample
from src.utils.io import read_jsonl


def load_examples(path: str | Path, max_examples: int | None = None) -> list[BenchmarkExample]:
    records = read_jsonl(path)
    if max_examples is not None:
        records = records[:max_examples]
    return [BenchmarkExample(**record) for record in records]
