from __future__ import annotations

from src.benchmark.generate_causalmembench import TASK_FAMILIES, generate_examples
from src.benchmark.schema import BenchmarkExample
from src.benchmark.validate_dataset import deterministic_checks


def test_generated_examples_validate_all_families():
    examples = generate_examples(len(TASK_FAMILIES), seed=1, task_families=TASK_FAMILIES)
    assert {example.task_family for example in examples} == set(TASK_FAMILIES)
    for example in examples:
        parsed = BenchmarkExample(**example.to_dict())
        assert deterministic_checks(parsed) == []
        assert parsed.memory_bank
