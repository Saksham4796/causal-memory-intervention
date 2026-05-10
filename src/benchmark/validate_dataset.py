from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import BenchmarkExample
from src.utils.io import read_jsonl, write_json


def validate_records(records: list[dict[str, Any]]) -> tuple[list[BenchmarkExample], list[dict[str, Any]]]:
    valid: list[BenchmarkExample] = []
    errors: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        try:
            example = BenchmarkExample(**record)
            extra_errors = deterministic_checks(example)
            if extra_errors:
                errors.append({"index": idx, "example_id": example.example_id, "errors": extra_errors})
            else:
                valid.append(example)
        except Exception as exc:  # noqa: BLE001
            errors.append({"index": idx, "example_id": record.get("example_id"), "errors": [str(exc)]})
    return valid, errors


def deterministic_checks(example: BenchmarkExample) -> list[str]:
    errors: list[str] = []
    ids = {memory.memory_id for memory in example.memory_bank}
    if not example.gold_behavior.strip():
        errors.append("gold behavior is empty")
    if not (example.scoring_criteria.must_include or example.scoring_criteria.must_not_include or example.scoring_criteria.style or example.scoring_criteria.required_steps):
        errors.append("scoring criteria are empty")
    if set(example.gold_memory_ids) - ids:
        errors.append("gold memory IDs contain unknown IDs")
    if set(example.bad_memory_ids) - ids:
        errors.append("bad memory IDs contain unknown IDs")
    if example.metadata.get("contains_poisoned_memory"):
        if not any(memory.label in {"poisoned", "harmful"} or memory.type == "poisoned" for memory in example.memory_bank):
            errors.append("metadata says poisoned but no poisoned/harmful memory exists")
    if example.metadata.get("contains_conflict"):
        if len(example.memory_bank) < 2:
            errors.append("metadata says conflict but fewer than two memories exist")
    if len(ids) != len(example.memory_bank):
        errors.append("duplicate memory IDs")
    return errors


def validate_file(path: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    records = read_jsonl(path)
    valid, errors = validate_records(records)
    report = {
        "input": str(path),
        "num_records": len(records),
        "num_valid": len(valid),
        "num_errors": len(errors),
        "errors": errors,
        "task_family_counts": dict(Counter(example.task_family for example in valid)),
    }
    if report_path:
        write_json(report, report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a CausalMemBench JSONL file.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    report = validate_file(args.input, args.report)
    print(f"Validated {report['num_valid']}/{report['num_records']} examples")
    if report["num_errors"]:
        print(f"Found {report['num_errors']} errors")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
