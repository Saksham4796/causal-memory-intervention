#!/usr/bin/env python3
"""
Merge filtered Causal-LoCoMo examples with their original chat histories.

The filter step intentionally removes ``past_sessions`` to produce a memory-only
gold file. For experiments that include a full-history baseline, we restore those
sessions from the raw builder output while keeping the filtered memory bank and
scoring fields.

Typical usage:
  causalmem/bin/python merge_causal_locomo_history.py \
    --raw causal_locomo.jsonl \
    --filtered causal_locomo_final.jsonl \
    --output causal_locomo_final_with_history.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object in {path} at line {line_no}")
            records.append(record)

    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def normalize_memory_timestamps(record: dict[str, Any]) -> int:
    """Repair null/invalid timestamps introduced for synthetic memories."""

    max_ts = 0
    for session in record.get("past_sessions") or []:
        ts = _safe_int(session.get("timestamp"))
        if ts is not None:
            max_ts = max(max_ts, ts)

    for memory in record.get("memory_bank") or []:
        ts = _safe_int(memory.get("timestamp"))
        if ts is not None:
            max_ts = max(max_ts, ts)

    fixed = 0
    for idx, memory in enumerate(record.get("memory_bank") or [], start=1):
        ts = _safe_int(memory.get("timestamp"))
        if ts is None:
            memory["timestamp"] = max_ts + idx
            fixed += 1
        else:
            memory["timestamp"] = ts

    return fixed


def sort_record_timelines(record: dict[str, Any]) -> None:
    label_rank = {
        "useful": 0,
        "context_dependent": 1,
        "irrelevant": 2,
        "outdated": 2,
        "harmful": 3,
        "poisoned": 3,
    }

    record["past_sessions"] = sorted(
        record.get("past_sessions") or [],
        key=lambda session: _safe_int(session.get("timestamp")) or 0,
    )
    record["memory_bank"] = sorted(
        record.get("memory_bank") or [],
        key=lambda memory: (
            _safe_int(memory.get("timestamp")) or 999999,
            label_rank.get(memory.get("label"), 9),
            memory.get("memory_id", ""),
        ),
    )


def summarize(records: list[dict[str, Any]], fixed_timestamps: int, missing_history: list[str]) -> dict[str, Any]:
    labels = Counter()
    families = Counter()
    quality = Counter()

    for record in records:
        families[record.get("task_family", "unknown")] += 1
        quality[record.get("quality_status", "unknown")] += 1
        for memory in record.get("memory_bank") or []:
            labels[memory.get("label", "unknown")] += 1

    return {
        "num_records": len(records),
        "task_family_counts": dict(families),
        "quality_status_counts": dict(quality),
        "memory_label_counts": dict(labels),
        "total_past_sessions": sum(len(record.get("past_sessions") or []) for record in records),
        "records_with_past_sessions": sum(bool(record.get("past_sessions")) for record in records),
        "total_memories": sum(len(record.get("memory_bank") or []) for record in records),
        "fixed_memory_timestamps": fixed_timestamps,
        "missing_history_count": len(missing_history),
        "missing_history_example_ids": missing_history,
    }


def merge_histories(raw_path: str | Path, filtered_path: str | Path, strict: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_records = load_jsonl(raw_path)
    filtered_records = load_jsonl(filtered_path)

    raw_by_id = {record["example_id"]: record for record in raw_records}
    merged: list[dict[str, Any]] = []
    missing_history: list[str] = []
    fixed_timestamps = 0

    for record in filtered_records:
        record = dict(record)
        source = raw_by_id.get(record["example_id"])

        if source is None:
            missing_history.append(record["example_id"])
            record["past_sessions"] = []
        else:
            record["past_sessions"] = source.get("past_sessions") or []

        fixed_timestamps += normalize_memory_timestamps(record)
        sort_record_timelines(record)
        merged.append(record)

    if strict and missing_history:
        raise ValueError(f"Missing raw histories for {len(missing_history)} examples: {missing_history[:10]}")

    return merged, summarize(merged, fixed_timestamps, missing_history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restore LOCOMO histories onto filtered Causal-LoCoMo records.")
    parser.add_argument("--raw", required=True, help="Raw builder JSONL with past_sessions, usually causal_locomo.jsonl")
    parser.add_argument("--filtered", required=True, help="Filtered JSONL, usually causal_locomo_final.jsonl")
    parser.add_argument("--output", required=True, help="Merged output JSONL with restored past_sessions")
    parser.add_argument("--summary-output", default=None, help="Optional summary JSON path")
    parser.add_argument("--strict", action="store_true", help="Fail if any filtered record is missing raw history")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records, summary = merge_histories(args.raw, args.filtered, strict=args.strict)
    write_jsonl(records, args.output)

    summary_path = args.summary_output or str(Path(args.output).with_suffix(".summary.json"))
    write_json(summary, summary_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote merged dataset to: {args.output}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()
