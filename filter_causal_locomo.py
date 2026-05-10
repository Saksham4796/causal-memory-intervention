#!/usr/bin/env python3
"""
filter_causal_locomo.py

Final deterministic cleanup for Causal-LoCoMo.

What this script does:
1. Loads causal_locomo.json or causal_locomo.jsonl.
2. Cleans synthetic harmful memories.
3. Removes fake timestamps and source IDs from synthetic memories.
4. Cleans overly broad must_not_include terms.
5. Cleans answer_aliases.
6. Checks whether gold memory contains/supports expected_answer.
7. Flags sensitive/health inference examples for manual review.
8. Optionally drops fail examples.
9. Strips past_sessions completely to prevent leakage.
10. Strips raw dialogue turns from provenance, keeping only evidence IDs.
11. Writes final JSONL file for experiments.

Recommended usage:
  python filter_causal_locomo.py \
    --input causal_locomo.json \
    --output causal_locomo_final.jsonl \
    --drop-fail

For main Causal Memory Intervention experiments, feed only:
  current_task + memory_bank
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "from", "by", "is", "are", "was", "were", "be", "been",
    "being", "what", "when", "where", "who", "why", "how", "did",
    "does", "do", "has", "have", "had", "would", "could", "should",
    "about", "which", "their", "her", "his", "they", "them", "she",
    "he", "it", "this", "that", "these", "those", "as", "into", "out",
    "up", "down", "over", "under", "before", "after", "during"
}


OVERLY_BROAD_FORBIDDEN = {
    "2023",
    "2022",
    "2021",
    "2020",
    "2019",
    "specific dates",
    "specific date",
    "specific calendar date",
    "other people's pets",
    "non-basketball aspirations",
    "unrelated information",
    "incorrect information",
    "misleading information",
    "any unrelated information",
    "irrelevant information",
    "unrelated details",
    "wrong answer",
    "wrong information",
}


SENSITIVE_INFERENCE_TERMS = {
    "obesity",
    "depression",
    "anxiety",
    "illness",
    "disease",
    "diagnosis",
    "medical condition",
    "mental illness",
    "health problem",
}


def load_dataset(path: str) -> List[Dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()

    if not text:
        return []

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            if "records" in data and isinstance(data["records"], list):
                return data["records"]
            return [data]

    except json.JSONDecodeError:
        pass

    records = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue

        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

    return records


def write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(obj: Any, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def normalize(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def tokenize(text: Any) -> List[str]:
    return re.findall(r"[a-zA-Z0-9']+", str(text).lower())


def answer_terms(answer: Any) -> List[str]:
    if answer is None:
        return []

    text = str(answer).strip()
    if not text:
        return []

    terms = [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]

    out = []
    if len(text.split()) <= 8:
        out.append(normalize(text))

    out.extend(terms)

    deduped = []
    seen = set()

    for item in out:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def contains_answer(gold_text: str, expected_answer: Any, aliases: List[str]) -> bool:
    gold = normalize(gold_text)

    candidates = []
    if expected_answer is not None:
        candidates.append(str(expected_answer))

    candidates.extend(aliases or [])

    for ans in candidates:
        ans_norm = normalize(ans)
        if ans_norm and ans_norm in gold:
            return True

    terms = answer_terms(expected_answer)
    important = [t for t in terms if len(t) > 2 and t not in STOPWORDS]

    if important and all(t in gold for t in important[: min(3, len(important))]):
        return True

    return False


def clean_synthetic_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    memory = dict(memory)

    is_synthetic = (
        memory.get("synthetic") is True
        or memory.get("type") == "synthetic_adversarial"
        or memory.get("label") == "harmful"
    )

    if is_synthetic:
        memory["synthetic"] = True
        memory["type"] = "synthetic_adversarial"
        memory["label"] = "harmful"
        memory["causal_role"] = "negative"

        memory["timestamp"] = None
        memory["source_session_id"] = None
        memory["source_session_ids"] = []
        memory["source_dia_ids"] = []
        memory["source_candidate_ids"] = []

        memory["derivation"] = {
            "is_derived": True,
            "method": "synthetic_adversarial_generation",
            "uses_gold_answer": True,
        }

    return memory


def clean_answer_aliases(expected_answer: Any, aliases: List[str]) -> List[str]:
    cleaned = []
    seen = set()

    if expected_answer is not None:
        base = str(expected_answer).strip()
        if base:
            cleaned.append(base)
            seen.add(normalize(base))

    for alias in aliases or []:
        alias = str(alias).strip()
        low = normalize(alias)

        if not alias:
            continue

        # Avoid long full-sentence aliases. Keep short phrase-style aliases.
        if len(alias.split()) > 10:
            continue

        if low not in seen:
            cleaned.append(alias)
            seen.add(low)

    return cleaned


def clean_must_not_include(
    items: List[str],
    expected_answer: Any,
    aliases: List[str],
) -> List[str]:
    expected_blob = normalize(" ".join([str(expected_answer)] + [str(a) for a in aliases or []]))

    cleaned = []
    seen = set()

    for item in items or []:
        raw = str(item).strip()
        low = normalize(raw)

        if not raw:
            continue

        if low in OVERLY_BROAD_FORBIDDEN:
            continue

        # Standalone years are usually too broad and may penalize correct date answers.
        if re.fullmatch(r"\d{4}", low):
            continue

        # Abstract instruction-like forbidden strings are not useful.
        if len(raw.split()) > 8:
            continue

        # Do not forbid a term that appears in the expected answer or alias.
        if low and low in expected_blob:
            continue

        if low not in seen:
            cleaned.append(raw)
            seen.add(low)

    return cleaned


def recompute_memory_id_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    memory_bank = record.get("memory_bank", [])

    record["gold_memory_ids"] = [
        m["memory_id"]
        for m in memory_bank
        if m.get("label") == "useful"
    ]

    record["bad_memory_ids"] = [
        m["memory_id"]
        for m in memory_bank
        if m.get("label") in {"irrelevant", "harmful"}
    ]

    record["context_dependent_memory_ids"] = [
        m["memory_id"]
        for m in memory_bank
        if m.get("label") == "context_dependent"
    ]

    return record


def strip_leakage_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove raw conversation fields so the final experimental dataset only tests
    current_task + memory_bank.

    This removes:
      - past_sessions content
      - raw dialogue turns from provenance.evidence_windows

    It keeps:
      - evidence IDs
      - session IDs
      - timestamps
      - date_time metadata
      - turn IDs only
    """

    record = dict(record)

    record["past_sessions"] = []

    provenance = record.get("provenance", {}) or {}
    evidence_windows = provenance.get("evidence_windows", []) or []

    stripped_windows = []

    for window in evidence_windows:
        turns = window.get("turns", []) or []

        stripped_windows.append(
            {
                "evidence_id": window.get("evidence_id"),
                "session_id": window.get("session_id"),
                "timestamp": window.get("timestamp"),
                "date_time": window.get("date_time"),
                "turn_ids": [
                    turn.get("dia_id")
                    for turn in turns
                    if isinstance(turn, dict) and turn.get("dia_id")
                ],
            }
        )

    record["provenance"] = {
        "evidence_windows": stripped_windows,
        "raw_dialogue_removed": True,
        "past_sessions_removed": True,
    }

    return record


def validate_gold_memory(record: Dict[str, Any]) -> Dict[str, Any]:
    scoring = record.get("scoring_criteria", {}) or {}
    expected = scoring.get("expected_answer")
    aliases = scoring.get("answer_aliases", []) or []

    gold_ids = set(record.get("gold_memory_ids", []))

    gold_text = " ".join(
        m.get("content", "")
        for m in record.get("memory_bank", [])
        if m.get("memory_id") in gold_ids
    )

    metadata = dict(record.get("metadata", {}) or {})
    flags = set(metadata.get("quality_flags", []) or [])

    if not contains_answer(gold_text, expected, aliases):
        flags.add("WARNING_gold_memory_may_not_contain_expected_answer")

    metadata["quality_flags"] = sorted(flags)
    record["metadata"] = metadata

    return record


def flag_sensitive_inferences(record: Dict[str, Any]) -> Dict[str, Any]:
    scoring = record.get("scoring_criteria", {}) or {}
    expected = normalize(scoring.get("expected_answer", ""))

    metadata = dict(record.get("metadata", {}) or {})
    flags = set(metadata.get("quality_flags", []) or [])

    if any(term in expected for term in SENSITIVE_INFERENCE_TERMS):
        flags.add("WARNING_sensitive_or_health_inference_check_manually")

    metadata["quality_flags"] = sorted(flags)
    record["metadata"] = metadata

    return record


def update_metadata_counts(record: Dict[str, Any]) -> Dict[str, Any]:
    memory_bank = record.get("memory_bank", [])
    metadata = dict(record.get("metadata", {}) or {})

    metadata["contains_poisoned_memory"] = any(
        m.get("label") == "harmful"
        for m in memory_bank
    )

    metadata["num_gold_memories"] = sum(
        1 for m in memory_bank if m.get("label") == "useful"
    )

    metadata["num_bad_memories"] = sum(
        1 for m in memory_bank if m.get("label") in {"irrelevant", "harmful"}
    )

    metadata["num_harmful_memories"] = sum(
        1 for m in memory_bank if m.get("label") == "harmful"
    )

    record["metadata"] = metadata
    return record


def assign_quality_status(record: Dict[str, Any]) -> Dict[str, Any]:
    metadata = record.get("metadata", {}) or {}
    flags = metadata.get("quality_flags", []) or []

    if any(flag.startswith("WARNING_gold_memory") for flag in flags):
        record["quality_status"] = "fail"
    elif any(flag.startswith("WARNING_") for flag in flags):
        record["quality_status"] = "warning"
    else:
        # Preserve existing pass/warning if no new warning was introduced.
        record["quality_status"] = record.get("quality_status", "pass")

    return record


def clean_record(record: Dict[str, Any]) -> Dict[str, Any]:
    record = dict(record)

    # 1. Clean memory bank.
    cleaned_memory_bank = []
    for memory in record.get("memory_bank", []):
        cleaned_memory_bank.append(clean_synthetic_memory(memory))

    record["memory_bank"] = cleaned_memory_bank

    # 2. Recompute ID lists from memory labels.
    record = recompute_memory_id_fields(record)

    # 3. Clean scoring criteria.
    scoring = dict(record.get("scoring_criteria", {}) or {})
    expected = scoring.get("expected_answer")

    aliases = clean_answer_aliases(expected, scoring.get("answer_aliases", []) or [])
    scoring["answer_aliases"] = aliases

    scoring["must_not_include"] = clean_must_not_include(
        scoring.get("must_not_include", []) or [],
        expected,
        aliases,
    )

    record["scoring_criteria"] = scoring

    # 4. Gold memory support validation.
    record = validate_gold_memory(record)

    # 5. Sensitive inference warning.
    record = flag_sensitive_inferences(record)

    # 6. Metadata counts.
    record = update_metadata_counts(record)

    # 7. Assign final quality status.
    record = assign_quality_status(record)

    # 8. Strip all raw conversation leakage fields.
    record = strip_leakage_fields(record)

    return record


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = {
        "num_records": len(records),
        "quality_status_counts": {},
        "task_family_counts": {},
        "contains_poisoned_memory": 0,
        "num_gold_memories": 0,
        "num_bad_memories": 0,
        "num_harmful_memories": 0,
        "num_records_with_raw_past_sessions": 0,
        "num_records_with_raw_provenance_turns": 0,
    }

    for record in records:
        status = record.get("quality_status", "unknown")
        stats["quality_status_counts"][status] = stats["quality_status_counts"].get(status, 0) + 1

        family = record.get("task_family", "unknown")
        stats["task_family_counts"][family] = stats["task_family_counts"].get(family, 0) + 1

        metadata = record.get("metadata", {}) or {}

        if metadata.get("contains_poisoned_memory"):
            stats["contains_poisoned_memory"] += 1

        stats["num_gold_memories"] += len(record.get("gold_memory_ids", []))
        stats["num_bad_memories"] += len(record.get("bad_memory_ids", []))
        stats["num_harmful_memories"] += metadata.get("num_harmful_memories", 0) or 0

        if record.get("past_sessions"):
            stats["num_records_with_raw_past_sessions"] += 1

        provenance = record.get("provenance", {}) or {}
        for window in provenance.get("evidence_windows", []) or []:
            if window.get("turns"):
                stats["num_records_with_raw_provenance_turns"] += 1
                break

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input causal_locomo.json or causal_locomo.jsonl")
    parser.add_argument("--output", required=True, help="Output cleaned JSONL file")
    parser.add_argument(
        "--drop-fail",
        action="store_true",
        help="Drop records marked as fail after filtering",
    )

    args = parser.parse_args()

    records = load_dataset(args.input)

    cleaned = []
    dropped = 0

    for record in records:
        cleaned_record = clean_record(record)

        if args.drop_fail and cleaned_record.get("quality_status") == "fail":
            dropped += 1
            continue

        cleaned.append(cleaned_record)

    write_jsonl(cleaned, args.output)

    stats = summarize(cleaned)
    stats["input_records"] = len(records)
    stats["dropped_records"] = dropped
    stats["output_records"] = len(cleaned)

    summary_path = str(Path(args.output).with_suffix(".summary.json"))
    write_json(stats, summary_path)

    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote cleaned dataset to: {args.output}")
    print(f"Wrote summary to: {summary_path}")


if __name__ == "__main__":
    main()