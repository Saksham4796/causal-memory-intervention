#!/usr/bin/env python3
"""
build_causal_locomo_llm_v3.py

LLM-assisted builder for a cleaner, paper-oriented Causal-LoCoMo dataset.

This version improves the earlier builder by:
  1. Reading OPENAI_API_KEY and OPENAI_API_URL from .env.
  2. Using LoCoMo evidence windows, session dates, observations, and QA answers.
  3. Prompting the model to create answer-sufficient useful memories.
  4. Making temporal useful memories self-contained without unsafe over-resolution.
  5. Preventing future-session leakage when --no-future-leakage is used.
  6. Grounding irrelevant memories in LoCoMo candidate memories by default.
  7. Marking harmful memories as synthetic_adversarial.
  8. Removing duplicate useful/context-dependent/irrelevant memories.
  9. Standardizing memory fields:
        type, label, causal_role, synthetic, derivation.
 10. Cleaning answer_aliases and must_not_include fields.
 11. Adding quality_status and split_group_id.
 12. Splitting by conversation/sample ID, not random QA rows.
 13. Writing JSONL continuously so partial output is inspectable while running.

Typical usage:
  python build_causal_locomo_llm_v3.py \
    --input locomo.json \
    --output causal_locomo_llm_v3.jsonl \
    --model gpt-4o-mini \
    --variant mixed \
    --num-distractors 4 \
    --num-harmful 1 \
    --harmful-fraction 0.5 \
    --evidence-window-before 1 \
    --evidence-window-after 2 \
    --no-future-leakage \
    --cache-path cache/causal_locomo_llm_v3_cache.jsonl \
    --split-dir causal_locomo_llm_v3_splits \
    --strict-validate

.env file:
  OPENAI_API_KEY=your_key_here
  OPENAI_API_URL=http://192.154.241.225:3000/v1

Do not commit your .env file to git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from openai import OpenAI
except ImportError as exc:
    raise SystemExit("Please install the OpenAI SDK: pip install openai") from exc


# -----------------------------
# .env and IO helpers
# -----------------------------


def load_env_file(path: str | Path = ".env") -> None:
    path = Path(path)
    if not path.exists():
        return

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def make_client(env_file: str, api_key: Optional[str], api_url: Optional[str]) -> OpenAI:
    load_env_file(env_file)

    key = api_key or os.environ.get("OPENAI_API_KEY")
    base_url = api_url or os.environ.get("OPENAI_API_URL") or os.environ.get("OPENAI_BASE_URL")

    if not key:
        raise ValueError("Missing API key. Put OPENAI_API_KEY in .env or pass --api-key.")

    if base_url:
        return OpenAI(api_key=key, base_url=base_url)

    return OpenAI(api_key=key)


def load_json_or_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(f"Empty input file: {path}")

    try:
        data = json.loads(text)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            for key in ["data", "samples", "conversations", "items", "records"]:
                if isinstance(data.get(key), list):
                    return data[key]

            if "conversation" in data and "qa" in data:
                return [data]

        raise ValueError("JSON did not match expected LoCoMo layout.")

    except json.JSONDecodeError:
        records: List[Dict[str, Any]] = []

        for line_no, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc

            if isinstance(obj, dict):
                records.append(obj)

        if not records:
            raise ValueError("No JSONL records found.")

        return records


def append_jsonl(record: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def write_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def reset_file(path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


def write_jsonl_all(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
    reset_file(path)
    for record in records:
        append_jsonl(record, path)


# -----------------------------
# Parsing LoCoMo
# -----------------------------

DIA_RE = re.compile(r"D(\d+):(\d+)")
SESSION_RE = re.compile(r"^session_(\d+)$")
OBS_SESSION_RE = re.compile(r"^session_(\d+)_observation$")


def safe_text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def parse_evidence_ids(evidence: Any) -> List[str]:
    if not evidence:
        return []

    if isinstance(evidence, str):
        evidence = [evidence]

    out: List[str] = []

    for item in evidence:
        for match in DIA_RE.finditer(str(item)):
            out.append(f"D{int(match.group(1))}:{int(match.group(2))}")

    return sorted(set(out), key=lambda x: (int(x.split(":")[0][1:]), int(x.split(":")[1])))


def dia_to_session(dia_id: str) -> Optional[int]:
    match = DIA_RE.match(str(dia_id))
    return int(match.group(1)) if match else None


def dia_to_turn_idx(dia_id: str) -> Optional[int]:
    match = DIA_RE.match(str(dia_id))
    return int(match.group(2)) if match else None


def extract_sessions(conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
    sessions = []

    for key, value in conversation.items():
        match = SESSION_RE.match(str(key))
        if not match or not isinstance(value, list):
            continue

        sid = int(match.group(1))

        sessions.append(
            {
                "session_id": f"s{sid}",
                "timestamp": sid,
                "date_time": conversation.get(f"session_{sid}_date_time"),
                "turns": [
                    {
                        "speaker": turn.get("speaker"),
                        "dia_id": turn.get("dia_id"),
                        "text": turn.get("text", ""),
                    }
                    for turn in value
                    if isinstance(turn, dict)
                ],
            }
        )

    sessions.sort(key=lambda s: s["timestamp"])
    return sessions


def get_session_by_id(sessions: Sequence[Dict[str, Any]], sid: int) -> Optional[Dict[str, Any]]:
    for session in sessions:
        if session["timestamp"] == sid:
            return session
    return None


def get_evidence_windows(
    evidence_ids: Sequence[str],
    sessions: Sequence[Dict[str, Any]],
    before: int,
    after: int,
) -> List[Dict[str, Any]]:
    windows = []

    for evidence_id in evidence_ids:
        sid = dia_to_session(evidence_id)
        if sid is None:
            continue

        session = get_session_by_id(sessions, sid)
        if not session:
            continue

        turns = session.get("turns", [])
        pos = None

        for i, turn in enumerate(turns):
            if turn.get("dia_id") == evidence_id:
                pos = i
                break

        if pos is None:
            continue

        start = max(0, pos - before)
        end = min(len(turns), pos + after + 1)

        windows.append(
            {
                "evidence_id": evidence_id,
                "session_id": f"s{sid}",
                "timestamp": sid,
                "date_time": session.get("date_time"),
                "turns": turns[start:end],
            }
        )

    return windows


def observation_candidates(sample: Dict[str, Any], cutoff_session: Optional[int]) -> List[Dict[str, Any]]:
    obs = sample.get("observation") or {}
    candidates: List[Dict[str, Any]] = []

    if not isinstance(obs, dict):
        return candidates

    for key, value in obs.items():
        match = OBS_SESSION_RE.match(str(key))
        if not match or not isinstance(value, dict):
            continue

        sid = int(match.group(1))

        if cutoff_session is not None and sid > cutoff_session:
            continue

        for speaker, entries in value.items():
            if not isinstance(entries, list):
                continue

            for entry in entries:
                content = ""
                dia_ids: List[str] = []

                if isinstance(entry, (list, tuple)):
                    if entry:
                        content = safe_text(entry[0])
                    if len(entry) > 1:
                        dia_ids = parse_evidence_ids(entry[1])

                elif isinstance(entry, dict):
                    content = safe_text(entry.get("content") or entry.get("text") or entry.get("summary"))
                    dia_ids = parse_evidence_ids(entry.get("evidence") or entry.get("dia_id") or entry.get("source"))

                else:
                    content = safe_text(entry)

                if not content:
                    continue

                candidates.append(
                    {
                        "candidate_id": f"obs_s{sid}_{len(candidates):05d}",
                        "content": content,
                        "scope": str(speaker),
                        "timestamp": sid,
                        "source_session_id": f"s{sid}",
                        "source_dia_ids": dia_ids,
                        "type": "observation",
                    }
                )

    return candidates


def dialogue_candidates(
    sessions: Sequence[Dict[str, Any]],
    cutoff_session: Optional[int],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for session in sessions:
        sid = session["timestamp"]

        if cutoff_session is not None and sid > cutoff_session:
            continue

        for turn in session.get("turns", []):
            dia_id = turn.get("dia_id")
            text = safe_text(turn.get("text"))

            if not dia_id or not text:
                continue

            out.append(
                {
                    "candidate_id": f"turn_{dia_id.replace(':', '_')}",
                    "content": f"{turn.get('speaker')}: {text}",
                    "scope": str(turn.get("speaker") or "unknown"),
                    "timestamp": sid,
                    "source_session_id": f"s{sid}",
                    "source_dia_ids": [dia_id],
                    "type": "dialogue_turn",
                }
            )

    return out


def speaker_names(conversation: Dict[str, Any]) -> List[str]:
    return [
        safe_text(conversation.get(key))
        for key in ["speaker_a", "speaker_b"]
        if safe_text(conversation.get(key))
    ]


# -----------------------------
# Text utilities
# -----------------------------

STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "at", "for", "with", "from",
    "by", "is", "are", "was", "were", "be", "been", "being", "what", "when", "where",
    "who", "why", "how", "did", "does", "do", "has", "have", "had", "would", "could",
    "should", "about", "which", "their", "her", "his", "they", "them", "she", "he", "it",
    "this", "that", "these", "those", "as", "into", "out", "up", "down", "over", "under",
    "before", "after", "during", "recently", "current", "likely", "answer", "question",
}

RELATIVE_TIME_WORDS = {
    "yesterday", "tomorrow", "today", "last", "next", "recently",
    "ago", "before", "after", "week", "month", "year", "sunday",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"
}

ABSTRACT_FORBIDDEN_PHRASES = {
    "any information",
    "unrelated",
    "incorrect date",
    "unrelated events",
    "details about",
    "activities",
    "information unrelated",
}

SOURCE_TYPE_NORMALIZATION = {
    "useful": "observation",
    "irrelevant": "observation",
    "harmful": "synthetic_adversarial",
    "context_dependent": "observation",
    "useful_memory": "observation",
    "irrelevant_memory": "observation",
    "wrong_answer_trap": "synthetic_adversarial",
    "wrong_time_trap": "synthetic_adversarial",
    "synthetic": "synthetic_adversarial",
}

CAUSAL_ROLE_BY_LABEL = {
    "useful": "positive",
    "irrelevant": "neutral",
    "harmful": "negative",
    "context_dependent": "conditional",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", safe_text(text).lower())


def normalize_text(text: str) -> str:
    return " ".join(tokenize(text))


def answer_terms(answer: Any, max_terms: int = 8) -> List[str]:
    text = safe_text(answer)

    if not text:
        return []

    terms: List[str] = []

    if len(text.split()) <= 8:
        terms.append(text)

    terms.extend([t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1])

    seen = set()
    out = []

    for term in terms:
        key = term.lower()
        if key not in seen:
            out.append(term)
            seen.add(key)

    return out[:max_terms]


def answer_aliases(answer: Any) -> List[str]:
    text = safe_text(answer)

    if not text:
        return []

    aliases = [text]

    match = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})$", text)
    if match:
        day, month, year = match.groups()
        aliases.append(f"{month} {int(day)}, {year}")

    match = re.match(r"^([A-Za-z]+)\s+(\d{4})$", text)
    if match:
        aliases.append(f"{match.group(1)} {match.group(2)}")

    seen = set()
    out = []

    for alias in aliases:
        key = alias.lower()
        if key not in seen:
            out.append(alias)
            seen.add(key)

    return out


def contains_answerish(text: str, answer: Any, aliases: Sequence[str]) -> bool:
    norm = normalize_text(text)
    all_answers = [safe_text(answer)] + [safe_text(alias) for alias in aliases]

    for ans in all_answers:
        norm_ans = normalize_text(ans)
        if norm_ans and norm_ans in norm:
            return True

    terms = [term for term in tokenize(safe_text(answer)) if term not in STOPWORDS]
    if terms and all(term in norm.split() for term in terms[: min(3, len(terms))]):
        return True

    return False


def is_temporal_question(question: str, answer: Any) -> bool:
    q = question.lower()
    a = safe_text(answer).lower()

    temporal_markers = [
        "when", "date", "year", "month", "day", "week", "ago", "before", "after",
        "yesterday", "tomorrow", "last", "next", "recently",
    ]

    return any(marker in q for marker in temporal_markers) or bool(re.search(r"\b\d{4}\b", a))


def is_exact_year(text: str) -> bool:
    return bool(re.fullmatch(r"\d{4}", safe_text(text)))


def is_exact_month_year(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+\s+\d{4}", safe_text(text)))


def is_exact_day_month_year(text: str) -> bool:
    return bool(re.fullmatch(r"\d{1,2}\s+[A-Za-z]+\s+\d{4}", safe_text(text)))


def is_exact_month_day_year(text: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z]+\s+\d{1,2},\s*\d{4}", safe_text(text)))


def expected_answer_is_exact_temporal(answer: Any) -> bool:
    text = safe_text(answer)
    return (
        is_exact_year(text)
        or is_exact_month_year(text)
        or is_exact_day_month_year(text)
        or is_exact_month_day_year(text)
    )


def contains_relative_time(text: str) -> bool:
    return bool(set(tokenize(text)) & RELATIVE_TIME_WORDS)


def looks_like_loose_alias(alias: str, expected_answer: Any) -> bool:
    alias = safe_text(alias)
    expected = safe_text(expected_answer)

    if not alias:
        return True

    if alias.lower() == expected.lower():
        return False

    generic_bad = [
        "question",
        "answer",
        "topic",
        "field of study",
        "fields of study",
        "educational pursuits",
        "research focus",
        "study topic",
        "current question",
    ]

    if any(item in alias.lower() for item in generic_bad):
        return True

    if expected_answer_is_exact_temporal(expected) and contains_relative_time(alias):
        return True

    ans_terms = {t for t in tokenize(expected) if t not in STOPWORDS}
    alias_terms = {t for t in tokenize(alias) if t not in STOPWORDS}

    if ans_terms and alias_terms and not (ans_terms & alias_terms):
        return True

    return False


def strict_answer_aliases(expected_answer: Any, llm_aliases: Sequence[str]) -> List[str]:
    aliases = answer_aliases(expected_answer)
    seen = {alias.lower() for alias in aliases}

    for alias in llm_aliases or []:
        alias = safe_text(alias)
        if not alias:
            continue

        if looks_like_loose_alias(alias, expected_answer):
            continue

        if alias.lower() not in seen:
            aliases.append(alias)
            seen.add(alias.lower())

    return aliases


def clean_must_not_include(items: Sequence[str]) -> List[str]:
    cleaned = []
    seen = set()

    for item in items or []:
        item = safe_text(item)
        if not item:
            continue

        low = item.lower()

        if any(bad in low for bad in ABSTRACT_FORBIDDEN_PHRASES):
            continue

        if len(item.split()) > 8:
            continue

        if low not in seen:
            cleaned.append(item)
            seen.add(low)

    return cleaned


def concrete_harmful_terms(harmful: Sequence[Dict[str, Any]], expected_answer: Any) -> List[str]:
    expected_norm = normalize_text(safe_text(expected_answer))
    terms = []
    seen = set()

    for mem in harmful or []:
        content = safe_text(mem.get("content"))
        tokens = [t for t in tokenize(content) if t not in STOPWORDS and len(t) > 2]

        for token in tokens:
            if token in expected_norm.split():
                continue

            if re.fullmatch(r"\d{4}", token) or token in {
                "married", "spouse", "engineering", "finance", "vacation",
                "sunset", "partner", "incorrect", "adopt", "career"
            }:
                if token not in seen:
                    terms.append(token)
                    seen.add(token)

    return terms[:8]


def memory_base_key(text: str) -> str:
    text = safe_text(text)
    text = re.sub(r"Resolved temporal answer:.*?$", "", text, flags=re.I).strip()
    text = re.sub(r"Resolved answer for the current question:.*?$", "", text, flags=re.I).strip()
    text = re.sub(r"Resolved answer:.*?$", "", text, flags=re.I).strip()
    return normalize_text(text)


def jaccard_tokens(a: str, b: str) -> float:
    aa = {t for t in tokenize(a) if t not in STOPWORDS}
    bb = {t for t in tokenize(b) if t not in STOPWORDS}

    if not aa or not bb:
        return 0.0

    return len(aa & bb) / len(aa | bb)


def too_similar_memory(a: str, b: str) -> bool:
    ak = memory_base_key(a)
    bk = memory_base_key(b)

    if not ak or not bk:
        return False

    if ak == bk:
        return True

    if ak in bk or bk in ak:
        return True

    return jaccard_tokens(ak, bk) >= 0.85


def sanitize_temporal_overresolution(content: str, expected_answer: Any, flags: List[str]) -> str:
    """
    Prevent the model from inventing exact calendar dates when LoCoMo's gold answer
    itself is relative, e.g. 'The sunday before 25 May 2023'.

    If expected_answer is already exact, keep exact dates.
    """
    content = safe_text(content)
    expected = safe_text(expected_answer)

    if expected_answer_is_exact_temporal(expected):
        return content

    new_content = re.sub(
        r",?\s*which was\s+[A-Za-z]+\s+\d{1,2},\s*\d{4}",
        "",
        content,
        flags=re.I,
    )
    new_content = re.sub(
        r",?\s*which was\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}",
        "",
        new_content,
        flags=re.I,
    )

    if new_content != content:
        flags.append("removed_unverified_exact_temporal_overresolution")

    return new_content.strip()


def normalize_memory_type(raw_type: str, label: str, default_type: str) -> str:
    raw = safe_text(raw_type).lower()

    if raw in SOURCE_TYPE_NORMALIZATION:
        return SOURCE_TYPE_NORMALIZATION[raw]

    if raw in {"observation", "dialogue_turn", "repaired_gold_memory", "context_dependent_memory"}:
        return raw

    if label == "harmful":
        return "synthetic_adversarial"

    return default_type


def derivation_info(mem: Dict[str, Any], label: str) -> Dict[str, Any]:
    content = safe_text(mem.get("content")).lower()
    mem_type = safe_text(mem.get("type")).lower()

    is_repaired = (
        "resolved answer for the current question" in content
        or "resolved temporal answer" in content
        or mem_type == "repaired_gold_memory"
    )

    if label == "harmful":
        return {
            "is_derived": True,
            "method": "synthetic_adversarial_generation",
            "uses_gold_answer": True,
        }

    if is_repaired:
        return {
            "is_derived": True,
            "method": "llm_or_deterministic_repair_from_evidence_window_and_gold_answer",
            "uses_gold_answer": True,
        }

    return {
        "is_derived": False,
        "method": "grounded_from_locomo_observation_or_dialogue",
        "uses_gold_answer": False,
    }


def quality_status_from_flags(flags: Sequence[str]) -> str:
    flags = [str(flag) for flag in flags or []]

    if any(flag.startswith("WARNING_") for flag in flags):
        return "warning"

    return "pass"


def infer_task_family(qa: Dict[str, Any], evidence_ids: Sequence[str]) -> str:
    category = qa.get("category")

    if category == 5:
        return "adversarial_person_confusion"

    if category == 3:
        return "inferential_memory_qa"

    if category == 2 or is_temporal_question(safe_text(qa.get("question")), qa.get("answer")):
        return "temporal_memory_qa"

    if len(evidence_ids) > 1:
        return "multi_evidence_memory_qa"

    if category == 4:
        return "single_session_detail_qa"

    return "factual_memory_qa"


def compute_cutoff_session(evidence_ids: Sequence[str], sessions: Sequence[Dict[str, Any]]) -> Optional[int]:
    ids = [dia_to_session(evidence_id) for evidence_id in evidence_ids]
    valid = [sid for sid in ids if sid is not None]

    if valid:
        return max(valid)

    return sessions[-1]["timestamp"] if sessions else None


# -----------------------------
# LLM prompt and schema
# -----------------------------

LLM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "is_answerable",
        "task_family",
        "gold_behavior",
        "answer_aliases",
        "useful_memories",
        "irrelevant_memories",
        "harmful_memories",
        "context_dependent_memories",
        "must_include",
        "must_not_include",
        "answerability_notes",
        "quality_flags",
    ],
    "properties": {
        "is_answerable": {"type": "boolean"},
        "task_family": {"type": "string"},
        "gold_behavior": {"type": "string"},
        "answer_aliases": {"type": "array", "items": {"type": "string"}},
        "must_include": {"type": "array", "items": {"type": "string"}},
        "must_not_include": {"type": "array", "items": {"type": "string"}},
        "answerability_notes": {"type": "string"},
        "quality_flags": {"type": "array", "items": {"type": "string"}},
        "useful_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "content",
                    "expected_effect",
                    "scope",
                    "type",
                    "source_candidate_ids",
                    "source_dia_ids",
                    "source_session_ids",
                ],
                "properties": {
                    "content": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "scope": {"type": "string"},
                    "type": {"type": "string"},
                    "source_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "source_dia_ids": {"type": "array", "items": {"type": "string"}},
                    "source_session_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "irrelevant_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "content",
                    "expected_effect",
                    "scope",
                    "type",
                    "source_candidate_ids",
                    "source_dia_ids",
                    "source_session_ids",
                ],
                "properties": {
                    "content": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "scope": {"type": "string"},
                    "type": {"type": "string"},
                    "source_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "source_dia_ids": {"type": "array", "items": {"type": "string"}},
                    "source_session_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "harmful_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["content", "expected_effect", "scope", "type"],
                "properties": {
                    "content": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "scope": {"type": "string"},
                    "type": {"type": "string"},
                },
            },
        },
        "context_dependent_memories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "content",
                    "expected_effect",
                    "scope",
                    "type",
                    "source_candidate_ids",
                    "source_dia_ids",
                    "source_session_ids",
                ],
                "properties": {
                    "content": {"type": "string"},
                    "expected_effect": {"type": "string"},
                    "scope": {"type": "string"},
                    "type": {"type": "string"},
                    "source_candidate_ids": {"type": "array", "items": {"type": "string"}},
                    "source_dia_ids": {"type": "array", "items": {"type": "string"}},
                    "source_session_ids": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}


def build_prompt_payload(
    sample_id: str,
    qa_index: int,
    qa: Dict[str, Any],
    evidence_windows: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    speakers: Sequence[str],
    variant: str,
    num_distractors: int,
    num_harmful: int,
    harmful_allowed: bool,
    include_context_dependent: bool,
) -> Dict[str, Any]:
    return {
        "sample_id": sample_id,
        "qa_index": qa_index,
        "question": safe_text(qa.get("question")),
        "expected_answer": qa.get("answer"),
        "adversarial_answer": qa.get("adversarial_answer"),
        "original_category": qa.get("category"),
        "speakers": list(speakers),
        "variant": variant,
        "num_irrelevant_memories_requested": num_distractors,
        "num_harmful_memories_requested": num_harmful if harmful_allowed else 0,
        "include_context_dependent_memories": include_context_dependent,
        "evidence_windows": evidence_windows,
        "grounded_candidate_memories": list(candidates)[:120],
        "instructions": [
            "Create a Causal-LoCoMo memory-intervention example for the current QA item.",
            "The final output must be auditable and grounded.",
            "Useful memories must be self-contained and sufficient to answer the question without reading provenance or past_sessions.",
            "Use expected_answer as the ground truth. Do not change it.",
            "For temporal questions, include the resolved expected_answer in the useful memory.",
            "Do NOT invent an exact calendar date if expected_answer itself is relative. Example: if expected_answer is 'the week before 9 June 2023', preserve that phrase instead of inventing a specific date.",
            "Irrelevant memories must be selected from grounded_candidate_memories. Do not invent irrelevant facts.",
            "If a memory is selected from grounded_candidate_memories, preserve its content or make only minimal compression.",
            "Harmful memories may be synthetic adversarial traps, but they must be marked as harmful and should contradict or distract from the expected answer.",
            "Do not duplicate the same content across useful, irrelevant, harmful, and context_dependent memories.",
            "Only create context_dependent_memories if include_context_dependent_memories is true. Otherwise return an empty list.",
            "context_dependent_memories should not directly answer the current question. If a memory directly answers the current question, put it in useful_memories instead.",
            "The type field should describe the source: observation, dialogue_turn, repaired_gold_memory, context_dependent_memory, or synthetic_adversarial. Do not use type='useful' or type='irrelevant'.",
            "answer_aliases must contain only valid alternative answers, not descriptions of the question.",
            "must_not_include must contain only concrete forbidden words or phrases, not abstract instructions.",
        ],
    }


def cache_key(payload: Dict[str, Any], model: str, schema_version: str) -> str:
    relevant = {
        "schema_version": schema_version,
        "model": model,
        "payload": payload,
    }
    blob = json.dumps(relevant, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_cache(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        return {}

    cache: Dict[str, Any] = {}

    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue

        try:
            obj = json.loads(line)
            if "key" in obj and "value" in obj:
                cache[obj["key"]] = obj["value"]
        except json.JSONDecodeError:
            continue

    return cache


def append_cache(path: Optional[str], key: str, value: Any) -> None:
    if not path:
        return
    append_jsonl({"key": key, "value": value}, path)


def extract_json_from_text(text: str) -> Dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])

        raise


def call_llm(
    client: OpenAI,
    model: str,
    payload: Dict[str, Any],
    temperature: float,
    max_retries: int,
    use_json_schema: bool,
) -> Dict[str, Any]:
    system = (
        "You are a meticulous dataset curator for LLM-agent memory evaluation. "
        "You create grounded, auditable Causal-LoCoMo examples. "
        "You must follow the JSON schema and avoid unsupported or hallucinated irrelevant memories."
    )

    user = json.dumps(payload, ensure_ascii=False, indent=2)
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": temperature,
            }

            if use_json_schema:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "causal_locomo_generation",
                        "strict": True,
                        "schema": LLM_SCHEMA,
                    },
                }
            else:
                kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or "{}"
            return extract_json_from_text(content)

        except Exception as exc:
            last_error = exc

            # Many OpenAI-compatible proxy servers do not support strict json_schema.
            if use_json_schema:
                use_json_schema = False

            time.sleep(min(8, 1.5 * (attempt + 1)))

    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_error}")


# -----------------------------
# Repair and validation
# -----------------------------


def candidate_by_id(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {candidate["candidate_id"]: candidate for candidate in candidates if candidate.get("candidate_id")}


def fill_memory_sources(mem: Dict[str, Any], candidates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    mem = dict(mem)
    ids = mem.get("source_candidate_ids") or []

    timestamps = []
    session_ids = list(mem.get("source_session_ids") or [])
    dia_ids = list(mem.get("source_dia_ids") or [])

    for candidate_id in ids:
        candidate = candidates.get(candidate_id)

        if not candidate:
            continue

        if candidate.get("timestamp") is not None:
            timestamps.append(candidate.get("timestamp"))

        if candidate.get("source_session_id") and candidate.get("source_session_id") not in session_ids:
            session_ids.append(candidate.get("source_session_id"))

        for dia_id in candidate.get("source_dia_ids") or []:
            if dia_id not in dia_ids:
                dia_ids.append(dia_id)

    if timestamps and mem.get("timestamp") is None:
        mem["timestamp"] = min(timestamps)

    mem["source_session_ids"] = session_ids
    mem["source_dia_ids"] = dia_ids

    return mem


def make_memory_record(
    mem: Dict[str, Any],
    memory_id: str,
    label: str,
    default_type: str,
    candidates: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    mem = fill_memory_sources(mem, candidates)

    session_ids = mem.get("source_session_ids") or []
    source_session_id = session_ids[0] if session_ids else None

    timestamp = mem.get("timestamp")
    if timestamp is None and source_session_id and re.match(r"s\d+", source_session_id):
        timestamp = int(source_session_id[1:])

    normalized_type = normalize_memory_type(
        raw_type=safe_text(mem.get("type")),
        label=label,
        default_type=default_type,
    )

    is_synthetic = normalized_type == "synthetic_adversarial"

    if is_synthetic:
        source_session_id = None
        session_ids = []
        source_dia_ids = []
        source_candidate_ids = []
        timestamp = mem.get("timestamp")
    else:
        source_dia_ids = mem.get("source_dia_ids") or []
        source_candidate_ids = mem.get("source_candidate_ids") or []

    return {
        "memory_id": memory_id,
        "content": safe_text(mem.get("content")),
        "type": normalized_type,
        "label": label,
        "causal_role": CAUSAL_ROLE_BY_LABEL.get(label, "unknown"),
        "synthetic": is_synthetic,
        "derivation": derivation_info(mem, label),
        "scope": safe_text(mem.get("scope")) or "unknown",
        "timestamp": timestamp,
        "expected_effect": safe_text(mem.get("expected_effect")),
        "source_session_id": source_session_id,
        "source_session_ids": session_ids,
        "source_dia_ids": source_dia_ids,
        "source_candidate_ids": source_candidate_ids,
    }


def repair_useful_memories(
    useful: List[Dict[str, Any]],
    qa: Dict[str, Any],
    evidence_windows: Sequence[Dict[str, Any]],
    flags: List[str],
) -> List[Dict[str, Any]]:
    answer = qa.get("answer")
    aliases = answer_aliases(answer)
    question = safe_text(qa.get("question"))
    answer_text = safe_text(answer)

    if not useful:
        flags.append("llm_returned_no_useful_memories")

    evidence_text_parts: List[str] = []
    source_dia_ids: List[str] = []
    source_session_ids: List[str] = []
    session_dates: List[str] = []

    for window in evidence_windows:
        if window.get("session_id") and window.get("session_id") not in source_session_ids:
            source_session_ids.append(window.get("session_id"))

        if window.get("date_time"):
            session_dates.append(str(window.get("date_time")))

        for turn in window.get("turns", []):
            dia_id = turn.get("dia_id")
            if dia_id and dia_id not in source_dia_ids:
                source_dia_ids.append(dia_id)

            speaker = turn.get("speaker") or "Speaker"
            text = safe_text(turn.get("text"))

            if text:
                evidence_text_parts.append(f"{speaker}: {text}")

    for memory in useful:
        memory["content"] = sanitize_temporal_overresolution(
            content=memory.get("content", ""),
            expected_answer=answer,
            flags=flags,
        )

    joined_useful = " ".join(memory.get("content", "") for memory in useful)
    needs_repair = not contains_answerish(joined_useful, answer, aliases)
    temporal = is_temporal_question(question, answer)

    if needs_repair:
        flags.append("gold_memory_missing_expected_answer_repaired")

        if useful:
            useful[0]["content"] = (
                f"{useful[0].get('content', '').rstrip()} "
                f"Resolved answer for the current question: {answer_text}."
            ).strip()
            useful[0]["type"] = "repaired_gold_memory"
            useful[0]["expected_effect"] = (
                safe_text(useful[0].get("expected_effect"))
                + " This memory is repaired to explicitly include the resolved gold answer."
            ).strip()

        else:
            useful.append(
                {
                    "content": (
                        f"Using the evidence from session date(s) "
                        f"{', '.join(session_dates) or 'unknown'}, "
                        f"the resolved answer to '{question}' is: {answer_text}. "
                        f"Evidence: {' '.join(evidence_text_parts[:4])}"
                    ),
                    "expected_effect": "Fallback repaired useful memory that is self-contained and includes the resolved answer.",
                    "scope": "gold_evidence",
                    "type": "repaired_gold_memory",
                    "source_candidate_ids": [],
                    "source_dia_ids": source_dia_ids,
                    "source_session_ids": source_session_ids,
                }
            )

    elif temporal:
        if useful and "resolved" not in useful[0].get("content", "").lower():
            useful[0]["content"] = (
                f"{useful[0].get('content', '').rstrip()} "
                f"Resolved temporal answer: {answer_text}."
            ).strip()
            useful[0]["type"] = "repaired_gold_memory"
            flags.append("temporal_gold_memory_made_explicit")

    if len(useful) > 1:
        strong = [memory for memory in useful if contains_answerish(memory.get("content", ""), answer, aliases)]
        weak = [memory for memory in useful if not contains_answerish(memory.get("content", ""), answer, aliases)]

        if strong and weak:
            useful = strong
            flags.append("removed_weak_useful_memories")

    return useful


def dedupe_memory_groups(
    useful: List[Dict[str, Any]],
    irrelevant: List[Dict[str, Any]],
    harmful: List[Dict[str, Any]],
    context_dep: List[Dict[str, Any]],
    flags: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:

    def dedupe_internal(group: List[Dict[str, Any]], group_name: str) -> List[Dict[str, Any]]:
        out = []

        for memory in group:
            if not safe_text(memory.get("content")):
                continue

            if any(too_similar_memory(memory.get("content", ""), existing.get("content", "")) for existing in out):
                flags.append(f"removed_duplicate_{group_name}_memory")
                continue

            out.append(memory)

        return out

    useful = dedupe_internal(useful, "useful")

    def remove_overlap_with_gold(group: List[Dict[str, Any]], group_name: str) -> List[Dict[str, Any]]:
        out = []

        for memory in group:
            if any(too_similar_memory(memory.get("content", ""), gold.get("content", "")) for gold in useful):
                flags.append(f"removed_duplicate_{group_name}_overlap_with_gold")
                continue

            out.append(memory)

        return dedupe_internal(out, group_name)

    irrelevant = remove_overlap_with_gold(irrelevant, "irrelevant")
    harmful = remove_overlap_with_gold(harmful, "harmful")
    context_dep = remove_overlap_with_gold(context_dep, "context_dependent")

    return useful, irrelevant, harmful, context_dep


def grounded_fill_irrelevant(
    irrelevant: List[Dict[str, Any]],
    useful: List[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    num_distractors: int,
    flags: List[str],
) -> List[Dict[str, Any]]:
    useful_norms = {memory_base_key(memory.get("content", "")) for memory in useful}
    existing_norms = {memory_base_key(memory.get("content", "")) for memory in irrelevant}
    out = list(irrelevant)

    for candidate in candidates:
        if len(out) >= num_distractors:
            break

        key = memory_base_key(candidate.get("content", ""))

        if not key or key in useful_norms or key in existing_norms:
            continue

        out.append(
            {
                "content": candidate["content"],
                "expected_effect": "Grounded distractor memory from the same conversation that should not be used for this question.",
                "scope": candidate.get("scope") or "unknown",
                "type": candidate.get("type") or "observation",
                "source_candidate_ids": [candidate["candidate_id"]],
                "source_dia_ids": candidate.get("source_dia_ids") or [],
                "source_session_ids": [candidate.get("source_session_id")] if candidate.get("source_session_id") else [],
            }
        )

        existing_norms.add(key)
        flags.append("filled_missing_irrelevant_from_grounded_candidate")

    return out[:num_distractors]


def synthetic_harmful_fallback(qa: Dict[str, Any], speakers: Sequence[str]) -> Dict[str, Any]:
    question = safe_text(qa.get("question"))
    answer = safe_text(qa.get("answer"))
    adversarial = safe_text(qa.get("adversarial_answer"))

    if adversarial:
        content = f"Misleading note: For the question '{question}', the answer is '{adversarial}'."
    elif len(speakers) >= 2:
        content = (
            f"Misleading note: The answer for {speakers[1]} should be used for {speakers[0]}. "
            f"This conflicts with the true answer '{answer}'."
        )
    elif answer:
        content = f"Misleading note: The answer is not '{answer}'; use a different value from another conversation."
    else:
        content = "Misleading note: This question should be answered from unrelated memories."

    return {
        "content": content,
        "expected_effect": "Synthetic adversarial memory designed to test harmful-memory rejection.",
        "scope": "synthetic_adversarial",
        "type": "synthetic_adversarial",
    }


def repair_scoring_criteria(
    qa: Dict[str, Any],
    llm_obj: Dict[str, Any],
    harmful_records: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    expected = safe_text(qa.get("answer"))

    aliases = strict_answer_aliases(
        expected_answer=expected,
        llm_aliases=llm_obj.get("answer_aliases") or [],
    )

    must_include = answer_terms(expected)

    if safe_text(expected) and safe_text(expected) not in must_include:
        must_include.insert(0, safe_text(expected))

    must_not_include = clean_must_not_include(llm_obj.get("must_not_include") or [])

    adversarial = safe_text(qa.get("adversarial_answer")) or None
    if adversarial:
        for term in answer_terms(adversarial):
            if term not in must_not_include:
                must_not_include.append(term)

    for term in concrete_harmful_terms(harmful_records or [], expected):
        if term not in must_not_include:
            must_not_include.append(term)

    return {
        "expected_answer": expected,
        "answer_aliases": aliases,
        "adversarial_answer": adversarial if adversarial is not None else None,
        "must_include": must_include,
        "must_not_include": must_not_include,
        "style": "answer concisely using only the applicable conversational memory",
        "max_words": 60,
        "evaluation_mode": "exact_or_llm_judge",
        "judge_instructions": (
            "Mark correct if the response answers the question consistently with expected_answer or an answer_alias, "
            "uses the useful memory, and does not adopt harmful or irrelevant memories. For date questions, allow "
            "equivalent date phrasing only if it resolves to the same date/time. Penalize adoption of bad memories."
        ),
    }


def postprocess_llm_output(
    llm_obj: Dict[str, Any],
    qa: Dict[str, Any],
    evidence_windows: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    speakers: Sequence[str],
    variant: str,
    num_distractors: int,
    num_harmful: int,
    harmful_allowed: bool,
    strict_grounded_irrelevant: bool,
    include_context_dependent: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:

    flags = list(llm_obj.get("quality_flags") or [])
    cand_map = candidate_by_id(candidates)

    useful_raw = list(llm_obj.get("useful_memories") or [])
    irrelevant_raw = list(llm_obj.get("irrelevant_memories") or [])
    harmful_raw = list(llm_obj.get("harmful_memories") or []) if harmful_allowed else []

    if include_context_dependent:
        context_raw = list(llm_obj.get("context_dependent_memories") or [])
    else:
        context_raw = []
        if llm_obj.get("context_dependent_memories"):
            flags.append("dropped_context_dependent_memories_by_default")

    useful_raw = repair_useful_memories(useful_raw, qa, evidence_windows, flags)

    if strict_grounded_irrelevant:
        grounded_irrelevant = []

        for memory in irrelevant_raw:
            if memory.get("source_candidate_ids"):
                grounded_irrelevant.append(memory)
            else:
                flags.append("dropped_ungrounded_irrelevant_memory")

        irrelevant_raw = grounded_irrelevant

    useful_raw, irrelevant_raw, harmful_raw, context_raw = dedupe_memory_groups(
        useful_raw, irrelevant_raw, harmful_raw, context_raw, flags
    )

    irrelevant_raw = grounded_fill_irrelevant(
        irrelevant=irrelevant_raw,
        useful=useful_raw,
        candidates=candidates,
        num_distractors=num_distractors,
        flags=flags,
    )

    if harmful_allowed and len(harmful_raw) < num_harmful:
        harmful_raw.append(synthetic_harmful_fallback(qa, speakers))
        flags.append("filled_missing_harmful_synthetic_adversarial")

    harmful_raw = harmful_raw[:num_harmful]

    useful = [
        make_memory_record(memory, f"gold_{i:02d}", "useful", "observation", cand_map)
        for i, memory in enumerate(useful_raw)
    ]

    irrelevant = [
        make_memory_record(memory, f"irr_{i:02d}", "irrelevant", "observation", cand_map)
        for i, memory in enumerate(irrelevant_raw[:num_distractors])
    ]

    harmful = [
        make_memory_record(memory, f"harm_{i:02d}", "harmful", "synthetic_adversarial", cand_map)
        for i, memory in enumerate(harmful_raw)
    ]

    context_dep = [
        make_memory_record(memory, f"ctx_{i:02d}", "context_dependent", "context_dependent_memory", cand_map)
        for i, memory in enumerate(context_raw)
    ]

    useful, irrelevant, harmful, context_dep = dedupe_memory_groups(
        useful, irrelevant, harmful, context_dep, flags
    )

    answer = qa.get("answer")
    aliases = answer_aliases(answer)

    filtered_context = []
    for memory in context_dep:
        if contains_answerish(memory.get("content", ""), answer, aliases):
            flags.append("removed_context_dependent_memory_that_directly_answers_question")
            continue
        filtered_context.append(memory)

    context_dep = filtered_context

    gold_text = " ".join(memory["content"] for memory in useful)
    if not contains_answerish(gold_text, qa.get("answer"), answer_aliases(qa.get("answer"))):
        flags.append("WARNING_gold_memory_still_may_not_contain_expected_answer")

    return useful, irrelevant, harmful, context_dep, flags


# -----------------------------
# Example builder
# -----------------------------


def build_past_sessions(
    sessions: Sequence[Dict[str, Any]],
    cutoff_session: Optional[int],
    include_full_sessions: bool,
) -> List[Dict[str, Any]]:
    out = []

    for session in sessions:
        if cutoff_session is not None and session["timestamp"] > cutoff_session:
            continue

        item = {
            "session_id": session["session_id"],
            "timestamp": session["timestamp"],
            "date_time": session.get("date_time"),
            "content": session_content(session),
        }

        if include_full_sessions:
            item["turns"] = session.get("turns", [])

        out.append(item)

    return out


def session_content(session: Dict[str, Any], max_turns: int = 80) -> str:
    parts = []
    if session.get("date_time"):
        parts.append(f"Session date/time: {session.get('date_time')}.")

    for turn in (session.get("turns") or [])[:max_turns]:
        speaker = safe_text(turn.get("speaker")) or "Speaker"
        text = safe_text(turn.get("text"))
        dia_id = safe_text(turn.get("dia_id"))
        if not text:
            continue
        prefix = f"{dia_id} " if dia_id else ""
        parts.append(f"{prefix}{speaker}: {text}")

    return "\n".join(parts) if parts else f"Conversation session {session.get('session_id')}."


def normalize_benchmark_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert rich Causal-LoCoMo output into the repo's BenchmarkExample schema.

    Extra provenance fields are preserved where Pydantic ignores them, but all
    fields required by the experiment runner are made concrete and ordered.
    """
    record = dict(record)
    past_sessions = []
    max_session_ts = 1

    for session in record.get("past_sessions") or []:
        session = dict(session)
        timestamp = session.get("timestamp")
        if timestamp is None:
            timestamp = max_session_ts
        timestamp = int(timestamp)
        max_session_ts = max(max_session_ts, timestamp)
        if not safe_text(session.get("content")):
            session["content"] = session_content(session)
        session["timestamp"] = timestamp
        past_sessions.append(session)

    past_sessions.sort(key=lambda item: item["timestamp"])
    record["past_sessions"] = past_sessions

    memory_bank = []
    for i, memory in enumerate(record.get("memory_bank") or [], start=1):
        memory = dict(memory)
        timestamp = memory.get("timestamp")
        if timestamp is None:
            timestamp = max_session_ts + i
        try:
            timestamp = int(timestamp)
        except (TypeError, ValueError):
            timestamp = max_session_ts + i
        memory["timestamp"] = timestamp
        memory["content"] = safe_text(memory.get("content"))
        memory["type"] = safe_text(memory.get("type")) or "observation"
        memory["label"] = safe_text(memory.get("label")) or "irrelevant"
        memory["scope"] = safe_text(memory.get("scope")) or "unknown"
        memory["expected_effect"] = safe_text(memory.get("expected_effect")) or "No expected effect provided."
        memory_bank.append(memory)

    label_rank = {"useful": 0, "irrelevant": 1, "context_dependent": 2, "harmful": 3}
    memory_bank.sort(key=lambda item: (item["timestamp"], label_rank.get(item.get("label"), 9), item.get("memory_id", "")))
    record["memory_bank"] = memory_bank

    scoring = dict(record.get("scoring_criteria") or {})
    if scoring.get("expected_answer") is not None:
        scoring["expected_answer"] = safe_text(scoring.get("expected_answer"))
    scoring["must_include"] = [safe_text(item) for item in scoring.get("must_include") or [] if safe_text(item)]
    scoring["must_not_include"] = [safe_text(item) for item in scoring.get("must_not_include") or [] if safe_text(item)]
    scoring["answer_aliases"] = [safe_text(item) for item in scoring.get("answer_aliases") or [] if safe_text(item)]
    record["scoring_criteria"] = scoring

    task = dict(record.get("current_task") or {})
    task.setdefault("recipient_type", "qa_user")
    task.setdefault("domain", "long_conversation_memory")
    record["current_task"] = task

    return record


def make_example(
    sample: Dict[str, Any],
    sample_id: str,
    qa: Dict[str, Any],
    qa_index: int,
    llm_obj: Dict[str, Any],
    sessions: Sequence[Dict[str, Any]],
    evidence_windows: Sequence[Dict[str, Any]],
    candidates: Sequence[Dict[str, Any]],
    speakers: Sequence[str],
    variant: str,
    cutoff_session: Optional[int],
    include_full_sessions: bool,
    num_distractors: int,
    num_harmful: int,
    harmful_allowed: bool,
    strict_grounded_irrelevant: bool,
    include_context_dependent: bool,
) -> Optional[Dict[str, Any]]:

    useful, irrelevant, harmful, context_dep, flags = postprocess_llm_output(
        llm_obj=llm_obj,
        qa=qa,
        evidence_windows=evidence_windows,
        candidates=candidates,
        speakers=speakers,
        variant=variant,
        num_distractors=num_distractors,
        num_harmful=num_harmful,
        harmful_allowed=harmful_allowed,
        strict_grounded_irrelevant=strict_grounded_irrelevant,
        include_context_dependent=include_context_dependent,
    )

    if not llm_obj.get("is_answerable", True):
        flags.append("not_answerable_by_llm")

    if not useful:
        flags.append("dropped_no_useful_memories")
        return None

    memory_bank = useful + irrelevant + context_dep + harmful

    gold_ids = [memory["memory_id"] for memory in useful]
    bad_ids = [memory["memory_id"] for memory in irrelevant + harmful]
    ctx_ids = [memory["memory_id"] for memory in context_dep]

    scoring = repair_scoring_criteria(qa, llm_obj, harmful_records=harmful)
    evidence_ids = parse_evidence_ids(qa.get("evidence"))
    task_family = infer_task_family(qa, evidence_ids)

    record = {
        "example_id": f"{sample_id}_qa_{qa_index:04d}",
        "source_dataset": "LoCoMo",
        "source_sample_id": sample_id,
        "split_group_id": sample_id,
        "quality_status": quality_status_from_flags(flags),
        "task_family": task_family,
        "input_format": "single_turn_conversational_memory_task",
        "dataset_variant": variant,
        "memory_setting": "temporal_no_future_leakage" if cutoff_session is not None else "full_history_memory_pool",
        "past_sessions": build_past_sessions(sessions, cutoff_session, include_full_sessions),
        "memory_bank": memory_bank,
        "current_task": {
            "task_id": f"qa_{qa_index:04d}",
            "instruction": safe_text(qa.get("question")),
            "task_type": "conversational_memory_qa",
            "domain": "long_conversation_memory",
        },
        "gold_memory_ids": gold_ids,
        "bad_memory_ids": bad_ids,
        "context_dependent_memory_ids": ctx_ids,
        "gold_behavior": safe_text(llm_obj.get("gold_behavior")) or f"Answer: {safe_text(qa.get('answer'))}",
        "scoring_criteria": scoring,
        "metadata": {
            "original_category": qa.get("category"),
            "original_evidence": evidence_ids,
            "cutoff_session": cutoff_session,
            "difficulty": "hard" if len(evidence_ids) > 1 or qa.get("category") in {3, 5} else "medium",
            "requires_temporal_reasoning": is_temporal_question(safe_text(qa.get("question")), qa.get("answer")),
            "requires_inference": qa.get("category") == 3,
            "contains_poisoned_memory": bool(harmful),
            "contains_conflict": qa.get("category") == 5 or bool(qa.get("adversarial_answer")),
            "num_gold_memories": len(gold_ids),
            "num_bad_memories": len(bad_ids),
            "num_harmful_memories": len(harmful),
            "llm_answerability_notes": safe_text(llm_obj.get("answerability_notes")),
            "quality_flags": sorted(set(flags)),
        },
        "intervention_tests": {
            "no_memory_expected": "The model may answer incorrectly or abstain because the relevant long-term memory is absent.",
            "with_gold_memory_expected": "The model should answer consistently with expected_answer using only useful memories.",
            "with_bad_memory_expected": "The model may be distracted by irrelevant, context-misaligned, or harmful memories.",
        },
        "provenance": {
            "evidence_windows": evidence_windows,
        },
    }

    return normalize_benchmark_record(record)


def harmful_allowed_for_example(variant: str, rng: random.Random, harmful_fraction: float) -> bool:
    if variant == "clean":
        return False

    if variant == "adversarial":
        return True

    if variant == "spurious":
        return False

    return rng.random() < harmful_fraction


def build_dataset(args: argparse.Namespace) -> List[Dict[str, Any]]:
    client = make_client(args.env_file, args.api_key, args.api_url)
    raw = load_json_or_jsonl(args.input)
    rng = random.Random(args.seed)
    cache = load_cache(args.cache_path)

    # Important: changing this prevents old cached generations from being reused.
    schema_version = "causal_locomo_improved_v3_no_ctx_dedup_temporal_validated"

    if args.output:
        reset_file(args.output)

    records: List[Dict[str, Any]] = []
    total_seen = 0

    prepared_samples: List[Dict[str, Any]] = []
    for sample_idx, sample in enumerate(raw):
        conversation = sample.get("conversation") or {}
        qa_items = sample.get("qa") or []

        if not isinstance(conversation, dict) or not isinstance(qa_items, list):
            continue

        sample_id = safe_text(
            sample.get("sample_id")
            or sample.get("conversation_id")
            or sample.get("id")
            or f"conv_{sample_idx}"
        )

        sessions = extract_sessions(conversation)
        speakers = speaker_names(conversation)

        prepared_samples.append(
            {
                "sample": sample,
                "sample_id": sample_id,
                "sessions": sessions,
                "speakers": speakers,
                "qa_items": qa_items,
            }
        )

    if args.sequential_samples:
        jobs = [
            (prepared, qa_index, qa)
            for prepared in prepared_samples
            for qa_index, qa in enumerate(prepared["qa_items"])
        ]
    else:
        jobs = []
        max_qas = max((len(prepared["qa_items"]) for prepared in prepared_samples), default=0)
        for qa_index in range(max_qas):
            for prepared in prepared_samples:
                if qa_index < len(prepared["qa_items"]):
                    jobs.append((prepared, qa_index, prepared["qa_items"][qa_index]))

    for prepared, qa_index, qa in jobs:
        sample = prepared["sample"]
        sample_id = prepared["sample_id"]
        sessions = prepared["sessions"]
        speakers = prepared["speakers"]

        if args.max_examples is not None and len(records) >= args.max_examples:
            return records

        if not isinstance(qa, dict) or not safe_text(qa.get("question")):
            continue

        if qa.get("answer") is None and args.skip_no_answer:
            continue

        evidence_ids = parse_evidence_ids(qa.get("evidence"))

        if not evidence_ids and args.require_evidence:
            continue

        cutoff = compute_cutoff_session(evidence_ids, sessions) if args.no_future_leakage else None

        evidence_windows = get_evidence_windows(
            evidence_ids,
            sessions,
            args.evidence_window_before,
            args.evidence_window_after,
        )

        candidates = observation_candidates(sample, cutoff)

        if args.include_dialogue_candidates:
            candidates.extend(dialogue_candidates(sessions, cutoff))

        if len(candidates) < args.num_distractors + 1:
            existing = {candidate["candidate_id"] for candidate in candidates}

            for candidate in dialogue_candidates(sessions, cutoff):
                if candidate["candidate_id"] not in existing:
                    candidates.append(candidate)
                    existing.add(candidate["candidate_id"])

        allowed_harm = harmful_allowed_for_example(args.variant, rng, args.harmful_fraction)

        payload = build_prompt_payload(
            sample_id=sample_id,
            qa_index=qa_index,
            qa=qa,
            evidence_windows=evidence_windows,
            candidates=candidates,
            speakers=speakers,
            variant=args.variant,
            num_distractors=args.num_distractors,
            num_harmful=args.num_harmful,
            harmful_allowed=allowed_harm,
            include_context_dependent=args.include_context_dependent,
        )

        key = cache_key(payload, args.model, schema_version)

        if key in cache:
            llm_obj = cache[key]
            from_cache = True
        else:
            llm_obj = call_llm(
                client=client,
                model=args.model,
                payload=payload,
                temperature=args.temperature,
                max_retries=args.max_retries,
                use_json_schema=not args.no_json_schema,
            )
            append_cache(args.cache_path, key, llm_obj)
            cache[key] = llm_obj
            from_cache = False

        example = make_example(
            sample=sample,
            sample_id=sample_id,
            qa=qa,
            qa_index=qa_index,
            llm_obj=llm_obj,
            sessions=sessions,
            evidence_windows=evidence_windows,
            candidates=candidates,
            speakers=speakers,
            variant=args.variant,
            cutoff_session=cutoff,
            include_full_sessions=args.include_full_sessions,
            num_distractors=args.num_distractors,
            num_harmful=args.num_harmful,
            harmful_allowed=allowed_harm,
            strict_grounded_irrelevant=args.strict_grounded_irrelevant,
            include_context_dependent=args.include_context_dependent,
        )

        total_seen += 1

        if example is None:
            continue

        if args.drop_warning_examples and example["quality_status"] == "warning":
            continue

        records.append(example)

        if args.output:
            append_jsonl(example, args.output)

        if len(records) % args.progress_every == 0:
            print(
                f"Built {len(records)} examples "
                f"({total_seen} QA seen, last={'cache' if from_cache else 'api'}): {example['example_id']}",
                flush=True,
            )
    return records


# -----------------------------
# Summary, split, validation
# -----------------------------


def validate_records(records: Sequence[Dict[str, Any]], strict: bool = False) -> None:
    try:
        from src.benchmark.schema import BenchmarkExample
    except Exception:  # noqa: BLE001
        BenchmarkExample = None

    seen = set()

    for record in records:
        if BenchmarkExample is not None:
            BenchmarkExample(**record)

        example_id = record["example_id"]

        if example_id in seen:
            raise ValueError(f"Duplicate example_id: {example_id}")

        seen.add(example_id)

        memory_ids = [memory["memory_id"] for memory in record["memory_bank"]]

        if len(memory_ids) != len(set(memory_ids)):
            raise ValueError(f"Duplicate memory_id in {example_id}")

        memory_set = set(memory_ids)

        for field in ["gold_memory_ids", "bad_memory_ids", "context_dependent_memory_ids"]:
            missing = set(record.get(field, [])) - memory_set
            if missing:
                raise ValueError(f"{example_id}: ids in {field} missing from memory_bank: {missing}")

        gold_text = " ".join(
            memory["content"]
            for memory in record["memory_bank"]
            if memory["memory_id"] in record["gold_memory_ids"]
        )
        expected = record["scoring_criteria"].get("expected_answer")
        aliases = record["scoring_criteria"].get("answer_aliases") or []

        if strict and not contains_answerish(gold_text, expected, aliases):
            raise ValueError(f"{example_id}: gold memories may not contain expected answer")

        gold_memories = [
            memory for memory in record["memory_bank"]
            if memory["memory_id"] in record["gold_memory_ids"]
        ]
        ctx_memories = [
            memory for memory in record["memory_bank"]
            if memory["memory_id"] in record["context_dependent_memory_ids"]
        ]

        for ctx in ctx_memories:
            if any(too_similar_memory(ctx["content"], gold["content"]) for gold in gold_memories):
                raise ValueError(f"{example_id}: context-dependent memory duplicates gold memory")


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    family_counts = Counter(record.get("task_family") for record in records)
    category_counts = Counter(str((record.get("metadata") or {}).get("original_category")) for record in records)
    flag_counts = Counter()
    status_counts = Counter(record.get("quality_status", "unknown") for record in records)

    harmful_count = 0
    warning_flag_count = 0
    synthetic_memory_count = 0
    context_memory_count = 0

    for record in records:
        metadata = record.get("metadata") or {}
        harmful_count += int(bool(metadata.get("contains_poisoned_memory")))

        for memory in record.get("memory_bank", []):
            synthetic_memory_count += int(bool(memory.get("synthetic")))
            context_memory_count += int(memory.get("label") == "context_dependent")

        for flag in metadata.get("quality_flags") or []:
            flag_counts[flag] += 1
            if str(flag).startswith("WARNING_"):
                warning_flag_count += 1

    n = len(records)

    return {
        "num_examples": n,
        "task_family_counts": dict(family_counts),
        "original_category_counts": dict(category_counts),
        "quality_status_counts": dict(status_counts),
        "avg_memory_bank_size": sum(len(record.get("memory_bank", [])) for record in records) / n if n else 0,
        "avg_gold_memories": sum(len(record.get("gold_memory_ids", [])) for record in records) / n if n else 0,
        "avg_bad_memories": sum(len(record.get("bad_memory_ids", [])) for record in records) / n if n else 0,
        "contains_poisoned_memory": harmful_count,
        "num_synthetic_memories": synthetic_memory_count,
        "num_context_dependent_memories": context_memory_count,
        "num_examples_with_warning_flags": warning_flag_count,
        "quality_flag_counts": dict(flag_counts),
        "num_split_groups": len(set(record.get("split_group_id") for record in records)),
    }


def split_records(
    records: Sequence[Dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(seed)

    groups: Dict[str, List[Dict[str, Any]]] = {}

    for record in records:
        group_id = record.get("split_group_id") or record.get("source_sample_id") or record.get("example_id")
        groups.setdefault(group_id, []).append(record)

    group_ids = list(groups.keys())
    rng.shuffle(group_ids)

    n_groups = len(group_ids)
    n_train = int(n_groups * train_ratio)
    n_dev = int(n_groups * dev_ratio)

    train_groups = set(group_ids[:n_train])
    dev_groups = set(group_ids[n_train : n_train + n_dev])

    train: List[Dict[str, Any]] = []
    dev: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []

    for group_id, items in groups.items():
        if group_id in train_groups:
            train.extend(items)
        elif group_id in dev_groups:
            dev.extend(items)
        else:
            test.extend(items)

    return train, dev, test


# -----------------------------
# CLI
# -----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build improved Causal-LoCoMo using an OpenAI-compatible LLM and deterministic repair."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--model", default="gpt-5")

    parser.add_argument("--variant", choices=["clean", "spurious", "adversarial", "mixed"], default="mixed")
    parser.add_argument("--num-distractors", type=int, default=4)
    parser.add_argument("--num-harmful", type=int, default=1)
    parser.add_argument("--harmful-fraction", type=float, default=0.5)

    parser.add_argument("--evidence-window-before", type=int, default=1)
    parser.add_argument("--evidence-window-after", type=int, default=2)

    parser.add_argument("--no-future-leakage", action="store_true")
    parser.add_argument("--include-full-sessions", action="store_true")
    parser.add_argument("--include-dialogue-candidates", action="store_true")
    parser.add_argument(
        "--sequential-samples",
        action="store_true",
        help="Walk all QA items from one conversation before moving to the next. Default is round-robin across conversations.",
    )

    parser.add_argument(
        "--include-context-dependent",
        action="store_true",
        help="Include context-dependent memories. Default is off to avoid duplicate gold/context memories.",
    )

    parser.add_argument("--strict-grounded-irrelevant", action="store_true", default=True)
    parser.add_argument("--allow-ungrounded-irrelevant", dest="strict_grounded_irrelevant", action="store_false")

    parser.add_argument("--require-evidence", action="store_true", default=True)
    parser.add_argument("--allow-no-evidence", dest="require_evidence", action="store_false")

    parser.add_argument("--skip-no-answer", action="store_true", default=True)
    parser.add_argument("--keep-no-answer", dest="skip_no_answer", action="store_false")

    parser.add_argument("--drop-warning-examples", action="store_true")

    parser.add_argument("--cache-path", default="cache/causal_locomo_llm_v3_cache.jsonl")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--summary-path", default=None)

    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument(
        "--no-json-schema",
        action="store_true",
        help="Use JSON mode instead of strict json_schema. Useful for limited OpenAI-compatible proxies.",
    )
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--strict-validate", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = build_dataset(args)
    validate_records(records, strict=args.strict_validate)

    # Re-write full output once at the end for clean ordering,
    # even though the script also appends partial outputs during generation.
    write_jsonl_all(records, args.output)

    summary = summarize(records)
    summary_path = args.summary_path or str(Path(args.output).with_suffix(Path(args.output).suffix + ".summary.json"))
    write_json(summary, summary_path)

    if args.split_dir:
        train, dev, test = split_records(records, args.train_ratio, args.dev_ratio, args.seed)
        split_dir = Path(args.split_dir)

        write_jsonl_all(train, split_dir / "train.jsonl")
        write_jsonl_all(dev, split_dir / "dev.jsonl")
        write_jsonl_all(test, split_dir / "test.jsonl")

        write_json(
            {
                "train": summarize(train),
                "dev": summarize(dev),
                "test": summarize(test),
            },
            split_dir / "split_summary.json",
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nWrote: {args.output}")
    print(f"Summary: {summary_path}")

    if args.split_dir:
        print(f"Splits: {args.split_dir}")


if __name__ == "__main__":
    main()
