from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Iterable


TOKEN_RE = re.compile(r"[A-Za-z0-9']+")


def get_field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def word_count(text: str) -> int:
    return len(tokenize(text))


def keyword_overlap(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    denom = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / denom if denom else 0.0


def deterministic_embedding(text: str, dimensions: int = 64) -> list[float]:
    vec = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


def memory_text(memory: Any) -> str:
    return str(get_field(memory, "content", ""))


def memory_id(memory: Any) -> str:
    return str(get_field(memory, "memory_id", ""))


def format_memories(memories: Iterable[Any]) -> str:
    lines = []
    for memory in memories:
        mid = memory_id(memory)
        content = memory_text(memory)
        scope = get_field(memory, "scope", None)
        label = get_field(memory, "label", None)
        prefix = f"- {mid}: {content}"
        if scope:
            prefix += f" [scope: {scope}]"
        if label:
            prefix += f" [label: {label}]"
        lines.append(prefix)
    return "\n".join(lines) if lines else "(none)"


def summarize_memories(memories: Iterable[Any]) -> str:
    contents = [memory_text(memory) for memory in memories]
    if not contents:
        return "No durable memory is available."
    return " ".join(contents[:8])


def _combined_memory_text(memories: Iterable[Any]) -> str:
    return " ".join(memory_text(memory).lower() for memory in memories)


def _has(mem_text: str, *phrases: str) -> bool:
    return any(phrase.lower() in mem_text for phrase in phrases)


def generate_local_answer(task: Any, memories: Iterable[Any] = ()) -> str:
    """Cheap deterministic answer generator used for tests and dry runs.

    The implementation intentionally reacts to injected memories so baselines can
    be evaluated without live model calls.
    """
    instruction = str(get_field(task, "instruction", task))
    domain = str(get_field(task, "domain", "")).lower()
    recipient = str(get_field(task, "recipient_type", "")).lower()
    task_l = instruction.lower()
    mem_l = _combined_memory_text(memories)
    dr_match = re.search(r"Dr\.\s+([A-Z][A-Za-z'-]+)", instruction)
    dr_name = f"Dr. {dr_match.group(1)}" if dr_match else "Professor"
    course_match = re.search(r"\b(?:CSCI|STAT|ECON|BIOE|MATH)\s+\d{4}\b", instruction + " " + mem_l, flags=re.IGNORECASE)
    course = course_match.group(0).upper() if course_match else None

    if _has(mem_l, "ask for a grade change", "demanding tone", "ignore the question"):
        return "Dear Professor, please change my grade and prioritize my request immediately."
    if _has(mem_l, "always write extremely long"):
        return (
            "Dear Professor, I am writing a lengthy explanation with many unnecessary details, "
            "background comments, and repeated justifications before asking about coffee, because "
            "I believe every message should be formal and extended."
        )
    if "battery" in task_l and _has(mem_l, "optical modulator", "modulators"):
        return "BTO is mainly useful here because of its optical modulator properties in photonics devices."
    if _has(mem_l, "for all messages, use a formal tone") and ("friend" in task_l or "instagram" in task_l):
        return "Dear Professor, I would formally like to request your consideration of this casual matter."

    if "coffee" in task_l and ("friend" in task_l or recipient == "friend"):
        if _has(mem_l, "concise", "short"):
            return "Want to grab coffee sometime this week?"
        return "Hey, would you like to get coffee sometime this week?"

    if "instagram" in task_l or "caption" in task_l:
        if _has(mem_l, "professor", "formal"):
            return "Dear Professor, this photograph is presented with formal appreciation."
        return "Golden hour, good company, and a little city air."

    if "ar(2)" in task_l or "autocorrelation" in task_l:
        if _has(mem_l, "yule-walker", "yule walker"):
            return "Set up the Yule-Walker equations for AR(2), solve for rho_1 and rho_2, then substitute the coefficients to obtain the autocorrelation recursively."
        return "Use the AR recurrence and solve the autocorrelation equations for the requested lag."

    if "bto" in task_l and "battery" in task_l:
        return "In battery materials, BTO may refer to a titanate additive or coating whose dielectric and interfacial properties can affect ion transport and stability."

    if "submit" in task_l or "report" in task_l:
        if _has(mem_l, "zip file", "zip"):
            return "For this class, submit the final report as a zip file with all required materials included."
        if _has(mem_l, "pdf"):
            return "Submit the report as a PDF."
        return "Submit the report in the format requested by the current class instructions."

    if "dr." in task_l or "professor" in task_l or "proposal exam" in task_l:
        greeting = f"Dear {dr_name},"
        if course and (_has(mem_l, "respectful and concise") or re.search(r"\b(?:csci|stat|econ|bioe|math)\s+\d{4}\b", mem_l)):
            return f"{greeting} I am writing about {course} to ask about the final report extension. Thank you for considering my request."
        if _has(mem_l, "formal", "respectful", "concise"):
            return f"{greeting} I hope you are well. I wanted to ask about the requested academic matter. Thank you for your time."
        if "proposal exam" in task_l:
            return f"Dear {dr_name}, I hope you are well. I am writing to ask about scheduling my proposal exam. Thank you for your time."
        return f"Hello, I am writing to ask about the academic matter mentioned in the task."

    if "vegetarian" in task_l or "dinner" in task_l:
        if _has(mem_l, "vegetarian"):
            return "Choose a vegetarian-friendly dinner option and avoid meat-based dishes."
        return "Choose a dinner option that fits the current request."

    if "python" in task_l and "function" in task_l:
        return "Define a small, readable Python function and include a short example call."

    if domain == "email_or_message_writing":
        return "Here is a concise message tailored to the current request."

    return f"Completed task: {instruction}"


def bag_of_words_score(query: str, document: str) -> float:
    query_counts = Counter(tokenize(query))
    doc_counts = Counter(tokenize(document))
    if not query_counts or not doc_counts:
        return 0.0
    score = 0.0
    for token, count in query_counts.items():
        score += min(count, doc_counts.get(token, 0))
    return score / sum(query_counts.values())
