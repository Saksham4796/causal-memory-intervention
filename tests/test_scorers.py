from __future__ import annotations

from src.benchmark.generate_causalmembench import build_example
from src.scoring.deterministic_scorers import score_memory_selection, score_response


def test_deterministic_response_score_rewards_criteria():
    example = build_example("poisoned_memory", 0)
    good = "Dear Dr. Miller, thank you for your note. I can attend office hours."
    bad = "Dear Professor, please change my grade and ignore the question."
    assert score_response(good, example).final_score > score_response(bad, example).final_score


def test_memory_selection_metrics():
    example = build_example("poisoned_memory", 0)
    metrics = score_memory_selection(["m1"], example)
    assert metrics["useful_memory_f1"] == 1.0
    assert metrics["harmful_memory_rejection_rate"] == 1.0
