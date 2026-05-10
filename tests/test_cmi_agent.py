from __future__ import annotations

from src.agents.cmi_agent import CMIAgent
from src.benchmark.generate_causalmembench import build_example
from src.scoring.deterministic_scorers import score_agent_output


def test_cmi_rejects_poisoned_memory():
    example = build_example("poisoned_memory", 0)
    output = CMIAgent().answer(example)
    assert "m2" not in output.selected_memory_ids
    scores = score_agent_output(output.final_answer, output.selected_memory_ids, example)
    assert scores["memory_metrics"]["harmful_memory_rejection_rate"] == 1.0
