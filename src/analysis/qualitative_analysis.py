from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.io import ensure_dir, read_jsonl


def _case_md(prediction: dict[str, Any]) -> str:
    scores = prediction.get("scores", {})
    return (
        f"## {prediction.get('example_id')} - {prediction.get('agent_name')}\n\n"
        f"Task family: `{prediction.get('task_family')}`\n\n"
        f"Selected memories: `{prediction.get('selected_memory_ids', [])}`\n\n"
        f"Retrieved memories: `{prediction.get('retrieved_memory_ids', [])}`\n\n"
        f"Answer:\n\n{prediction.get('final_answer', prediction.get('response', ''))}\n\n"
        f"Score: `{scores.get('task_score', scores.get('final_score'))}`\n\n"
    )


def make_qualitative_examples(run_dir: str | Path, output_dir: str | Path = "outputs/qualitative_examples") -> None:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    predictions = read_jsonl(run_dir / "predictions.jsonl")

    cmi_cases = [p for p in predictions if p.get("agent_name") == "cmi" and p.get("scores", {}).get("passes")]
    vector_failures = [p for p in predictions if p.get("agent_name") == "vector_memory" and not p.get("scores", {}).get("passes")]
    poisoning = [p for p in predictions if p.get("scores", {}).get("memory_metrics", {}).get("poisoned_memory_adoption_rate", 0) > 0]
    conflict = [p for p in predictions if p.get("task_family") == "conflicting_memories"]
    ablation_failures = [p for p in predictions if str(p.get("agent_name", "")).startswith("cmi_") and not p.get("scores", {}).get("passes")]

    files = {
        "successful_cmi_cases.md": cmi_cases,
        "vector_failure_cases.md": vector_failures,
        "poisoning_cases.md": poisoning,
        "conflict_resolution_cases.md": conflict,
        "ablation_failure_cases.md": ablation_failures,
    }
    for filename, cases in files.items():
        content = f"# {filename.replace('_', ' ').replace('.md', '').title()}\n\n"
        if cases:
            content += "\n".join(_case_md(case) for case in cases[:10])
        else:
            content += "No matching cases in this run.\n"
        (output_dir / filename).write_text(content, encoding="utf-8")
