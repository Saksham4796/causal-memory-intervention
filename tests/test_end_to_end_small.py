from __future__ import annotations

from pathlib import Path

from src.analysis.make_artifacts import make_all_artifacts
from src.benchmark.generate_causalmembench import generate_examples, write_dataset
from src.benchmark.validate_dataset import validate_file
from src.experiments.run_experiment import run_experiment


def test_end_to_end_small(tmp_path):
    dataset_path = tmp_path / "small.jsonl"
    write_dataset(generate_examples(5), dataset_path)
    report = validate_file(dataset_path)
    assert report["num_errors"] == 0
    run_dir = tmp_path / "run"
    run_experiment(
        "config/default.yaml",
        dataset_path,
        max_examples=5,
        run_dir=run_dir,
        skip_llm_judge=True,
        deterministic_only=True,
    )
    assert (run_dir / "metrics_by_agent.csv").exists()
    assert (run_dir / "memory_selection_metrics.csv").exists()
    assert (run_dir / "causal_utility_diagnostics.csv").exists()
    make_all_artifacts(run_dir)
    assert Path("outputs/paper_ready/table_main_results.tex").exists()
    assert Path("outputs/paper_ready/fig_main_task_success.png").exists()
