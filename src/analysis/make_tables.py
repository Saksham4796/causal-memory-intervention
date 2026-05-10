from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import ensure_dir


def _write_latex(df: pd.DataFrame, path: Path, columns: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    if columns:
        columns = [col for col in columns if col in df.columns]
        df = df[columns]
    with path.open("w", encoding="utf-8") as handle:
        handle.write(_dataframe_to_latex(df))


def _escape_latex(value: object) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _format_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return _escape_latex(value)


def _dataframe_to_latex(df: pd.DataFrame) -> str:
    if df.empty:
        return "\\begin{tabular}{l}\nNo data \\\\\n\\end{tabular}\n"
    alignment = "l" * len(df.columns)
    lines = [f"\\begin{{tabular}}{{{alignment}}}", "\\toprule"]
    lines.append(" & ".join(_escape_latex(col) for col in df.columns) + r" \\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(_format_value(row[col]) for col in df.columns) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def make_tables(run_dir: str | Path, output_dir: str | Path = "outputs/paper_ready") -> None:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    metrics = pd.read_csv(run_dir / "metrics_by_agent.csv")
    memory = pd.read_csv(run_dir / "memory_selection_metrics.csv")
    causal_path = run_dir / "causal_utility_diagnostics.csv"
    causal = pd.read_csv(causal_path) if causal_path.exists() and causal_path.stat().st_size else pd.DataFrame()

    _write_latex(
        metrics,
        output_dir / "table_main_results.tex",
        ["agent_name", "task_score", "task_success_rate", "useful_memory_f1", "harmful_memory_rejection_rate", "poisoned_memory_adoption_rate", "cost_usd"],
    )
    _write_latex(memory, output_dir / "table_memory_selection.tex")
    _write_latex(
        metrics[["agent_name", "task_score", "poisoned_memory_adoption_rate", "harmful_memory_rejection_rate"]],
        output_dir / "table_poisoning_robustness.tex",
    )
    ablation_path = run_dir / "ablation_results.csv"
    if ablation_path.exists():
        ablations = pd.read_csv(ablation_path)
    else:
        ablations = metrics[metrics["agent_name"].str.startswith("cmi", na=False)]
    _write_latex(ablations, output_dir / "table_ablations.tex")
    _write_latex(metrics[["agent_name", "task_score", "cost_usd", "total_tokens"]], output_dir / "table_cost_accuracy_tradeoff.tex")
    if "task_family" in metrics.columns:
        dataset_stats = metrics[["agent_name", "task_score"]]
    else:
        dataset_stats = pd.DataFrame({"metric": ["num_agents"], "value": [len(metrics)]})
    _write_latex(dataset_stats, output_dir / "table_dataset_statistics.tex")

    if not causal.empty:
        grouped = causal.groupby(["agent_name", "label"], as_index=False)[["utility", "stability"]].mean(numeric_only=True)
        _write_latex(grouped, output_dir / "table_causal_diagnostics.tex")
