from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.io import ensure_dir


def flatten_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    scores = prediction.get("scores", {})
    memory_metrics = scores.get("memory_metrics", {})
    row = {
        "example_id": prediction.get("example_id"),
        "task_family": prediction.get("task_family"),
        "agent_name": prediction.get("agent_name"),
        "task_score": scores.get("task_score", scores.get("final_score", 0.0)),
        "deterministic_score": scores.get("deterministic_score", 0.0),
        "llm_judge_score": scores.get("llm_judge_score"),
        "passes": float(bool(scores.get("passes", False))),
        "num_retrieved_memories": len(prediction.get("retrieved_memory_ids", [])),
        "num_selected_memories": len(prediction.get("selected_memory_ids", [])),
        "latency_seconds": prediction.get("latency_seconds", 0.0),
        "cost_usd": prediction.get("cost_usd", prediction.get("estimated_cost_usd", 0.0)),
    }
    row.update(memory_metrics)
    usage = prediction.get("token_usage", {})
    row["input_tokens"] = usage.get("input_tokens", 0)
    row["output_tokens"] = usage.get("output_tokens", 0)
    row["total_tokens"] = usage.get("total_tokens", 0)
    return row


def aggregate_metrics(predictions: list[dict[str, Any]], output_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    rows = [flatten_prediction(prediction) for prediction in predictions]
    df = pd.DataFrame(rows)
    if df.empty:
        metrics_by_agent = pd.DataFrame()
        metrics_by_family = pd.DataFrame()
        memory_metrics = pd.DataFrame()
    else:
        mean_cols = [
            "task_score",
            "deterministic_score",
            "passes",
            "useful_memory_precision",
            "useful_memory_recall",
            "useful_memory_f1",
            "harmful_memory_rejection_rate",
            "irrelevant_memory_rejection_rate",
            "outdated_memory_rejection_rate",
            "poisoned_memory_adoption_rate",
            "context_dependent_memory_accuracy",
            "num_retrieved_memories",
            "num_selected_memories",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "latency_seconds",
        ]
        present = [col for col in mean_cols if col in df.columns]
        metrics_by_agent = df.groupby("agent_name", as_index=False)[present].mean(numeric_only=True)
        metrics_by_agent = metrics_by_agent.rename(columns={"passes": "task_success_rate"})
        metrics_by_family = df.groupby(["agent_name", "task_family"], as_index=False)[present].mean(numeric_only=True)
        metrics_by_family = metrics_by_family.rename(columns={"passes": "task_success_rate"})
        memory_cols = [
            "useful_memory_precision",
            "useful_memory_recall",
            "useful_memory_f1",
            "harmful_memory_rejection_rate",
            "irrelevant_memory_rejection_rate",
            "outdated_memory_rejection_rate",
            "poisoned_memory_adoption_rate",
            "context_dependent_memory_accuracy",
        ]
        memory_metrics = df.groupby("agent_name", as_index=False)[[col for col in memory_cols if col in df.columns]].mean(numeric_only=True)

    causal_diagnostics = causal_utility_diagnostics(predictions)
    if output_dir is not None:
        output_dir = Path(output_dir)
        ensure_dir(output_dir)
        metrics_by_agent.to_csv(output_dir / "metrics_by_agent.csv", index=False)
        metrics_by_family.to_csv(output_dir / "metrics_by_task_family.csv", index=False)
        memory_metrics.to_csv(output_dir / "memory_selection_metrics.csv", index=False)
        causal_diagnostics.to_csv(output_dir / "causal_utility_diagnostics.csv", index=False)
    return {
        "metrics_by_agent": metrics_by_agent,
        "metrics_by_task_family": metrics_by_family,
        "memory_selection_metrics": memory_metrics,
        "causal_utility_diagnostics": causal_diagnostics,
    }


def causal_utility_diagnostics(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for prediction in predictions:
        diagnostics = prediction.get("raw_model_outputs", {}).get("cmi_diagnostics", {})
        for memory_id, diag in diagnostics.items():
            rows.append(
                {
                    "example_id": prediction.get("example_id"),
                    "task_family": prediction.get("task_family"),
                    "agent_name": prediction.get("agent_name"),
                    "memory_id": memory_id,
                    "s_no": diag.get("s_no"),
                    "s_with": diag.get("s_with"),
                    "s_perturbed": diag.get("s_perturbed"),
                    "utility": diag.get("utility"),
                    "stability": diag.get("stability"),
                    "selected": diag.get("selected", False),
                    "label": diag.get("label"),
                }
            )
    return pd.DataFrame(rows)


def cost_summary(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total_cost = sum(float(prediction.get("cost_usd", prediction.get("estimated_cost_usd", 0.0)) or 0.0) for prediction in predictions)
    total_tokens = sum(int(prediction.get("token_usage", {}).get("total_tokens", 0) or 0) for prediction in predictions)
    return {
        "num_predictions": len(predictions),
        "total_cost_usd": total_cost,
        "total_tokens": total_tokens,
        "average_cost_usd": total_cost / len(predictions) if predictions else 0.0,
    }
