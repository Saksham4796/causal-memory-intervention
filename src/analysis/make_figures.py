from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.io import ensure_dir


def _save_bar(df: pd.DataFrame, x: str, y: str, path: Path, title: str, ylabel: str) -> None:
    plt.figure(figsize=(9, 5))
    if df.empty or x not in df or y not in df:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        sns.barplot(data=df, x=x, y=y)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def make_figures(run_dir: str | Path, output_dir: str | Path = "outputs/paper_ready") -> None:
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    metrics = pd.read_csv(run_dir / "metrics_by_agent.csv")
    family_path = run_dir / "metrics_by_task_family.csv"
    family = pd.read_csv(family_path) if family_path.exists() else pd.DataFrame()
    causal_path = run_dir / "causal_utility_diagnostics.csv"
    causal = pd.read_csv(causal_path) if causal_path.exists() and causal_path.stat().st_size else pd.DataFrame()

    _save_bar(metrics, "agent_name", "task_success_rate", output_dir / "fig_main_task_success.png", "Task Success Rate", "Success Rate")
    _save_bar(metrics, "agent_name", "useful_memory_f1", output_dir / "fig_memory_selection_f1.png", "Useful Memory F1", "F1")
    _save_bar(metrics, "agent_name", "poisoned_memory_adoption_rate", output_dir / "fig_poisoned_memory_adoption.png", "Poisoned Memory Adoption", "Adoption Rate")

    plt.figure(figsize=(7, 5))
    if not metrics.empty and {"cost_usd", "task_score", "agent_name"} <= set(metrics.columns):
        sns.scatterplot(data=metrics, x="cost_usd", y="task_score", hue="agent_name", s=90)
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    plt.title("Cost vs Accuracy")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_cost_vs_accuracy.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    if not causal.empty and "utility" in causal:
        sns.histplot(data=causal, x="utility", hue="label", kde=True)
    else:
        plt.text(0.5, 0.5, "No CMI diagnostics", ha="center", va="center")
    plt.title("CMI Utility Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_utility_distribution.png", dpi=200)
    plt.close()

    threshold_df = causal.copy()
    plt.figure(figsize=(8, 5))
    if not threshold_df.empty and "utility" in threshold_df:
        xs = sorted(threshold_df["utility"].dropna().unique())
        ys = [(threshold_df["utility"] >= x).mean() for x in xs]
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Utility threshold")
        plt.ylabel("Acceptance rate")
    else:
        plt.text(0.5, 0.5, "No threshold data", ha="center", va="center")
    plt.title("Threshold Sweep")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_threshold_sweep.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    if not family.empty and {"task_family", "task_score", "agent_name"} <= set(family.columns):
        sns.barplot(data=family, x="task_family", y="task_score", hue="agent_name")
        plt.xticks(rotation=35, ha="right")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    else:
        plt.text(0.5, 0.5, "No family data", ha="center", va="center")
    plt.title("Task Family Breakdown")
    plt.tight_layout()
    plt.savefig(output_dir / "fig_task_family_breakdown.png", dpi=200)
    plt.close()
