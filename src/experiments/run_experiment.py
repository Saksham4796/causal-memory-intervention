from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.agents.cmi_agent import CMIAgent
from src.agents.full_history_agent import FullHistoryAgent
from src.agents.graph_memory_agent import GraphMemoryAgent
from src.agents.no_memory_agent import NoMemoryAgent
from src.agents.reflection_memory_agent import ReflectionMemoryAgent
from src.agents.summary_memory_agent import SummaryMemoryAgent
from src.agents.vector_memory_agent import VectorMemoryAgent
from src.api.openai_client import OpenAIClient
from src.benchmark.load_dataset import load_examples
from src.experiments.cost_tracker import CostTracker
from src.experiments.seed_control import set_seed
from src.scoring.aggregate_metrics import aggregate_metrics, cost_summary
from src.scoring.deterministic_scorers import score_agent_output
from src.scoring.llm_judge import LLMJudge, hybrid_score
from src.utils.io import ensure_dir, load_config, write_json, write_jsonl, write_yaml
from src.utils.logging import setup_logger


AGENT_REGISTRY = {
    "no_memory": NoMemoryAgent,
    "full_history": FullHistoryAgent,
    "vector_memory": VectorMemoryAgent,
    "summary_memory": SummaryMemoryAgent,
    "reflection_memory": ReflectionMemoryAgent,
    "graph_memory": GraphMemoryAgent,
    "cmi": CMIAgent,
}


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        return result.stdout.strip() or None
    except Exception:
        return None


def create_run_dir(base: str | Path = "outputs/runs", prefix: str = "run") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(base)
    for suffix in [""] + [f"_{idx}" for idx in range(1, 1000)]:
        path = root / f"{prefix}_{timestamp}{suffix}"
        if not path.exists():
            ensure_dir(path)
            return path
    raise RuntimeError("Could not allocate a unique run directory")


def make_agent(agent_name: str, config: dict[str, Any], client: OpenAIClient) -> BaseAgent:
    if agent_name.startswith("cmi_"):
        return CMIAgent(config=config, client=client, variant=agent_name.removeprefix("cmi_"))
    cls = AGENT_REGISTRY[agent_name]
    return cls(config=config, client=client)


def run_experiment(
    config_path: str | Path,
    dataset_path: str | Path,
    max_examples: int | None = None,
    agents: list[str] | None = None,
    run_dir: str | Path | None = None,
    dry_run: bool = False,
    skip_llm_judge: bool | None = None,
    deterministic_only: bool | None = None,
    use_cache: bool = True,
) -> Path:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if deterministic_only is not None:
        config.setdefault("experiment", {})["deterministic_only"] = deterministic_only
    if skip_llm_judge is not None:
        config.setdefault("experiment", {})["skip_llm_judge"] = skip_llm_judge
    if max_examples is None:
        max_examples = config.get("experiment", {}).get("max_examples")
    agent_names = agents or config.get("experiment", {}).get("agents", list(AGENT_REGISTRY))
    if dry_run:
        max_examples = min(int(max_examples or 5), 5)

    run_path = Path(run_dir) if run_dir else create_run_dir()
    ensure_dir(run_path)
    logger = setup_logger("run_experiment", run_path / "run.log")
    logger.info("Starting run in %s", run_path)
    config["run"] = {"dataset_path": str(dataset_path), "git_commit": git_commit_hash()}
    write_yaml(config, run_path / "config.yaml")
    if Path(config_path).exists():
        shutil.copyfile(config_path, run_path / "source_config.yaml")

    client = OpenAIClient(
        use_cache=use_cache,
        use_api=bool(config.get("openai", {}).get("use_api", False)),
        cache_dir=".cache/openai",
    )
    examples = load_examples(dataset_path, max_examples=max_examples)
    judge = None
    if not config.get("experiment", {}).get("skip_llm_judge", True):
        judge = LLMJudge(client=client, model=config.get("openai", {}).get("judge_model", "gpt-4.1-mini"))
    cost_tracker = CostTracker()
    predictions: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    predictions_path = run_path / "predictions.jsonl"
    scores_path = run_path / "scores.jsonl"
    with predictions_path.open("w", encoding="utf-8") as pred_handle, scores_path.open("w", encoding="utf-8") as score_handle:
        for example in examples:
            for agent_name in agent_names:
                try:
                    agent = make_agent(agent_name, config, client)
                    output = agent.answer(example)
                    scores = score_agent_output(output.final_answer, output.selected_memory_ids, example)
                    if judge is not None:
                        judge_result = judge.score(output.final_answer, example)
                        output.token_usage = BaseAgent._sum_usage(output.token_usage, judge_result.token_usage)
                        output.estimated_cost_usd += judge_result.estimated_cost_usd
                        scores["llm_judge_score"] = judge_result.task_success_score
                        scores["final_score"] = hybrid_score(
                            scores["deterministic_score"],
                            judge_result.task_success_score,
                            config.get("scoring", {}).get("deterministic_weight", 0.7),
                            config.get("scoring", {}).get("llm_judge_weight", 0.3),
                        )
                        scores["task_score"] = scores["final_score"]
                        scores["passes"] = scores["final_score"] >= 0.7
                        scores["judge"] = judge_result.dict() if hasattr(judge_result, "dict") else judge_result.model_dump()
                    prediction = output.to_prediction(example.task_family, scores)
                    prediction["prompt"] = output.raw_model_outputs.get("prompt", "")
                    predictions.append(prediction)
                    pred_handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    score_handle.write(
                        json.dumps(
                            {
                                "example_id": example.example_id,
                                "task_family": example.task_family,
                                "agent_name": output.agent_name,
                                "scores": scores,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    cost_tracker.add(output.agent_name, example.example_id, output.estimated_cost_usd, output.token_usage)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed example=%s agent=%s", example.example_id, agent_name)
                    failures.append({"example_id": example.example_id, "agent_name": agent_name, "error": str(exc)})

    aggregate_metrics(predictions, output_dir=run_path)
    write_json(cost_summary(predictions) | {"tracker": cost_tracker.summary()}, run_path / "cost_summary.json")
    write_jsonl(failures, run_path / "failed_examples.jsonl")
    logger.info("Completed run with %d predictions and %d failures", len(predictions), len(failures))
    return run_path


def parse_agents(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CausalMemBench experiments.")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--agents", default=None, help="Comma-separated agent names.")
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_llm_judge", action="store_true")
    parser.add_argument("--deterministic_only", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()
    run_path = run_experiment(
        args.config,
        args.dataset,
        max_examples=args.max_examples,
        agents=parse_agents(args.agents),
        run_dir=args.run_dir,
        dry_run=args.dry_run,
        skip_llm_judge=True if args.skip_llm_judge else None,
        deterministic_only=True if args.deterministic_only else None,
        use_cache=not args.no_cache,
    )
    print(run_path)


if __name__ == "__main__":
    main()
