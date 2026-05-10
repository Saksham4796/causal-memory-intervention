from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data


def write_yaml(data: dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Any, path: str | Path, indent: int = 2) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, ensure_ascii=False)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            if hasattr(record, "model_dump"):
                payload = record.model_dump()
            elif hasattr(record, "dict"):
                payload = record.dict()
            else:
                payload = record
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    if load_dotenv is not None:
        load_dotenv()
    config = read_yaml(path)
    for section in ("openai", "experiment"):
        config.setdefault(section, {})
    env_overrides = {
        "OPENAI_AGENT_MODEL": ("openai", "agent_model"),
        "OPENAI_JUDGE_MODEL": ("openai", "judge_model"),
        "OPENAI_DATASET_MODEL": ("openai", "dataset_model"),
        "OPENAI_EMBEDDING_MODEL": ("openai", "embedding_model"),
        "OPENAI_API_URL": ("openai", "api_url"),
        "OPENAI_BASE_URL": ("openai", "api_url"),
    }
    for env_name, (section, key) in env_overrides.items():
        value = os.getenv(env_name)
        if value:
            config.setdefault(section, {})[key] = value
    use_api = os.getenv("OPENAI_USE_API")
    if use_api is not None:
        config.setdefault("openai", {})["use_api"] = use_api.lower() in {"1", "true", "yes", "on"}
    return config


def latest_run_dir(base: str | Path = "outputs/runs") -> Path | None:
    run_root = Path(base)
    if not run_root.exists():
        return None
    dirs = [p for p in run_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.stat().st_mtime)[-1]


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}
