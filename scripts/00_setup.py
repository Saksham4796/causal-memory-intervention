from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from src.utils.io import ensure_dir

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))


REQUIRED_PACKAGES = [
    "pydantic",
    "pandas",
    "numpy",
    "sklearn",
    "scipy",
    "matplotlib",
    "seaborn",
    "tqdm",
    "yaml",
    "pytest",
    "networkx",
]

DIRECTORIES = [
    "data/raw",
    "data/generated",
    "data/processed",
    "data/external",
    "outputs/runs",
    "outputs/tables",
    "outputs/figures",
    "outputs/qualitative_examples",
    "outputs/paper_ready",
    ".cache/openai",
]


def main() -> None:
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        raise SystemExit("Python 3.9+ is required.")

    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except Exception:
            missing.append(package)
    if missing:
        print("Missing packages:", ", ".join(missing))
        print("Install with: pip install -r requirements.txt")
    else:
        print("All core packages are importable.")

    for directory in DIRECTORIES:
        ensure_dir(directory)
    print("Created required local directories.")

    env_path = Path(".env")
    if not env_path.exists():
        print("Warning: .env is missing. Live OpenAI calls will be disabled unless environment variables are already set.")
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY is not set. Deterministic fallback mode remains available.")
    print("Setup checks completed.")


if __name__ == "__main__":
    main()
