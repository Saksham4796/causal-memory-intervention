from __future__ import annotations

from pathlib import Path

from .compute_results import compute_results
from .make_figures import make_figures
from .make_tables import make_tables
from .qualitative_analysis import make_qualitative_examples


def make_all_artifacts(run_dir: str | Path) -> None:
    compute_results(run_dir)
    make_tables(run_dir)
    make_figures(run_dir)
    make_qualitative_examples(run_dir)
