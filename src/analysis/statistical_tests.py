from __future__ import annotations

import random
from typing import Iterable

import numpy as np


def paired_bootstrap_ci(a: Iterable[float], b: Iterable[float], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    a_arr = np.array(list(a), dtype=float)
    b_arr = np.array(list(b), dtype=float)
    if len(a_arr) != len(b_arr) or len(a_arr) == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0, "effect_size": 0.0}
    diffs = a_arr - b_arr
    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boot.append(float(np.mean(sample)))
    mean_diff = float(np.mean(diffs))
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    p_value = float(min((np.array(boot) <= 0).mean(), (np.array(boot) >= 0).mean()) * 2)
    effect_size = float(mean_diff / (np.std(diffs) + 1e-9))
    return {
        "mean_diff": mean_diff,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": min(1.0, p_value),
        "effect_size": effect_size,
    }
