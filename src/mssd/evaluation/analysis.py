"""Aggregation and analysis across experimental trials."""

import numpy as np
from typing import List, Dict, Tuple
from .metrics import (
    TrialResult,
    compute_add,
    compute_far,
    compute_discrimination_accuracy,
    compute_baseline_add,
)


def aggregate_by_condition(
    results: List[TrialResult],
) -> Dict[Tuple[str, str, float], List[TrialResult]]:
    """Group results by (env_name, shift_type, severity)."""
    groups: Dict[Tuple[str, str, float], List[TrialResult]] = {}
    for r in results:
        key = (r.env_name, r.shift_type, r.severity)
        groups.setdefault(key, []).append(r)
    return groups


def compute_summary_table(
    results: List[TrialResult],
    no_shift_results: List[TrialResult],
) -> dict:
    """Compute full summary: ADD, FAR, discrimination for each condition.

    Returns:
        Dict with keys (env_name, shift_type, severity) -> metrics dict
    """
    grouped = aggregate_by_condition(results)
    summary = {}

    for key, group in grouped.items():
        far = compute_far(no_shift_results)
        add_mssd = compute_add(group)
        add_baseline = compute_baseline_add(group)
        disc = compute_discrimination_accuracy(group)

        add_ci = _bootstrap_ci(
            [
                r.mssd_alarm_step - r.shift_injection_step
                for r in group
                if r.mssd_alarm_fired and r.mssd_alarm_step is not None
            ]
        )

        summary[key] = {
            "add_mssd": add_mssd,
            "add_baseline": add_baseline,
            "discrimination": disc,
            "far": far,
            "n_trials": len(group),
            "add_mssd_ci": add_ci,
        }

    return summary


def _bootstrap_ci(
    values: list, n_boot: int = 1000, ci: float = 0.95
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    if not values:
        return (float("inf"), float("inf"))
    arr = np.array(values)
    rng = np.random.default_rng(0)
    boot_means = [
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ]
    lo = np.percentile(boot_means, 100 * (1 - ci) / 2)
    hi = np.percentile(boot_means, 100 * (1 + ci) / 2)
    return (float(lo), float(hi))
