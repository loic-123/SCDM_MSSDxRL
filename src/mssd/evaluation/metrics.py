"""Core evaluation metrics: ADD, FAR, probe discrimination accuracy."""

import numpy as np
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class TrialResult:
    """Result from a single experimental trial."""

    env_name: str
    shift_type: str  # "body", "tail", "structure", "none"
    severity: float
    trial_id: int
    seed: int

    # MSSD results
    mssd_alarm_fired: bool
    mssd_alarm_step: Optional[int]
    mssd_diagnosed_probe: Optional[str]
    mssd_log_wealth: dict  # probe_name -> list

    # Baseline results
    baseline_alarm_fired: bool
    baseline_alarm_step: Optional[int]

    # Metadata
    shift_injection_step: int
    total_steps: int


def compute_add(results: List[TrialResult]) -> float:
    """Average Detection Delay: mean(alarm_step - shift_injection_step)
    over trials where alarm fired."""
    delays = []
    for r in results:
        if r.mssd_alarm_fired and r.mssd_alarm_step is not None:
            delay = r.mssd_alarm_step - r.shift_injection_step
            delays.append(max(delay, 0))
    if not delays:
        return float("inf")
    return float(np.mean(delays))


def compute_far(no_shift_results: List[TrialResult]) -> float:
    """False Alarm Rate: fraction of no-shift trials where alarm fired."""
    if not no_shift_results:
        return 0.0
    false_alarms = sum(1 for r in no_shift_results if r.mssd_alarm_fired)
    return false_alarms / len(no_shift_results)


def compute_discrimination_accuracy(results: List[TrialResult]) -> float:
    """Probe discrimination: fraction of trials where first-firing probe
    matches the injected shift type."""
    correct = 0
    total = 0
    for r in results:
        if r.mssd_alarm_fired and r.mssd_diagnosed_probe is not None:
            total += 1
            if r.mssd_diagnosed_probe == r.shift_type:
                correct += 1
    if total == 0:
        return 0.0
    return correct / total


def compute_baseline_add(results: List[TrialResult]) -> float:
    """ADD for the global MMD baseline."""
    delays = []
    for r in results:
        if r.baseline_alarm_fired and r.baseline_alarm_step is not None:
            delay = r.baseline_alarm_step - r.shift_injection_step
            delays.append(max(delay, 0))
    if not delays:
        return float("inf")
    return float(np.mean(delays))
