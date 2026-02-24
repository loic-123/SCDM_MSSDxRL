"""Tests for evaluation metrics."""

from mssd.evaluation.metrics import (
    TrialResult,
    compute_add,
    compute_far,
    compute_discrimination_accuracy,
)


def _make_trial(
    shift_type="body",
    alarm_fired=True,
    alarm_step=250,
    diagnosed_probe="body",
    shift_injection_step=200,
):
    return TrialResult(
        env_name="CartPole-v1",
        shift_type=shift_type,
        severity=0.5,
        trial_id=0,
        seed=42,
        mssd_alarm_fired=alarm_fired,
        mssd_alarm_step=alarm_step,
        mssd_diagnosed_probe=diagnosed_probe,
        mssd_log_wealth={},
        baseline_alarm_fired=False,
        baseline_alarm_step=None,
        shift_injection_step=shift_injection_step,
        total_steps=500,
    )


def test_compute_add():
    results = [
        _make_trial(alarm_step=250),
        _make_trial(alarm_step=300),
    ]
    add = compute_add(results)
    assert add == 75.0  # mean of (250-200, 300-200) = mean(50, 100)


def test_compute_add_no_alarms():
    results = [_make_trial(alarm_fired=False, alarm_step=None)]
    add = compute_add(results)
    assert add == float("inf")


def test_compute_far():
    no_shift = [
        _make_trial(shift_type="none", alarm_fired=False, alarm_step=None),
        _make_trial(shift_type="none", alarm_fired=True, alarm_step=100),
    ]
    far = compute_far(no_shift)
    assert far == 0.5


def test_compute_far_no_false_alarms():
    no_shift = [
        _make_trial(shift_type="none", alarm_fired=False, alarm_step=None),
        _make_trial(shift_type="none", alarm_fired=False, alarm_step=None),
    ]
    far = compute_far(no_shift)
    assert far == 0.0


def test_discrimination_accuracy():
    results = [
        _make_trial(shift_type="body", diagnosed_probe="body"),   # correct
        _make_trial(shift_type="body", diagnosed_probe="tail"),   # wrong
        _make_trial(shift_type="tail", diagnosed_probe="tail"),   # correct
    ]
    acc = compute_discrimination_accuracy(results)
    assert abs(acc - 2.0 / 3.0) < 1e-6
