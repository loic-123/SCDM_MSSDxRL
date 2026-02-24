"""Tests for product martingale."""

import numpy as np
from mssd.martingale.product_martingale import ProductMartingale


def test_martingale_fires_with_large_evalues():
    mart = ProductMartingale(
        probe_names=["body", "tail", "structure"], alpha=0.05
    )
    for _ in range(20):
        fired = mart.update({"body": 5.0, "tail": 1.0, "structure": 1.0})
        if fired:
            break
    assert mart._alarm_fired, "Should fire with consistent large e-values"
    assert mart._alarm_probe == "body", "Body should be diagnosed"


def test_martingale_no_alarm_under_null():
    mart = ProductMartingale(
        probe_names=["body", "tail", "structure"], alpha=0.05
    )
    for _ in range(100):
        fired = mart.update({"body": 1.0, "tail": 1.0, "structure": 1.0})
    assert not fired, "Should not fire with e-values of 1"


def test_martingale_threshold():
    mart = ProductMartingale(probe_names=["p1"], alpha=0.05)
    expected = np.log(20)
    assert abs(mart.threshold - expected) < 1e-6


def test_martingale_reset():
    mart = ProductMartingale(
        probe_names=["body", "tail", "structure"], alpha=0.05
    )
    mart.update({"body": 100.0, "tail": 100.0, "structure": 100.0})
    assert mart._alarm_fired
    mart.reset()
    assert not mart._alarm_fired
    assert mart._step == 0


def test_martingale_get_result():
    mart = ProductMartingale(
        probe_names=["body", "tail", "structure"], alpha=0.05
    )
    mart.update({"body": 2.0, "tail": 1.0, "structure": 1.0})
    result = mart.get_result()
    assert not result.fired
    assert len(result.log_wealth_history["body"]) == 2  # initial 0 + 1 update
    assert len(result.product_log_wealth_history) == 2


def test_martingale_diagnoses_correct_probe():
    mart = ProductMartingale(
        probe_names=["body", "tail", "structure"], alpha=0.05
    )
    # Feed large e-values only for tail
    for _ in range(50):
        fired = mart.update({"body": 1.0, "tail": 10.0, "structure": 1.0})
        if fired:
            break
    assert mart._alarm_probe == "tail"
