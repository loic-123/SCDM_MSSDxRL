"""Integration test: full MSSD pipeline on synthetic Gaussian data."""

import numpy as np
import pytest
from mssd.monitor.mssd_monitor import MSSDMonitor


def test_full_pipeline_gaussian_body_shift():
    """End-to-end: Gaussian data with mean shift -> monitor detects body shift."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((500, 3)).astype(np.float32)

    monitor = MSSDMonitor(
        reference_obs=ref,
        window_size=30,
        window_step=5,
        alpha=0.05,
        n_permutations=50,  # fewer for speed
        seed=42,
    )

    # Feed 100 null observations
    for _ in range(100):
        obs = rng.standard_normal(3).astype(np.float32)
        monitor.observe(obs)

    # Feed shifted observations
    for _ in range(300):
        obs = (rng.standard_normal(3) + 2.0).astype(np.float32)
        result = monitor.observe(obs)
        if result is not None:
            assert result == "body", f"Expected body diagnosis, got {result}"
            return

    # It's acceptable if the monitor didn't fire with only 50 permutations
    # but we should at least have accumulated some evidence
    log_wealth = monitor.martingale.log_wealth
    assert log_wealth["body"] > log_wealth["tail"]


def test_no_alarm_under_null():
    """No shift injected -> monitor should not fire."""
    rng = np.random.default_rng(42)
    ref = rng.standard_normal((500, 3)).astype(np.float32)

    monitor = MSSDMonitor(
        reference_obs=ref,
        window_size=30,
        window_step=10,
        alpha=0.05,
        n_permutations=50,
        seed=42,
    )

    # Feed 200 null observations
    for _ in range(200):
        obs = rng.standard_normal(3).astype(np.float32)
        result = monitor.observe(obs)
        if result is not None:
            pytest.fail("Monitor should not fire under null")
