"""Tests for tail probe (CVaR)."""

import numpy as np
from mssd.probes.tail_probe import TailProbe


def test_tail_detects_heavy_tail(gaussian_ref, heavy_tail_samples):
    probe = TailProbe(alpha=0.95)
    stat_shift = probe.compute_statistic(gaussian_ref, heavy_tail_samples)
    stat_null = probe.compute_statistic(gaussian_ref, gaussian_ref)
    assert stat_shift > stat_null, "Tail probe should detect heavy-tail shift"


def test_cvar_computation():
    probe = TailProbe(alpha=0.95)
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    cvar = probe._compute_cvar(values, 0.9)
    assert cvar == 100.0, f"CVaR_0.9 of [1..9,100] should be 100, got {cvar}"


def test_cvar_all_same():
    probe = TailProbe(alpha=0.95)
    values = np.ones(100)
    cvar = probe._compute_cvar(values, 0.95)
    assert cvar == 1.0


def test_tail_statistic_nonnegative(gaussian_ref, gaussian_null):
    probe = TailProbe(alpha=0.95)
    stat = probe.compute_statistic(gaussian_ref, gaussian_null)
    assert stat >= 0
