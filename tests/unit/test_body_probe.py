"""Tests for body probe (MMD/KS)."""

import numpy as np
from mssd.probes.body_probe import BodyProbe


def test_body_detects_mean_shift(gaussian_ref, gaussian_shifted):
    probe = BodyProbe()
    stat_shift = probe.compute_statistic(gaussian_ref, gaussian_shifted)
    stat_null = probe.compute_statistic(gaussian_ref, gaussian_ref[:100])
    assert stat_shift > stat_null, "Body probe should detect mean shift"


def test_body_evalue_under_shift(gaussian_ref, gaussian_shifted):
    probe = BodyProbe()
    rng = np.random.default_rng(42)
    e_val = probe.to_evalue(
        gaussian_ref, gaussian_shifted, n_permutations=100, rng=rng
    )
    assert e_val > 5, f"E-value under shift should be large, got {e_val}"


def test_body_uses_ks_for_low_dim(gaussian_ref):
    """With 3D data (d<=4), body probe should use KS (returns value in [0, 1])."""
    probe = BodyProbe()
    stat = probe.compute_statistic(gaussian_ref[:100], gaussian_ref[100:])
    assert 0 <= stat <= 1, f"KS stat should be in [0,1], got {stat}"


def test_body_statistic_nonnegative(gaussian_ref, gaussian_null):
    probe = BodyProbe()
    stat = probe.compute_statistic(gaussian_ref, gaussian_null)
    assert stat >= 0
