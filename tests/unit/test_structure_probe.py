"""Tests for structure probe (Frobenius norm of correlation difference)."""

import numpy as np
from mssd.probes.structure_probe import StructureProbe


def test_structure_detects_decorrelation(correlated_samples, decorrelated_samples):
    probe = StructureProbe()
    stat_shift = probe.compute_statistic(correlated_samples, decorrelated_samples)
    stat_null = probe.compute_statistic(
        correlated_samples, correlated_samples
    )
    assert stat_shift > stat_null, "Structure probe should detect decorrelation"


def test_structure_zero_on_same_data(gaussian_ref):
    probe = StructureProbe()
    stat = probe.compute_statistic(gaussian_ref, gaussian_ref)
    assert stat < 0.01, f"Same data should give ~0, got {stat}"


def test_structure_statistic_nonnegative(gaussian_ref, gaussian_null):
    probe = StructureProbe()
    stat = probe.compute_statistic(gaussian_ref, gaussian_null)
    assert stat >= 0


def test_safe_corrcoef_handles_constant_column():
    probe = StructureProbe()
    X = np.random.randn(100, 3).astype(np.float32)
    X[:, 1] = 5.0  # constant column
    corr = probe._safe_corrcoef(X)
    assert corr.shape == (3, 3)
    assert np.all(np.isfinite(corr))
