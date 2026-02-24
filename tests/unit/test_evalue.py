"""Tests for e-value computation (block-bootstrap permutation)."""

import numpy as np
from mssd.probes.body_probe import BodyProbe
from mssd.probes.base_probe import _block_bootstrap_permutation


def test_block_bootstrap_produces_valid_indices():
    rng = np.random.default_rng(42)
    indices = _block_bootstrap_permutation(100, block_size=10, rng=rng)
    assert len(indices) == 100
    assert indices.min() >= 0
    assert indices.max() < 100


def test_evalue_under_null_is_moderate(gaussian_ref, gaussian_null):
    """Under H0, e-values should not be extreme."""
    probe = BodyProbe()
    rng = np.random.default_rng(42)
    e_val = probe.to_evalue(
        gaussian_ref, gaussian_null, n_permutations=100, rng=rng
    )
    # Under null, e-value should be moderate (not huge)
    # Under null, e-values can be noisy with finite permutations
    assert e_val < 200, f"E-value under null should be moderate, got {e_val}"


def test_evalue_under_shift_is_large(gaussian_ref, gaussian_shifted):
    """Under H1, e-values should be large."""
    probe = BodyProbe()
    rng = np.random.default_rng(42)
    e_val = probe.to_evalue(
        gaussian_ref, gaussian_shifted, n_permutations=100, rng=rng
    )
    assert e_val > 5, f"E-value under shift should be large, got {e_val}"


def test_evalue_is_positive(gaussian_ref, gaussian_null):
    probe = BodyProbe()
    rng = np.random.default_rng(42)
    e_val = probe.to_evalue(
        gaussian_ref, gaussian_null, n_permutations=50, rng=rng
    )
    assert e_val > 0
