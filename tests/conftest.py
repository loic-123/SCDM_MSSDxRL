"""Shared test fixtures for MSSD test suite."""

import pytest
import numpy as np
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def gaussian_ref(rng):
    """200 samples from N(0, I_3)."""
    return rng.standard_normal((200, 3)).astype(np.float32)


@pytest.fixture
def gaussian_shifted(rng):
    """200 samples from N(1, I_3) — body shift."""
    return (rng.standard_normal((200, 3)) + 1.0).astype(np.float32)


@pytest.fixture
def gaussian_null(rng):
    """200 more samples from N(0, I_3) — no shift."""
    return rng.standard_normal((200, 3)).astype(np.float32)


@pytest.fixture
def heavy_tail_samples():
    """200 samples from t-distribution (df=3) — tail shift."""
    from scipy.stats import t as t_dist
    return t_dist.rvs(df=3, size=(200, 3), random_state=12345).astype(np.float32)


@pytest.fixture
def correlated_samples(rng):
    """200 samples with correlation between dims 0 and 2."""
    x = rng.standard_normal((200, 3)).astype(np.float32)
    x[:, 2] = x[:, 0] + 0.1 * rng.standard_normal(200).astype(np.float32)
    return x


@pytest.fixture
def decorrelated_samples(rng):
    """200 samples with broken correlation structure."""
    x = rng.standard_normal((200, 3)).astype(np.float32)
    x[:, 2] = rng.standard_normal(200).astype(np.float32)
    return x
