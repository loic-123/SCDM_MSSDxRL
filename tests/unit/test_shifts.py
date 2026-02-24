"""Tests for shift injection functions."""

import numpy as np
from mssd.envs.shifts import ShiftInjector, ShiftConfig


def test_body_shift_cliffwalking():
    rng = np.random.default_rng(42)
    cfg = ShiftConfig(shift_type="body", severity=1.0, env_name="cliffwalking")
    injector = ShiftInjector(cfg, rng)
    obs = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shifted = injector(obs)
    np.testing.assert_array_almost_equal(shifted, [2.0, 3.0, 3.0])


def test_tail_shift_cliffwalking_mostly_unchanged():
    rng = np.random.default_rng(42)
    cfg = ShiftConfig(shift_type="tail", severity=1.0, env_name="cliffwalking")
    injector = ShiftInjector(cfg, rng)
    obs = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    # Most observations should be unchanged (97% probability)
    unchanged_count = sum(
        1 for _ in range(100) if np.array_equal(injector(obs), obs)
    )
    assert unchanged_count > 80  # should be ~97


def test_structure_shift_cliffwalking():
    rng = np.random.default_rng(42)
    cfg = ShiftConfig(shift_type="structure", severity=1.0, env_name="cliffwalking")
    injector = ShiftInjector(cfg, rng)
    obs = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    shifted = injector(obs)
    # Only dim 2 (cliff_dist) should change
    assert shifted[0] == obs[0]
    assert shifted[1] == obs[1]
    assert shifted[2] != obs[2]


def test_body_shift_cartpole():
    rng = np.random.default_rng(42)
    cfg = ShiftConfig(shift_type="body", severity=0.5, env_name="cartpole")
    injector = ShiftInjector(cfg, rng)
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    shifted = injector(obs)
    # Dims 0, 1 unchanged; dims 2, 3 scaled by 1.5
    assert shifted[0] == obs[0]
    assert shifted[1] == obs[1]
    np.testing.assert_almost_equal(shifted[2], 0.3 * 1.5)
    np.testing.assert_almost_equal(shifted[3], 0.4 * 1.5)


def test_structure_shift_cartpole():
    rng = np.random.default_rng(42)
    cfg = ShiftConfig(shift_type="structure", severity=1.0, env_name="cartpole")
    injector = ShiftInjector(cfg, rng)
    obs = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    shifted = injector(obs)
    # Only dim 0 (x) should change
    assert shifted[0] != obs[0]
    assert shifted[1] == obs[1]
    assert shifted[2] == obs[2]
    assert shifted[3] == obs[3]
