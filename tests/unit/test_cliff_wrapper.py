"""Tests for CliffWalking continuous observation wrapper."""

import numpy as np
import gymnasium as gym
from mssd.envs.cliff_wrapper import CliffWalkingContinuousObs


def test_observation_shape():
    env = CliffWalkingContinuousObs(gym.make("CliffWalking-v1"))
    obs, _ = env.reset()
    assert obs.shape == (3,)
    assert obs.dtype == np.float32


def test_observation_range():
    env = CliffWalkingContinuousObs(gym.make("CliffWalking-v1"))
    obs, _ = env.reset()
    # row in [0, 3], col in [0, 11], cliff_dist >= 0
    assert 0 <= obs[0] <= 3
    assert 0 <= obs[1] <= 11
    assert obs[2] >= 0


def test_cliff_distance_at_cliff():
    wrapper = CliffWalkingContinuousObs(gym.make("CliffWalking-v1"))
    # State 37 = row 3, col 1 (first cliff cell)
    obs = wrapper.observation(37)
    assert obs[0] == 3.0
    assert obs[1] == 1.0
    assert obs[2] == 0.0  # on the cliff


def test_cliff_distance_away():
    wrapper = CliffWalkingContinuousObs(gym.make("CliffWalking-v1"))
    # State 0 = row 0, col 0 (top-left corner)
    obs = wrapper.observation(0)
    assert obs[0] == 0.0
    assert obs[1] == 0.0
    assert obs[2] > 0  # not on the cliff


def test_step_returns_correct_obs():
    env = CliffWalkingContinuousObs(gym.make("CliffWalking-v1"))
    obs, _ = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(0)
    assert next_obs.shape == (3,)
    assert next_obs.dtype == np.float32
