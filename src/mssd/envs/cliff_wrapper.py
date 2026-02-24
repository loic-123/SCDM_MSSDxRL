"""Gymnasium wrapper for CliffWalking: converts int state to (row, col, cliff_distance)."""

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box


class CliffWalkingContinuousObs(ObservationWrapper):
    """Wraps CliffWalking: converts int state -> np.array([row, col, cliff_distance]).

    cliff_distance = min Manhattan distance from current cell to nearest cliff cell.
    The cliff cells are at row=3, col=1..10.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.nrow = 4
        self.ncol = 12
        self.observation_space = Box(
            low=np.array([0.0, 0.0, 0.0]),
            high=np.array([3.0, 11.0, 13.0]),
            dtype=np.float32,
        )
        self._cliff_cells = [(3, c) for c in range(1, 11)]

    def observation(self, obs: int) -> np.ndarray:
        row = obs // self.ncol
        col = obs % self.ncol
        cliff_dist = min(
            abs(row - cr) + abs(col - cc) for cr, cc in self._cliff_cells
        )
        return np.array([row, col, cliff_dist], dtype=np.float32)
