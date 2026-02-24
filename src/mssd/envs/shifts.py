"""Shift injection functions for all 6 conditions (3 types x 2 environments).

Shifts operate on observations post-wrapper. The agent still acts on the
original observation; the monitor sees the shifted version. This models
sensor degradation or environment change.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class ShiftConfig:
    shift_type: str  # "body", "tail", "structure"
    severity: float  # 0.0 = no shift, higher = stronger
    env_name: str  # "cliffwalking" or "cartpole"


class ShiftInjector:
    """Applies observation-space shifts to a stream of observations."""

    def __init__(self, config: ShiftConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._step = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Apply shift to a single observation."""
        self._step += 1
        method = getattr(self, f"_{self.config.env_name}_{self.config.shift_type}")
        return method(obs, self.config.severity)

    # -- CliffWalking shifts --

    def _cliffwalking_body(self, obs: np.ndarray, severity: float) -> np.ndarray:
        """Coordinate offset: add [severity, severity, 0] to (row, col, cliff_dist)."""
        return obs + np.array([severity, severity, 0.0], dtype=np.float32)

    def _cliffwalking_tail(self, obs: np.ndarray, severity: float) -> np.ndarray:
        """Hazard state injection: with prob p=0.03, replace obs with extreme value."""
        if self.rng.random() < 0.03:
            return np.array(
                [3.0, self.rng.integers(1, 11), 0.0], dtype=np.float32
            ) * severity
        return obs

    def _cliffwalking_structure(
        self, obs: np.ndarray, severity: float
    ) -> np.ndarray:
        """Feature decorrelation: break correlation between col and cliff_distance."""
        noisy = obs.copy()
        noisy[2] = obs[2] + self.rng.normal(0, severity)
        return noisy

    # -- CartPole shifts --

    def _cartpole_body(self, obs: np.ndarray, severity: float) -> np.ndarray:
        """Pole mass drift: scale theta and theta_dot (indices 2, 3)."""
        shifted = obs.copy()
        shifted[2] *= 1.0 + severity
        shifted[3] *= 1.0 + severity
        return shifted

    def _cartpole_tail(self, obs: np.ndarray, severity: float) -> np.ndarray:
        """Rare extreme-angle injection: with prob 0.03, push theta toward boundary."""
        if self.rng.random() < 0.03:
            shifted = obs.copy()
            sign = self.rng.choice([-1.0, 1.0])
            shifted[2] = sign * 0.20 * severity
            shifted[3] = sign * 2.0 * severity
            return shifted
        return obs

    def _cartpole_structure(self, obs: np.ndarray, severity: float) -> np.ndarray:
        """Break x-theta correlation: add independent noise to x (index 0)."""
        shifted = obs.copy()
        shifted[0] += self.rng.normal(0, severity)
        return shifted
