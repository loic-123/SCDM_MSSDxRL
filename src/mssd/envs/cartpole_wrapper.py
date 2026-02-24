"""Thin Gymnasium wrapper for CartPole-v1: ensures float32 dtype."""

import numpy as np
from gymnasium import ObservationWrapper


class CartPoleLogged(ObservationWrapper):
    """Thin wrapper: ensures float32 for consistent dtype across environments."""

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32)
