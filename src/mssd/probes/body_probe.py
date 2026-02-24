"""Body probe: detects mean/variance shift using MMD or KS test."""

import numpy as np
from scipy.stats import ks_2samp
from .base_probe import BaseProbe
from .kernels import rbf_mmd_squared, median_heuristic_bandwidth


class BodyProbe(BaseProbe):
    """Detects mean/variance shift (body of the distribution).

    Strategy: MMD with RBF kernel for d > 4, KS test for d <= 4.
    For d <= 4, we run KS on each dimension and take the max statistic.
    """

    def __init__(self, bandwidth: float = None):
        super().__init__(name="body")
        self.bandwidth = bandwidth

    def compute_statistic(self, ref: np.ndarray, test: np.ndarray) -> float:
        obs_dim = ref.shape[1]

        if obs_dim <= 4:
            max_ks = 0.0
            for d in range(obs_dim):
                stat, _ = ks_2samp(ref[:, d], test[:, d])
                max_ks = max(max_ks, stat)
            return max_ks
        else:
            bw = self.bandwidth or median_heuristic_bandwidth(
                np.concatenate([ref, test], axis=0)
            )
            return rbf_mmd_squared(ref, test, bandwidth=bw)
