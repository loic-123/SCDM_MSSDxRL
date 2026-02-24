"""Tail probe: detects rare dangerous states via CVaR difference."""

import numpy as np
from .base_probe import BaseProbe


class TailProbe(BaseProbe):
    """Detects tail shift (rare dangerous states) via CVaR difference.

    Projects observations onto a 1D direction (first principal component),
    then computes the absolute difference in CVaR_alpha between ref and test.
    """

    def __init__(self, alpha: float = 0.95):
        super().__init__(name="tail")
        self.alpha = alpha

    def compute_statistic(self, ref: np.ndarray, test: np.ndarray) -> float:
        pooled = np.concatenate([ref, test], axis=0)
        pooled_centered = pooled - pooled.mean(axis=0)

        _, _, Vt = np.linalg.svd(pooled_centered, full_matrices=False)
        direction = Vt[0]

        ref_proj = ref @ direction
        test_proj = test @ direction

        cvar_ref = self._compute_cvar(ref_proj, self.alpha)
        cvar_test = self._compute_cvar(test_proj, self.alpha)

        return abs(cvar_test - cvar_ref)

    @staticmethod
    def _compute_cvar(values: np.ndarray, alpha: float) -> float:
        """Compute CVaR_alpha = E[X | X >= VaR_alpha].

        For shift detection we care about the upper tail (extreme states).
        """
        var_threshold = np.quantile(values, alpha)
        tail_values = values[values >= var_threshold]
        if len(tail_values) == 0:
            return float(var_threshold)
        return float(np.mean(tail_values))
