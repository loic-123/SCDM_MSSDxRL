"""Structure probe: detects correlation breakdown via Frobenius distance."""

import numpy as np
from .base_probe import BaseProbe


class StructureProbe(BaseProbe):
    """Detects structural shift (sensor decoupling) via correlation matrix distance.

    Computes the Frobenius norm of the difference between the empirical
    correlation matrices of reference and test windows.
    """

    def __init__(self):
        super().__init__(name="structure")

    def compute_statistic(self, ref: np.ndarray, test: np.ndarray) -> float:
        corr_ref = self._safe_corrcoef(ref)
        corr_test = self._safe_corrcoef(test)

        diff = corr_ref - corr_test
        return float(np.sqrt(np.sum(diff**2)))

    @staticmethod
    def _safe_corrcoef(X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix, handling constant columns gracefully."""
        n_features = X.shape[1]
        stds = X.std(axis=0)

        mask = stds > 1e-10
        if not mask.all():
            X_norm = np.zeros_like(X)
            X_norm[:, mask] = (X[:, mask] - X[:, mask].mean(axis=0)) / stds[mask]
            corr = (X_norm.T @ X_norm) / max(X.shape[0] - 1, 1)
            np.fill_diagonal(corr, 1.0)
            return corr
        return np.corrcoef(X, rowvar=False)
