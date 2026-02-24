"""RBF kernel and bandwidth selection utilities."""

import numpy as np
from scipy.spatial.distance import cdist


def rbf_mmd_squared(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> float:
    """Compute unbiased squared MMD with RBF kernel.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where k(a,b) = exp(-||a-b||^2 / (2 * bandwidth^2))
    """
    gamma = 1.0 / (2.0 * bandwidth**2)

    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")

    KXX = np.exp(-gamma * XX)
    KYY = np.exp(-gamma * YY)
    KXY = np.exp(-gamma * XY)

    n = len(X)
    m = len(Y)

    np.fill_diagonal(KXX, 0.0)
    np.fill_diagonal(KYY, 0.0)

    mmd_sq = (
        KXX.sum() / (n * (n - 1))
        + KYY.sum() / (m * (m - 1))
        - 2.0 * KXY.sum() / (n * m)
    )
    return float(max(mmd_sq, 0.0))


def median_heuristic_bandwidth(X: np.ndarray) -> float:
    """Median heuristic: bandwidth = median of pairwise distances."""
    dists = cdist(X, X, "euclidean")
    triu_idx = np.triu_indices(len(X), k=1)
    median_dist = np.median(dists[triu_idx])
    return max(float(median_dist), 1e-6)
