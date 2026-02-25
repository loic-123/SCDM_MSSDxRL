"""Abstract base class for shift detection probes with block-bootstrap e-value computation."""

from abc import ABC, abstractmethod
import numpy as np


class BaseProbe(ABC):
    """Abstract base class for shift detection probes.

    Each probe:
    1. Takes a reference window and a test window of observations
    2. Computes a scalar test statistic
    3. Converts it to an e-value via block-bootstrap permutation
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def compute_statistic(self, ref: np.ndarray, test: np.ndarray) -> float:
        """Compute the raw test statistic between reference and test windows.

        Args:
            ref:  (n_ref, obs_dim) reference observations
            test: (n_test, obs_dim) test observations
        Returns:
            Scalar test statistic (higher = more evidence of shift)
        """
        ...

    def to_evalue(
        self,
        ref: np.ndarray,
        test: np.ndarray,
        n_permutations: int = 200,
        block_size: int = 10,
        rng: np.random.Generator = None,
        betting_fraction: float = 0.5,
    ) -> float:
        """Convert test statistic to an e-value via block-bootstrap permutation.

        Uses a betting-fraction approach for safe sequential testing:
            e_safe = betting_fraction * (1/p_hat) + (1 - betting_fraction)

        This preserves the e-value property (E[e] <= 1 under H0) while
        preventing single-step explosions that cause premature alarms.

        Under H0 (no shift), e-values are ~1.
        Under H1 (shift present), e-values grow steadily.
        """
        if rng is None:
            rng = np.random.default_rng()

        observed_stat = self.compute_statistic(ref, test)

        pooled = np.concatenate([ref, test], axis=0)
        n_ref = len(ref)
        n_total = len(pooled)

        count_geq = 0
        for _ in range(n_permutations):
            perm_idx = _block_bootstrap_permutation(n_total, block_size, rng)
            perm_ref = pooled[perm_idx[:n_ref]]
            perm_test = pooled[perm_idx[n_ref:]]
            perm_stat = self.compute_statistic(perm_ref, perm_test)
            if perm_stat >= observed_stat:
                count_geq += 1

        p_hat = (count_geq + 1) / (n_permutations + 1)
        e_raw = 1.0 / p_hat
        e_value = betting_fraction * e_raw + (1.0 - betting_fraction)
        return e_value


def _block_bootstrap_permutation(
    n: int, block_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a block-bootstrap permutation of indices 0..n-1.

    Draws blocks of consecutive indices, wrapping around, until we have n indices.
    Then randomly permutes the result.
    """
    indices = []
    while len(indices) < n:
        start = rng.integers(0, n)
        block = [(start + i) % n for i in range(block_size)]
        indices.extend(block)
    indices = np.array(indices[:n])
    rng.shuffle(indices)
    return indices
