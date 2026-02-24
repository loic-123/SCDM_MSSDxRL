"""Global MMD baseline detector — single test, no probe decomposition."""

import numpy as np
from ..probes.kernels import rbf_mmd_squared, median_heuristic_bandwidth
from ..martingale.product_martingale import ProductMartingale, AlarmResult


class GlobalMMDBaseline:
    """Baseline: single global MMD test (no probe decomposition).

    Uses the same windowing and martingale framework but with a single MMD
    probe over all observation dimensions. Cannot diagnose shift type.
    """

    def __init__(
        self,
        reference_obs: np.ndarray,
        window_size: int = 50,
        window_step: int = 10,
        alpha: float = 0.05,
        n_permutations: int = 200,
        block_size: int = 10,
        seed: int = 42,
    ):
        self.reference = reference_obs
        self.window_size = window_size
        self.window_step = window_step
        self.n_permutations = n_permutations
        self.block_size = block_size
        self.rng = np.random.default_rng(seed)
        self.martingale = ProductMartingale(
            probe_names=["global_mmd"], alpha=alpha
        )
        self._obs_buffer = []
        self._total_obs = 0
        self._last_probe_step = 0

    def observe(self, obs: np.ndarray) -> bool:
        """Returns True if alarm fires."""
        self._obs_buffer.append(obs)
        self._total_obs += 1

        if (
            self._total_obs - self._last_probe_step >= self.window_step
            and len(self._obs_buffer) >= self.window_size
        ):
            self._last_probe_step = self._total_obs
            return self._run_test()
        return False

    def _run_test(self) -> bool:
        test_window = np.array(self._obs_buffer[-self.window_size :])
        ref_idx = self.rng.choice(
            len(self.reference),
            size=min(self.window_size, len(self.reference)),
            replace=False,
        )
        ref_window = self.reference[ref_idx]

        bw = median_heuristic_bandwidth(
            np.concatenate([ref_window, test_window])
        )
        observed = rbf_mmd_squared(ref_window, test_window, bandwidth=bw)

        pooled = np.concatenate([ref_window, test_window], axis=0)
        n_ref = len(ref_window)
        count_geq = 0
        for _ in range(self.n_permutations):
            perm = pooled.copy()
            self.rng.shuffle(perm)
            p_ref = perm[:n_ref]
            p_test = perm[n_ref:]
            stat = rbf_mmd_squared(p_ref, p_test, bandwidth=bw)
            if stat >= observed:
                count_geq += 1

        p_hat = (count_geq + 1) / (self.n_permutations + 1)
        e_value = 1.0 / p_hat

        return self.martingale.update({"global_mmd": e_value})

    def get_result(self) -> AlarmResult:
        return self.martingale.get_result()
