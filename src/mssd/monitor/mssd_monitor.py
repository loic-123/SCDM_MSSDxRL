"""MSSD runtime monitor — the central orchestrator."""

import numpy as np
from typing import Dict, Optional, List
from ..probes.base_probe import BaseProbe
from ..probes.body_probe import BodyProbe
from ..probes.tail_probe import TailProbe
from ..probes.structure_probe import StructureProbe
from ..martingale.product_martingale import ProductMartingale, AlarmResult


class MSSDMonitor:
    """Multi-Scale Sequential Shift Detector.

    Wraps a deployed RL agent. At each monitoring step, it:
    1. Collects a test window of recent observations
    2. Runs all three probes against the reference buffer
    3. Converts to e-values and updates the product martingale
    4. Reports if/which alarm fires

    Usage:
        monitor = MSSDMonitor(reference_obs, ...)
        for obs in deployment_stream:
            result = monitor.observe(obs)
            if result is not None:
                print(f"Shift detected: {result}")
                break
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

        self.probes: Dict[str, BaseProbe] = {
            "body": BodyProbe(),
            "tail": TailProbe(alpha=0.95),
            "structure": StructureProbe(),
        }

        self.martingale = ProductMartingale(
            probe_names=list(self.probes.keys()),
            alpha=alpha,
        )

        self._obs_buffer: List[np.ndarray] = []
        self._total_obs = 0
        self._last_probe_step = 0

    def observe(self, obs: np.ndarray) -> Optional[str]:
        """Feed a single observation to the monitor.

        Returns:
            None if no alarm, or the diagnosed shift type string if alarm fires.
        """
        self._obs_buffer.append(obs)
        self._total_obs += 1

        if (
            self._total_obs - self._last_probe_step >= self.window_step
            and len(self._obs_buffer) >= self.window_size
        ):
            self._last_probe_step = self._total_obs
            return self._run_probes()

        return None

    def _run_probes(self) -> Optional[str]:
        """Run all probes on the current test window."""
        test_window = np.array(self._obs_buffer[-self.window_size :])

        ref_idx = self.rng.choice(
            len(self.reference),
            size=min(self.window_size, len(self.reference)),
            replace=False,
        )
        ref_window = self.reference[ref_idx]

        e_values = {}
        for name, probe in self.probes.items():
            e_values[name] = probe.to_evalue(
                ref_window,
                test_window,
                n_permutations=self.n_permutations,
                block_size=self.block_size,
                rng=self.rng,
            )

        alarm = self.martingale.update(e_values)

        if alarm:
            result = self.martingale.get_result()
            return result.firing_probe
        return None

    @property
    def alarm_fired(self) -> bool:
        return self.martingale._alarm_fired

    @property
    def diagnosis(self) -> Optional[str]:
        return self.martingale._alarm_probe

    def get_result(self) -> AlarmResult:
        return self.martingale.get_result()
