"""Product martingale accumulator with probe attribution for MSSD."""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AlarmResult:
    """Result from a martingale monitoring session."""

    fired: bool
    firing_step: Optional[int]
    firing_probe: Optional[str]  # "body", "tail", "structure", or None
    log_wealth_history: dict  # probe_name -> list of log(M_t)
    product_log_wealth_history: List[float]


class ProductMartingale:
    """Maintains per-probe log-wealth processes and a product martingale.

    For each probe k, the log-wealth at time t is:
        log(M_t^k) = sum_{i=1}^{t} log(e_i^k)

    An alarm fires when sum_k log(M_t^k) > log(1/alpha).
    The firing probe is the one with the highest individual log-wealth.
    """

    def __init__(self, probe_names: List[str], alpha: float = 0.05):
        self.probe_names = probe_names
        self.alpha = alpha
        self.threshold = np.log(1.0 / alpha)

        self.log_wealth = {name: 0.0 for name in probe_names}
        self.log_wealth_history = {name: [0.0] for name in probe_names}
        self.product_log_wealth_history = [0.0]
        self._step = 0
        self._alarm_fired = False
        self._alarm_step = None
        self._alarm_probe = None

    def update(self, e_values: dict) -> bool:
        """Update with new e-values from all probes.

        Args:
            e_values: {probe_name: e_value} for this time step
        Returns:
            True if alarm fires at this step
        """
        if self._alarm_fired:
            return True

        self._step += 1
        product_log = 0.0

        for name in self.probe_names:
            ev = e_values[name]
            log_ev = np.log(max(ev, 1e-10))
            self.log_wealth[name] += log_ev
            self.log_wealth_history[name].append(self.log_wealth[name])
            product_log += self.log_wealth[name]

        self.product_log_wealth_history.append(product_log)

        if product_log > self.threshold:
            self._alarm_fired = True
            self._alarm_step = self._step
            self._alarm_probe = max(
                self.probe_names, key=lambda n: self.log_wealth[n]
            )
            return True

        return False

    def get_result(self) -> AlarmResult:
        return AlarmResult(
            fired=self._alarm_fired,
            firing_step=self._alarm_step,
            firing_probe=self._alarm_probe,
            log_wealth_history=dict(self.log_wealth_history),
            product_log_wealth_history=list(self.product_log_wealth_history),
        )

    def reset(self):
        self.log_wealth = {name: 0.0 for name in self.probe_names}
        self.log_wealth_history = {name: [0.0] for name in self.probe_names}
        self.product_log_wealth_history = [0.0]
        self._step = 0
        self._alarm_fired = False
        self._alarm_step = None
        self._alarm_probe = None
