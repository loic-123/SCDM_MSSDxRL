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
    """Maintains per-probe log-wealth processes with Bonferroni correction.

    For each probe k, the log-wealth at time t is:
        log(M_t^k) = sum_{i=1}^{t} log(e_i^k)

    An alarm fires when any individual probe's log-wealth exceeds
    log(K / alpha), where K is the number of probes (Bonferroni correction).
    The firing probe (diagnosis) is the one with the highest log-wealth.
    """

    def __init__(self, probe_names: List[str], alpha: float = 0.05,
                 min_probe_steps: int = 1, alarm_probes: List[str] = None):
        self.probe_names = probe_names
        self.alpha = alpha
        # alarm_probes: only these probes can trigger the alarm; others are
        # tracked for log-wealth but only used for post-alarm diagnosis.
        self.alarm_probes = alarm_probes or probe_names
        self.n_alarm_probes = len(self.alarm_probes)
        self.min_probe_steps = min_probe_steps
        # Bonferroni-corrected threshold over alarm probes only
        self.threshold = np.log(self.n_alarm_probes / alpha)

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

        for name in self.probe_names:
            ev = e_values[name]
            log_ev = np.log(max(ev, 1e-10))
            self.log_wealth[name] += log_ev
            self.log_wealth_history[name].append(self.log_wealth[name])

        # Only alarm probes can trigger; others contribute to diagnosis only
        max_alarm_wealth = max(
            self.log_wealth[n] for n in self.alarm_probes
        )
        self.product_log_wealth_history.append(max_alarm_wealth)

        if max_alarm_wealth > self.threshold and self._step >= self.min_probe_steps:
            self._alarm_fired = True
            self._alarm_step = self._step
            # Diagnosis: probe with highest log-wealth among alarm probes
            self._alarm_probe = max(
                self.alarm_probes, key=lambda n: self.log_wealth[n]
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
