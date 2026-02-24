"""Martingale log-wealth trajectory plots."""

import numpy as np
import matplotlib.pyplot as plt
from .style import apply_style, PROBE_COLORS


def plot_log_wealth_trajectories(
    trial_result,
    alpha: float = 0.05,
    save_path: str = None,
):
    """Plot log-wealth over time for all three probes + threshold line."""
    apply_style()

    fig, ax = plt.subplots(figsize=(7, 4))
    threshold = np.log(1.0 / alpha)

    for probe_name, trajectory in trial_result.mssd_log_wealth.items():
        color = PROBE_COLORS.get(probe_name, "gray")
        ax.plot(
            trajectory, label=f"{probe_name} probe", color=color, linewidth=1.5
        )

    ax.axhline(
        y=threshold,
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Threshold (log(1/{alpha})={threshold:.1f})",
    )

    if trial_result.shift_injection_step is not None:
        ax.axvline(
            x=trial_result.shift_injection_step // 10,
            color="gray",
            linestyle=":",
            alpha=0.7,
            label="Shift start",
        )

    ax.set_xlabel("Probe Evaluation Step")
    ax.set_ylabel("Log-Wealth log(M_t)")
    ax.set_title(f"Martingale Trajectories — {trial_result.shift_type} shift")
    ax.legend(loc="upper left")

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig
