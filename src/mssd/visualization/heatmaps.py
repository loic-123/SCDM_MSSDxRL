"""3x3 orthogonality heatmap — the key diagnostic figure."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .style import apply_style, SHIFT_LABELS


def plot_orthogonality_heatmap(
    trial_results: list,
    env_name: str,
    severity: float,
    save_path: str = None,
):
    """Plot 3x3 heatmap: rows = injected shift type, cols = firing probe.

    Cell (i, j) = fraction of trials where shift type i triggered probe j first.
    Perfect orthogonality = identity matrix.
    """
    apply_style()

    shift_types = ["body", "tail", "structure"]
    probe_names = ["body", "tail", "structure"]

    matrix = np.zeros((3, 3))

    for i, st in enumerate(shift_types):
        trials = [
            t
            for t in trial_results
            if t.env_name == env_name
            and t.shift_type == st
            and abs(t.severity - severity) < 1e-6
        ]
        fired_trials = [t for t in trials if t.mssd_alarm_fired]
        if not fired_trials:
            continue
        for j, pn in enumerate(probe_names):
            count = sum(
                1 for t in fired_trials if t.mssd_diagnosed_probe == pn
            )
            matrix[i, j] = count / len(fired_trials)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[SHIFT_LABELS[p] for p in probe_names],
        yticklabels=[SHIFT_LABELS[s] for s in shift_types],
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "Fraction"},
    )
    ax.set_xlabel("Firing Probe")
    ax.set_ylabel("Injected Shift")
    ax.set_title(f"Probe Orthogonality — {env_name} (severity={severity})")

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig
