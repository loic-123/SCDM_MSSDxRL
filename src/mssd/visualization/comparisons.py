"""ADD comparison bar charts: MSSD vs Global MMD baseline."""

import numpy as np
import matplotlib.pyplot as plt
from .style import apply_style


def plot_add_comparison(summary: dict, env_name: str, save_path: str = None):
    """Bar chart: ADD for MSSD vs Global MMD baseline across all conditions.

    Grouped by (shift_type, severity), with MSSD and baseline bars side by side.
    """
    apply_style()

    conditions = [
        (k, v)
        for k, v in summary.items()
        if k[0] == env_name and k[1] != "none"
    ]
    conditions.sort(key=lambda x: (x[0][1], x[0][2]))

    if not conditions:
        return None

    labels = [f"{k[1]}\nsev={k[2]}" for k, v in conditions]
    adds_mssd = [v["add_mssd"] for _, v in conditions]
    adds_baseline = [v["add_baseline"] for _, v in conditions]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, adds_mssd, width, label="MSSD", color="#2196F3")
    ax.bar(
        x + width / 2, adds_baseline, width, label="Global MMD", color="#FF9800"
    )

    ax.set_ylabel("Average Detection Delay (steps)")
    ax.set_xlabel("Shift Condition")
    ax.set_title(f"Detection Delay Comparison — {env_name}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()

    if save_path:
        fig.savefig(save_path)
    plt.close(fig)
    return fig
