"""Shared matplotlib style and color configuration for MSSD figures."""

import matplotlib.pyplot as plt
import seaborn as sns

STYLE_CONFIG = {
    "figure.figsize": (6, 4),
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
}

PROBE_COLORS = {
    "body": "#2196F3",
    "tail": "#F44336",
    "structure": "#4CAF50",
}

SHIFT_LABELS = {
    "body": "Body Shift",
    "tail": "Tail Shift",
    "structure": "Structure Shift",
}


def apply_style():
    plt.rcParams.update(STYLE_CONFIG)
    sns.set_palette("Set2")
