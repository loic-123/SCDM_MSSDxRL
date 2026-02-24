"""Reproducible seeding for numpy, torch, and gymnasium."""

import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Set seeds for all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int) -> np.random.Generator:
    """Create an independent numpy Generator (preferred over global seed)."""
    return np.random.default_rng(seed)


def resolve_device(device: str = "auto") -> str:
    """Resolve device string: 'auto' picks cuda if available, else cpu."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
