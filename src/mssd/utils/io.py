"""Save/load utilities for arrays, checkpoints, and results."""

from pathlib import Path
import numpy as np


def ensure_dir(path: Path) -> Path:
    """Ensure parent directory exists, return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_numpy(path: str, **arrays) -> None:
    """Save numpy arrays to compressed npz file."""
    path = ensure_dir(Path(path))
    np.savez_compressed(path, **arrays)


def load_numpy(path: str) -> dict:
    """Load numpy arrays from npz file, returns dict-like."""
    return dict(np.load(path, allow_pickle=True))
