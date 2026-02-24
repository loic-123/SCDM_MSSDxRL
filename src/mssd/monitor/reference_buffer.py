"""Reference observation buffer for MSSD."""

import numpy as np
from pathlib import Path


class ReferenceBuffer:
    """Collects and stores reference observations from nominal deployment.

    Used during the reference collection phase: run the trained agent in
    the unshifted environment and store all observations.
    """

    def __init__(self):
        self._observations: list = []

    def add(self, obs: np.ndarray):
        self._observations.append(obs.copy())

    def get_array(self) -> np.ndarray:
        return np.array(self._observations)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, observations=self.get_array())

    @classmethod
    def load(cls, path: Path) -> "ReferenceBuffer":
        buf = cls()
        data = np.load(path)
        buf._observations = list(data["observations"])
        return buf

    def __len__(self) -> int:
        return len(self._observations)
