"""Abstract base class for RL agents."""

from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path


class BaseAgent(ABC):
    """Abstract base class for RL agents. MSSD wraps agents without modifying them."""

    @abstractmethod
    def select_action(self, obs: np.ndarray) -> int:
        """Select action given observation (greedy, no exploration)."""
        ...

    @abstractmethod
    def train(self, env, num_episodes: int, **kwargs) -> dict:
        """Train the agent. Returns training metrics dict."""
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> None:
        ...
