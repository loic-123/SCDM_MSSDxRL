"""Tabular Q-learning agent for CliffWalking."""

import numpy as np
from pathlib import Path
from .base_agent import BaseAgent


class TabularQAgent(BaseAgent):
    """Tabular Q-learning agent for discretized CliffWalking.

    Since the wrapper produces continuous obs (row, col, cliff_dist),
    we discretize back to int state for the Q-table lookup.
    """

    def __init__(
        self,
        n_states: int = 48,
        n_actions: int = 4,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 0.1,
    ):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.ncol = 12

    def _obs_to_state(self, obs: np.ndarray) -> int:
        """Convert (row, col, cliff_dist) back to flat index for Q-table."""
        return int(obs[0]) * self.ncol + int(obs[1])

    def select_action(self, obs: np.ndarray) -> int:
        state = self._obs_to_state(obs)
        return int(np.argmax(self.q_table[state]))

    def train(self, env, num_episodes: int = 5000, **kwargs) -> dict:
        rewards_per_episode = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                state = self._obs_to_state(obs)
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = int(np.argmax(self.q_table[state]))

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                next_state = self._obs_to_state(next_obs)

                td_target = reward + self.gamma * np.max(
                    self.q_table[next_state]
                ) * (1 - terminated)
                self.q_table[state, action] += self.lr * (
                    td_target - self.q_table[state, action]
                )

                obs = next_obs
                total_reward += reward
            rewards_per_episode.append(total_reward)
        return {"episode_rewards": rewards_per_episode}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, self.q_table)

    def load(self, path: Path) -> None:
        self.q_table = np.load(path)
