"""Deep Q-Network agent for CartPole-v1."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from .base_agent import BaseAgent
from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int = 4, n_actions: int = 2, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent for CartPole-v1."""

    def __init__(
        self,
        obs_dim: int = 4,
        n_actions: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 500,
        buffer_size: int = 10000,
        batch_size: int = 64,
        target_update_freq: int = 10,
        hidden: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.q_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.n_actions = n_actions
        self._steps = 0

    @property
    def epsilon(self) -> float:
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -self._steps / self.epsilon_decay
        )

    def select_action(self, obs: np.ndarray) -> int:
        """Greedy action (no exploration). Used at deployment time."""
        with torch.no_grad():
            q_vals = self.q_net(
                torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            )
            return int(q_vals.argmax(dim=1).item())

    def train(self, env, num_episodes: int = 500, **kwargs) -> dict:
        rewards_per_episode = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                if np.random.random() < self.epsilon:
                    action = np.random.randint(self.n_actions)
                else:
                    action = self.select_action(obs)

                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                self.buffer.push(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
                self._steps += 1

                if len(self.buffer) >= self.batch_size:
                    self._update()

            if ep % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.q_net.state_dict())
            rewards_per_episode.append(total_reward)
        return {"episode_rewards": rewards_per_episode}

    def _update(self):
        batch = self.buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = batch

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: Path) -> None:
        self.q_net.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        self.q_net.eval()
