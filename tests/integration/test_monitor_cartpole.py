"""Integration test: MSSD monitor on CartPole environment."""

import numpy as np
import gymnasium as gym
from mssd.envs.cartpole_wrapper import CartPoleLogged
from mssd.agents.dqn import DQNAgent
from mssd.monitor.mssd_monitor import MSSDMonitor
from mssd.monitor.reference_buffer import ReferenceBuffer
from mssd.envs.shifts import ShiftInjector, ShiftConfig


def test_monitor_runs_on_cartpole():
    """Verify the monitor can be instantiated and run on CartPole."""
    env = CartPoleLogged(gym.make("CartPole-v1"))
    agent = DQNAgent()

    # Quick training (just enough to have a policy)
    agent.train(env, num_episodes=20)

    # Collect reference
    buffer = ReferenceBuffer()
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            buffer.add(obs)
            action = agent.select_action(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    reference_obs = buffer.get_array()

    # Create monitor with small params for speed
    monitor = MSSDMonitor(
        reference_obs=reference_obs,
        window_size=20,
        window_step=5,
        alpha=0.05,
        n_permutations=20,
        seed=42,
    )

    # Run a short deployment
    obs, _ = env.reset()
    for step in range(100):
        action = agent.select_action(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        monitor.observe(next_obs)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # Just verify it ran without errors
    result = monitor.get_result()
    assert len(result.log_wealth_history["body"]) > 0
