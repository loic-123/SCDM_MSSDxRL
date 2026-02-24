"""Integration test: MSSD monitor on CliffWalking environment."""

import numpy as np
import gymnasium as gym
from mssd.envs.cliff_wrapper import CliffWalkingContinuousObs
from mssd.agents.tabular_q import TabularQAgent
from mssd.monitor.mssd_monitor import MSSDMonitor
from mssd.monitor.reference_buffer import ReferenceBuffer
from mssd.envs.shifts import ShiftInjector, ShiftConfig


def test_monitor_runs_on_cliffwalking():
    """Verify the monitor can be instantiated and run on CliffWalking."""
    env = CliffWalkingContinuousObs(gym.make("CliffWalking-v1"))
    agent = TabularQAgent()

    # Quick training (just enough to have a policy)
    agent.train(env, num_episodes=100)

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

    # Run a short deployment with body shift
    rng = np.random.default_rng(42)
    injector = ShiftInjector(
        ShiftConfig(shift_type="body", severity=1.0, env_name="cliffwalking"),
        rng=rng,
    )

    obs, _ = env.reset()
    for step in range(200):
        action = agent.select_action(obs)
        next_obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        monitored_obs = injector(next_obs) if step >= 50 else next_obs
        monitor.observe(monitored_obs)

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs

    # Just verify it ran without errors
    result = monitor.get_result()
    assert len(result.log_wealth_history["body"]) > 0
