"""Run a single experiment trial: deploy agent with monitor, inject shift, record outcome."""

import numpy as np

from mssd.utils.seeding import set_all_seeds
from mssd.monitor.mssd_monitor import MSSDMonitor
from mssd.baselines.global_mmd import GlobalMMDBaseline
from mssd.evaluation.metrics import TrialResult


def run_single_trial(
    env, agent, reference_obs, shift_injector, config, trial_seed
) -> TrialResult:
    """Run one trial: deploy agent with monitor, inject shift, record outcome."""
    set_all_seeds(trial_seed)

    monitor = MSSDMonitor(
        reference_obs=reference_obs,
        window_size=config.get("window_size", 100),
        window_step=config.get("window_step", 25),
        alpha=config.get("alpha", 0.05),
        n_permutations=config.get("n_permutations", 200),
        block_size=config.get("block_size", 10),
        seed=trial_seed,
        min_probe_steps=config.get("min_probe_steps", 3),
    )

    baseline = GlobalMMDBaseline(
        reference_obs=reference_obs,
        window_size=config.get("window_size", 100),
        window_step=config.get("window_step", 25),
        alpha=config.get("alpha", 0.05),
        n_permutations=config.get("n_permutations", 200),
        block_size=config.get("block_size", 10),
        seed=trial_seed + 999999,
    )

    shift_step = config.get("shift_injection_step", 500)
    max_steps = config.get("max_monitoring_steps", 2000)

    obs, _ = env.reset(seed=trial_seed)
    step = 0
    mssd_alarm_env_step = None
    baseline_alarm_env_step = None

    while step < max_steps:
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        monitored_obs = next_obs.copy()
        if step >= shift_step and shift_injector is not None:
            monitored_obs = shift_injector(next_obs)

        monitor.observe(monitored_obs)
        baseline.observe(monitored_obs)

        # Record the env step at which each alarm first fires
        if monitor.alarm_fired and mssd_alarm_env_step is None:
            mssd_alarm_env_step = step
        if baseline.martingale._alarm_fired and baseline_alarm_env_step is None:
            baseline_alarm_env_step = step

        if done:
            obs, _ = env.reset()
        else:
            obs = next_obs
        step += 1

        if monitor.alarm_fired:
            break

    mssd_result = monitor.get_result()
    baseline_result = baseline.get_result()

    return TrialResult(
        env_name=config["env"]["name"],
        shift_type=shift_injector.config.shift_type if shift_injector else "none",
        severity=shift_injector.config.severity if shift_injector else 0.0,
        trial_id=config.get("trial_id", 0),
        seed=trial_seed,
        mssd_alarm_fired=mssd_result.fired,
        mssd_alarm_step=mssd_alarm_env_step,
        mssd_diagnosed_probe=mssd_result.firing_probe,
        mssd_log_wealth=mssd_result.log_wealth_history,
        baseline_alarm_fired=baseline_result.fired,
        baseline_alarm_step=baseline_alarm_env_step,
        shift_injection_step=shift_step,
        total_steps=step,
    )
