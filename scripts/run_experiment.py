"""Run a single experiment trial: deploy agent with monitor, inject shift, record outcome."""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mssd.utils.config import load_config
from mssd.utils.seeding import set_all_seeds
from mssd.utils.io import ensure_dir
from mssd.envs import make_env
from mssd.envs.shifts import ShiftInjector, ShiftConfig
from mssd.agents import make_agent
from mssd.monitor.reference_buffer import ReferenceBuffer
from mssd.evaluation.metrics import TrialResult
from mssd.evaluation.runner import run_single_trial


def main():
    parser = argparse.ArgumentParser(description="Run a single experiment trial")
    parser.add_argument("--config", required=True, help="Path to env config YAML")
    parser.add_argument("--defaults", default="defaults.yaml", help="Path to defaults YAML")
    parser.add_argument("--agent", required=True, help="Path to saved agent")
    parser.add_argument("--reference", required=True, help="Path to reference .npz")
    parser.add_argument("--shift-type", required=True, choices=["body", "tail", "structure", "none"])
    parser.add_argument("--severity", type=float, default=0.0)
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", required=True, help="Path to save result .npz")
    args = parser.parse_args()

    env_cfg = load_config(args.config)
    defaults = load_config(args.defaults)
    config = {**defaults, **env_cfg, "trial_id": args.trial_id}

    set_all_seeds(args.seed)
    env = make_env(config["env"])
    agent = make_agent(config["agent"])
    agent.load(args.agent)

    ref_buf = ReferenceBuffer.load(args.reference)
    reference_obs = ref_buf.get_array()

    shift_injector = None
    if args.shift_type != "none":
        rng = np.random.default_rng(args.seed)
        env_name = "cliffwalking" if "Cliff" in config["env"]["name"] else "cartpole"
        shift_injector = ShiftInjector(
            ShiftConfig(shift_type=args.shift_type, severity=args.severity, env_name=env_name),
            rng=rng,
        )

    result = run_single_trial(env, agent, reference_obs, shift_injector, config, args.seed)

    output_path = ensure_dir(Path(args.output))
    np.savez_compressed(
        output_path,
        env_name=result.env_name,
        shift_type=result.shift_type,
        severity=result.severity,
        trial_id=result.trial_id,
        seed=result.seed,
        mssd_alarm_fired=result.mssd_alarm_fired,
        mssd_alarm_step=result.mssd_alarm_step if result.mssd_alarm_step is not None else -1,
        mssd_diagnosed_probe=result.mssd_diagnosed_probe or "none",
        baseline_alarm_fired=result.baseline_alarm_fired,
        baseline_alarm_step=result.baseline_alarm_step if result.baseline_alarm_step is not None else -1,
        shift_injection_step=result.shift_injection_step,
        total_steps=result.total_steps,
    )
    print(f"Trial complete. MSSD alarm: {result.mssd_alarm_fired}, "
          f"Probe: {result.mssd_diagnosed_probe}, "
          f"Delay: {(result.mssd_alarm_step or 0) - result.shift_injection_step}")


if __name__ == "__main__":
    main()
