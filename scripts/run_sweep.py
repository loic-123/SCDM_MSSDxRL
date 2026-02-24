"""Run a full experimental sweep (parallelized across trials)."""

import argparse
import sys
import numpy as np
from pathlib import Path
from multiprocessing import Pool
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mssd.utils.config import load_config
from mssd.utils.seeding import set_all_seeds
from mssd.utils.io import ensure_dir
from mssd.envs import make_env
from mssd.envs.shifts import ShiftInjector, ShiftConfig
from mssd.agents import make_agent
from mssd.monitor.reference_buffer import ReferenceBuffer
from scripts.run_experiment import run_single_trial


def _run_trial_worker(trial_spec: dict, cfg: dict):
    """Worker function for parallel trial execution."""
    env = make_env(cfg["env"])
    agent = make_agent(cfg["agent"])
    agent.load(trial_spec["agent_path"])

    ref_buf = ReferenceBuffer.load(trial_spec["reference_path"])
    reference_obs = ref_buf.get_array()

    shift_injector = None
    if trial_spec["shift_type"] != "none":
        rng = np.random.default_rng(trial_spec["seed"])
        env_name = "cliffwalking" if "Cliff" in cfg["env"]["name"] else "cartpole"
        shift_injector = ShiftInjector(
            ShiftConfig(
                shift_type=trial_spec["shift_type"],
                severity=trial_spec["severity"],
                env_name=env_name,
            ),
            rng=rng,
        )

    trial_cfg = {**cfg, "trial_id": trial_spec["trial_id"]}
    result = run_single_trial(
        env, agent, reference_obs, shift_injector, trial_cfg, trial_spec["seed"]
    )

    # Save individual result
    output_path = ensure_dir(
        Path(trial_spec["output_dir"])
        / f"{trial_spec['shift_type']}_{trial_spec['severity']}_trial_{trial_spec['trial_id']}.npz"
    )
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
    return result


def main():
    parser = argparse.ArgumentParser(description="Run full experimental sweep")
    parser.add_argument("--config", required=True, help="Path to sweep config YAML")
    parser.add_argument("--agent", required=True, help="Path to saved agent")
    parser.add_argument("--reference", required=True, help="Path to reference .npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    exp = cfg.get("experiment", cfg)

    base_seed = cfg.get("seed", 42)
    trials = []
    trial_idx = 0

    for shift_type in exp.get("shift_types", ["body"]):
        severities = cfg.get("shifts", {}).get(shift_type, {}).get("severities", [1.0])
        for severity in severities:
            for t in range(exp.get("n_trials", 2)):
                trials.append({
                    "shift_type": shift_type,
                    "severity": severity,
                    "trial_id": t,
                    "seed": base_seed + trial_idx,
                    "agent_path": args.agent,
                    "reference_path": args.reference,
                    "output_dir": exp.get("output_dir", "artifacts/results"),
                })
                trial_idx += 1

    if exp.get("include_no_shift"):
        for t in range(exp.get("n_no_shift_trials", 50)):
            trials.append({
                "shift_type": "none",
                "severity": 0.0,
                "trial_id": t,
                "seed": base_seed + trial_idx,
                "agent_path": args.agent,
                "reference_path": args.reference,
                "output_dir": exp.get("output_dir", "artifacts/results"),
            })
            trial_idx += 1

    n_workers = exp.get("parallel_workers", 1)
    print(f"Running {len(trials)} trials with {n_workers} workers...")

    worker = partial(_run_trial_worker, cfg=cfg)

    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = pool.map(worker, trials)
    else:
        results = [worker(t) for t in trials]

    n_detected = sum(1 for r in results if r.mssd_alarm_fired)
    print(f"Sweep complete. {len(results)} trials, {n_detected} alarms fired.")


if __name__ == "__main__":
    main()
