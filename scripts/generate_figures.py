"""Generate all paper figures from experiment results."""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mssd.evaluation.metrics import TrialResult
from mssd.evaluation.analysis import compute_summary_table, aggregate_by_condition
from mssd.visualization.heatmaps import plot_orthogonality_heatmap
from mssd.visualization.trajectories import plot_log_wealth_trajectories
from mssd.visualization.comparisons import plot_add_comparison


def load_results(results_dir: str) -> list:
    """Load all .npz result files into TrialResult objects."""
    results = []
    for npz_path in Path(results_dir).rglob("*.npz"):
        data = dict(np.load(npz_path, allow_pickle=True))
        alarm_step = int(data["mssd_alarm_step"])
        baseline_step = int(data["baseline_alarm_step"])
        results.append(
            TrialResult(
                env_name=str(data["env_name"]),
                shift_type=str(data["shift_type"]),
                severity=float(data["severity"]),
                trial_id=int(data["trial_id"]),
                seed=int(data["seed"]),
                mssd_alarm_fired=bool(data["mssd_alarm_fired"]),
                mssd_alarm_step=alarm_step if alarm_step >= 0 else None,
                mssd_diagnosed_probe=str(data["mssd_diagnosed_probe"]) if str(data["mssd_diagnosed_probe"]) != "none" else None,
                mssd_log_wealth={},  # Not stored in compressed format
                baseline_alarm_fired=bool(data["baseline_alarm_fired"]),
                baseline_alarm_step=baseline_step if baseline_step >= 0 else None,
                shift_injection_step=int(data["shift_injection_step"]),
                total_steps=int(data["total_steps"]),
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_dir)
    if not results:
        print("No results found!")
        return

    shift_results = [r for r in results if r.shift_type != "none"]
    no_shift_results = [r for r in results if r.shift_type == "none"]

    summary = compute_summary_table(shift_results, no_shift_results)

    # Get unique env names and severities
    env_names = sorted(set(r.env_name for r in shift_results))
    for env_name in env_names:
        env_results = [r for r in shift_results if r.env_name == env_name]
        severities = sorted(set(r.severity for r in env_results))

        # Orthogonality heatmap for median severity
        if severities:
            mid_sev = severities[len(severities) // 2]
            short_name = "cliffwalking" if "Cliff" in env_name else "cartpole"
            plot_orthogonality_heatmap(
                shift_results, env_name, mid_sev,
                save_path=str(output / f"orthogonality_{short_name}.pdf"),
            )
            print(f"Generated orthogonality heatmap for {short_name} (severity={mid_sev})")

        # ADD comparison
        plot_add_comparison(
            summary, env_name,
            save_path=str(output / f"add_comparison_{short_name}.pdf"),
        )
        print(f"Generated ADD comparison for {short_name}")

    print(f"All figures saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
