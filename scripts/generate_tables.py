"""Generate LaTeX tables from experiment results."""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mssd.evaluation.metrics import TrialResult
from mssd.evaluation.analysis import compute_summary_table


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
                mssd_log_wealth={},
                baseline_alarm_fired=bool(data["baseline_alarm_fired"]),
                baseline_alarm_step=baseline_step if baseline_step >= 0 else None,
                shift_injection_step=int(data["shift_injection_step"]),
                total_steps=int(data["total_steps"]),
            )
        )
    return results


def format_add(val: float) -> str:
    if val == float("inf"):
        return "--"
    return f"{val:.1f}"


def main():
    parser = argparse.ArgumentParser(description="Generate LaTeX tables")
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

    # Main results table
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{MSSD Results Summary}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Env & Shift & Severity & ADD (MSSD) & ADD (MMD) & Disc. Acc. & FAR \\",
        r"\midrule",
    ]

    for key in sorted(summary.keys()):
        env, shift, sev = key
        m = summary[key]
        short_env = "Cliff" if "Cliff" in env else "CartPole"
        lines.append(
            f"{short_env} & {shift} & {sev} & "
            f"{format_add(m['add_mssd'])} & {format_add(m['add_baseline'])} & "
            f"{m['discrimination']:.2f} & {m['far']:.3f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    table_path = output / "table_main_results.tex"
    table_path.write_text("\n".join(lines))
    print(f"Main results table saved to: {table_path}")


if __name__ == "__main__":
    main()
