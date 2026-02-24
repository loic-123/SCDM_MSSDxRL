"""Generate LaTeX tables from experiment results."""

import argparse
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.generate_figures import load_results
from mssd.evaluation.analysis import compute_summary_table


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
