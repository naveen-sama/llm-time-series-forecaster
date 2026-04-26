"""
evaluate.py — Load benchmark results and produce evaluation artefacts.

Actions
-------
1. Load results/benchmark_results.csv produced by run_benchmark.py.
2. Compute per-forecaster aggregate MAE, MAPE, RMSE (mean ± std).
3. Print a formatted table to stdout.
4. Save a grouped bar chart to results/comparison.png.

Usage
-----
  python evaluate.py
  python evaluate.py --results results/benchmark_results.csv
  python evaluate.py --metric rmse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ROOT_DIR / "results"
DEFAULT_CSV = RESULTS_DIR / "benchmark_results.csv"
OUTPUT_PNG = RESULTS_DIR / "comparison.png"

METRICS = ["mae", "rmse", "mape"]
METRIC_LABELS = {"mae": "MAE (kW)", "rmse": "RMSE (kW)", "mape": "MAPE (%)"}


# ---------------------------------------------------------------------------
# Loading & aggregation
# ---------------------------------------------------------------------------


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Results file not found: {csv_path}\n"
            "Run `python run_benchmark.py` first to generate results."
        )
    df = pd.read_csv(csv_path)
    required = {"forecaster", "mae", "rmse", "mape"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Results CSV is missing columns: {missing}")
    return df


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean and std of MAE, RMSE, MAPE per forecaster.

    Returns a DataFrame with columns:
      forecaster, mae_mean, mae_std, rmse_mean, rmse_std, mape_mean, mape_std,
      latency_s_mean, n_windows
    """
    records: list[dict] = []
    for name, group in df.groupby("forecaster"):
        row: dict = {"forecaster": name, "n_windows": len(group)}
        for m in METRICS:
            row[f"{m}_mean"] = group[m].mean()
            row[f"{m}_std"] = group[m].std()
        if "latency_s" in group.columns:
            row["latency_s_mean"] = group["latency_s"].mean()
        records.append(row)

    summary = pd.DataFrame(records).sort_values("mae_mean").reset_index(drop=True)
    return summary


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def print_table(summary: pd.DataFrame) -> None:
    header = (
        f"{'Forecaster':<42} "
        f"{'MAE':>10} "
        f"{'RMSE':>10} "
        f"{'MAPE%':>10} "
        f"{'Latency(s)':>12}"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("  TIME SERIES FORECASTER BENCHMARK — EVALUATION RESULTS")
    print("=" * len(header))
    print(header)
    print(sep)

    for _, row in summary.iterrows():
        lat = f"{row['latency_s_mean']:.3f}" if "latency_s_mean" in row else "  N/A"
        print(
            f"{row['forecaster']:<42} "
            f"{row['mae_mean']:>8.4f}±{row['mae_std']:>6.4f} "
            f"{row['rmse_mean']:>8.4f}±{row['rmse_std']:>6.4f} "
            f"{row['mape_mean']:>8.2f}±{row['mape_std']:>5.2f} "
            f"{lat:>12}"
        )

    print(sep)
    best = summary.iloc[0]
    print(
        f"\nBest model by MAE: {best['forecaster']}  "
        f"(MAE={best['mae_mean']:.4f}, RMSE={best['rmse_mean']:.4f}, "
        f"MAPE={best['mape_mean']:.2f}%)"
    )
    print()


# ---------------------------------------------------------------------------
# Bar chart
# ---------------------------------------------------------------------------


def save_chart(summary: pd.DataFrame, output_path: Path, primary_metric: str = "mae") -> None:
    """
    Save a grouped bar chart with MAE, RMSE, and MAPE for each forecaster.

    The primary metric bar is highlighted; error bars show ±1 std.
    """
    forecasters = summary["forecaster"].tolist()
    n = len(forecasters)
    x = np.arange(n)
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(max(12, n * 2), 6))
    fig.suptitle("LLM Time Series Forecaster Benchmark", fontsize=14, fontweight="bold")

    palette = ["#2196F3", "#FF5722", "#4CAF50"]  # blue, orange, green

    for ax, metric, colour in zip(axes, METRICS, palette):
        means = summary[f"{metric}_mean"].to_numpy()
        stds = summary[f"{metric}_std"].to_numpy()

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=colour, alpha=0.82,
                      edgecolor="white", linewidth=0.8)

        # Annotate bar tops
        for bar, val in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(stds) * 0.05,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, color="#333333",
            )

        ax.set_title(METRIC_LABELS[metric], fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_shorten(f) for f in forecasters],
            rotation=35, ha="right", fontsize=8,
        )
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Chart saved to {output_path}")


def _shorten(name: str, max_len: int = 22) -> str:
    """Truncate long forecaster names for axis labels."""
    return name if len(name) <= max_len else "…" + name[-(max_len - 1):]


# ---------------------------------------------------------------------------
# Additional per-horizon analysis
# ---------------------------------------------------------------------------


def horizon_profile(df: pd.DataFrame, forecaster: str) -> None:
    """
    Print the average absolute error at each horizon step for *forecaster*.
    Requires the 'forecast' and 'actual' columns to be JSON lists.
    """
    import ast

    sub = df[df["forecaster"] == forecaster].copy()
    if sub.empty:
        print(f"No data for forecaster '{forecaster}'.")
        return

    try:
        forecasts = sub["forecast"].apply(
            lambda x: np.array(ast.literal_eval(x) if isinstance(x, str) else x)
        )
        actuals = sub["actual"].apply(
            lambda x: np.array(ast.literal_eval(x) if isinstance(x, str) else x)
        )
    except Exception as exc:
        print(f"Could not parse forecast/actual columns: {exc}")
        return

    horizon = len(forecasts.iloc[0])
    errors = np.zeros(horizon)
    for fc, ac in zip(forecasts, actuals):
        errors += np.abs(fc - ac)
    errors /= len(sub)

    print(f"\nHorizon MAE profile — {forecaster}")
    for h, e in enumerate(errors, 1):
        bar = "█" * int(e / max(errors) * 30)
        print(f"  h+{h:02d}: {e:.4f}  {bar}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate benchmark results from run_benchmark.py.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_CSV,
        help="Path to benchmark_results.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PNG,
        help="Output path for the comparison chart.",
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="mae",
        help="Primary metric to highlight in the chart.",
    )
    parser.add_argument(
        "--horizon-profile",
        metavar="FORECASTER",
        default=None,
        help="Print per-horizon MAE profile for the named forecaster.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    df = load_results(args.results)
    summary = aggregate(df)

    print_table(summary)
    save_chart(summary, args.output, primary_metric=args.metric)

    if args.horizon_profile:
        horizon_profile(df, args.horizon_profile)

    # Save a clean summary CSV alongside the main results
    summary_csv = args.results.parent / "summary.csv"
    summary.to_csv(summary_csv, index=False)
    print(f"Summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
