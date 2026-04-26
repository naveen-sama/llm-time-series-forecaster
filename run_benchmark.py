"""
run_benchmark.py — Main benchmark runner.

Runs all forecasters (or a subset) on the test split of the processed
dataset and saves per-window predictions to results/benchmark_results.csv.

Usage
-----
  # Run all classical models + one LLM strategy
  python run_benchmark.py

  # Run only classical models (no LLM calls)
  python run_benchmark.py --skip-llm

  # Run a specific LLM model and strategy
  python run_benchmark.py --model gpt-4o --strategy cot

  # Limit to the first N test windows
  python run_benchmark.py --max-windows 50

  # Use synthetic data (re-generates if processed/ is missing)
  python run_benchmark.py --synthetic
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Project root on sys.path so that local imports work regardless of cwd
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from data.loaders import TimeSeriesDataset, load_split
from data.preprocess import run as preprocess_run
from forecasters.classical.arima_model import ARIMAForecaster
from forecasters.classical.lstm import LSTMForecaster
from forecasters.classical.xgboost_model import XGBoostForecaster

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("benchmark")

RESULTS_DIR = ROOT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULTS_CSV = RESULTS_DIR / "benchmark_results.csv"

HISTORY_LEN = 168  # 1 week of hourly data
HORIZON = 24        # forecast 24 hours ahead


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


# ---------------------------------------------------------------------------
# Forecaster registry
# ---------------------------------------------------------------------------


def _build_classical_forecasters(horizon: int) -> list[tuple[str, object]]:
    """Return list of (name, forecaster_instance) for classical models."""
    return [
        ("ARIMA", ARIMAForecaster(horizon=horizon)),
        ("XGBoost", XGBoostForecaster(horizon=horizon)),
        ("LSTM", LSTMForecaster(seq_len=HISTORY_LEN, horizon=horizon, epochs=20)),
    ]


def _build_llm_forecasters(
    model_name: str,
    strategy: str | None,
    horizon: int,
    history_len: int,
) -> list[tuple[str, object]]:
    """Return list of (name, forecaster_instance) for LLM strategies."""
    from forecasters.llm.strategies import STRATEGY_REGISTRY, get_strategy

    if strategy:
        strategies = [strategy]
    else:
        strategies = list(STRATEGY_REGISTRY.keys())

    forecasters: list[tuple[str, object]] = []
    for strat in strategies:
        name = f"LLM/{model_name}/{strat}"
        try:
            f = get_strategy(strat, model_name=model_name,
                             history_len=history_len, horizon=horizon)
            forecasters.append((name, f))
        except Exception as exc:
            logger.warning("Could not build LLM forecaster '%s': %s", name, exc)

    return forecasters


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------


def run_benchmark(
    model_name: str = "gpt-4o",
    strategy: str | None = None,
    max_windows: int | None = None,
    skip_llm: bool = False,
    synthetic: bool = False,
    stride: int = 24,
) -> pd.DataFrame:
    """
    Execute the full benchmark and return the results DataFrame.

    Parameters
    ----------
    model_name  : LLM model identifier
    strategy    : specific LLM strategy or None (runs all)
    max_windows : cap on number of test windows evaluated
    skip_llm    : if True, only classical models run
    synthetic   : regenerate data from scratch using synthetic series
    stride      : step between windows (default 24 = non-overlapping days)
    """
    # --- 1. Ensure processed data exists ------------------------------------
    processed_dir = ROOT_DIR / "data" / "processed"
    if not (processed_dir / "train.csv").exists() or synthetic:
        logger.info("Preprocessed data not found — running preprocessing …")
        preprocess_run(use_synthetic=synthetic)

    # --- 2. Load dataset ----------------------------------------------------
    logger.info("Loading dataset …")
    _, test_ds = load_split(
        processed_dir=processed_dir,
        history_len=HISTORY_LEN,
        horizon=HORIZON,
        stride=stride,
    )
    logger.info("Test windows available: %d", len(test_ds))

    n_windows = len(test_ds)
    if max_windows:
        n_windows = min(n_windows, max_windows)

    if n_windows == 0:
        raise RuntimeError("No test windows available. Run preprocessing first.")

    # --- 3. Build forecasters -----------------------------------------------
    forecasters: list[tuple[str, object]] = _build_classical_forecasters(HORIZON)
    if not skip_llm:
        llm_forecasters = _build_llm_forecasters(
            model_name, strategy, HORIZON, HISTORY_LEN
        )
        forecasters.extend(llm_forecasters)

    logger.info(
        "Running %d forecaster(s) × %d windows …",
        len(forecasters),
        n_windows,
    )

    # --- 4. Evaluation loop -------------------------------------------------
    records: list[dict] = []

    for f_name, forecaster in forecasters:
        logger.info("  → %s", f_name)
        window_maes, window_rmses, window_mapes = [], [], []

        for w_idx in range(n_windows):
            history, horizon_true = test_ds[w_idx]

            t0 = time.perf_counter()
            try:
                forecast = forecaster.predict(history)
            except Exception as exc:
                logger.warning(
                    "    [%s] window %d failed: %s", f_name, w_idx, exc
                )
                forecast = np.full(HORIZON, history.mean(), dtype=np.float32)

            elapsed = time.perf_counter() - t0

            w_mae = mae(horizon_true, forecast)
            w_rmse = rmse(horizon_true, forecast)
            w_mape = mape(horizon_true, forecast)

            window_maes.append(w_mae)
            window_rmses.append(w_rmse)
            window_mapes.append(w_mape)

            records.append(
                {
                    "forecaster": f_name,
                    "window_idx": w_idx,
                    "mae": w_mae,
                    "rmse": w_rmse,
                    "mape": w_mape,
                    "latency_s": elapsed,
                    "forecast": forecast.tolist(),
                    "actual": horizon_true.tolist(),
                }
            )

        logger.info(
            "    MAE=%.4f  RMSE=%.4f  MAPE=%.2f%%",
            np.mean(window_maes),
            np.mean(window_rmses),
            np.mean(window_mapes),
        )

    results_df = pd.DataFrame(records)

    # --- 5. Save ------------------------------------------------------------
    results_df.to_csv(RESULTS_CSV, index=False)
    logger.info("Results saved to %s", RESULTS_CSV)

    return results_df


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def print_summary(df: pd.DataFrame) -> None:
    """Print a compact MAE/RMSE/MAPE summary table to stdout."""
    summary = (
        df.groupby("forecaster")[["mae", "rmse", "mape", "latency_s"]]
        .mean()
        .sort_values("mae")
    )
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(
        summary.to_string(
            float_format=lambda x: f"{x:.4f}",
            header=True,
        )
    )
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LLM Time Series Forecaster Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM model identifier (e.g. gpt-4o, claude-3-5-sonnet-20241022, llama3-70b-8192).",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        help=(
            "LLM strategy to run. If omitted all 8 strategies are run. "
            "Options: direct, cot, few_shot, statistical_context, "
            "seasonal_decomp, role_prompt, self_consistency, hybrid."
        ),
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap on number of test windows to evaluate (useful for quick runs).",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run only classical baselines (no LLM API calls).",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force generation of a synthetic dataset.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=24,
        help="Stride between test windows.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    results = run_benchmark(
        model_name=args.model,
        strategy=args.strategy,
        max_windows=args.max_windows,
        skip_llm=args.skip_llm,
        synthetic=args.synthetic,
        stride=args.stride,
    )
    print_summary(results)
