"""
Preprocessing pipeline for the UCI Household Power Consumption dataset.

Downloads the dataset, resamples to hourly frequency, performs a train/test
split, and saves the resulting CSVs to data/processed/.  If the download
fails for any reason a synthetic dataset is generated so the rest of the
pipeline can still run.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00235/household_power_consumption.zip"
)
ZIP_NAME = "household_power_consumption.zip"
TXT_NAME = "household_power_consumption.txt"

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_uci(dest: Path) -> Path:
    """Download and extract the UCI zip archive. Returns path to the .txt file."""
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / ZIP_NAME
    txt_path = dest / TXT_NAME

    if txt_path.exists():
        print(f"[preprocess] Raw file already exists: {txt_path}")
        return txt_path

    print("[preprocess] Downloading UCI Household Power Consumption dataset …")
    response = requests.get(UCI_URL, timeout=120, stream=True)
    response.raise_for_status()

    with open(zip_path, "wb") as fh:
        for chunk in response.iter_content(chunk_size=1 << 16):
            fh.write(chunk)

    print("[preprocess] Extracting archive …")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)

    if not txt_path.exists():
        raise FileNotFoundError(f"Expected {txt_path} after extraction.")

    return txt_path


def _load_uci(txt_path: Path) -> pd.DataFrame:
    """Parse the raw .txt file and return a clean DataFrame indexed by datetime."""
    print("[preprocess] Parsing raw file …")
    df = pd.read_csv(
        txt_path,
        sep=";",
        parse_dates={"datetime": ["Date", "Time"]},
        infer_datetime_format=True,
        na_values=["?"],
        low_memory=False,
    )
    df.set_index("datetime", inplace=True)
    df.sort_index(inplace=True)
    # Use Global_active_power as the target series
    series = df["Global_active_power"].dropna().astype(float)
    return series.to_frame(name="power_kw")


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------


def _make_synthetic(n_hours: int = 8760) -> pd.DataFrame:
    """
    Generate a synthetic household power-consumption time series.

    Combines a daily cycle, a weekly cycle, trend, and Gaussian noise so
    that classical and ML models have a realistic signal to learn from.
    """
    print("[preprocess] Generating synthetic dataset …")
    rng = np.random.default_rng(42)
    t = np.arange(n_hours)

    # Daily seasonality (period 24 h)
    daily = 0.6 * np.sin(2 * np.pi * t / 24) + 0.3 * np.cos(2 * np.pi * t / 12)
    # Weekly seasonality (period 168 h)
    weekly = 0.4 * np.sin(2 * np.pi * t / 168)
    # Slow upward trend
    trend = 0.00005 * t
    # Gaussian noise
    noise = rng.normal(0, 0.1, n_hours)
    # Baseline consumption
    power = 1.5 + daily + weekly + trend + noise
    power = np.clip(power, 0.05, None)  # power cannot be negative

    start = pd.Timestamp("2007-01-01 00:00:00")
    index = pd.date_range(start, periods=n_hours, freq="h")
    return pd.DataFrame({"power_kw": power}, index=index)


# ---------------------------------------------------------------------------
# Resample + split
# ---------------------------------------------------------------------------


def _resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a DataFrame with a sub-hourly DatetimeIndex to hourly means."""
    if df.index.freq is not None and df.index.freq == "h":
        return df
    return df.resample("h").mean().dropna()


def _train_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    split = int(n * (1 - test_fraction))
    return df.iloc[:split], df.iloc[split:]


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run(use_synthetic: bool = False) -> None:
    """Full preprocessing pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if use_synthetic:
        hourly = _make_synthetic()
    else:
        try:
            txt_path = _download_uci(RAW_DIR)
            raw_df = _load_uci(txt_path)
            hourly = _resample_hourly(raw_df)
        except Exception as exc:
            print(f"[preprocess] Download/parse failed ({exc}). Using synthetic data.")
            hourly = _make_synthetic()

    print(f"[preprocess] Hourly series: {len(hourly)} rows "
          f"({hourly.index[0]} … {hourly.index[-1]})")

    train, test = _train_test_split(hourly)
    print(f"[preprocess] Train: {len(train)} rows  |  Test: {len(test)} rows")

    train_path = PROCESSED_DIR / "train.csv"
    test_path = PROCESSED_DIR / "test.csv"
    train.to_csv(train_path)
    test.to_csv(test_path)
    print(f"[preprocess] Saved → {train_path}")
    print(f"[preprocess] Saved → {test_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess time series data.")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Skip download and use a synthetic dataset.",
    )
    args = parser.parse_args()
    run(use_synthetic=args.synthetic)
