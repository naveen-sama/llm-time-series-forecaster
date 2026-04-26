"""
Data loaders for the LLM Time Series Forecaster benchmark.

Provides:
  - TimeSeriesDataset  — iterable of (history, horizon) numpy windows
  - load_split         — convenience function that returns train/test datasets
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Window = Tuple[np.ndarray, np.ndarray]  # (history [H,], horizon [F,])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TimeSeriesDataset:
    """
    Sliding-window dataset over a univariate time series stored in a CSV.

    Each window is a pair (history, horizon):
      - history : float32 array of shape (history_len,)
      - horizon : float32 array of shape (horizon,)

    Windows are generated with a stride of 1 by default.  Set stride > 1
    to thin the dataset for faster benchmarking.

    Parameters
    ----------
    csv_path : str | Path
        Path to a CSV with a DatetimeIndex column and a single numeric
        target column (e.g. 'power_kw').
    target_col : str
        Name of the column to forecast.
    history_len : int
        Number of historical time steps fed to the model (default 168 = 1 week).
    horizon : int
        Number of future time steps to predict (default 24 = 1 day).
    stride : int
        Step size between consecutive windows (default 1).
    normalize : bool
        If True, each history window is z-score normalised before being
        returned; the horizon is scaled by the same statistics.
    """

    def __init__(
        self,
        csv_path: str | Path,
        target_col: str = "power_kw",
        history_len: int = 168,
        horizon: int = 24,
        stride: int = 1,
        normalize: bool = False,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.target_col = target_col
        self.history_len = history_len
        self.horizon = horizon
        self.stride = stride
        self.normalize = normalize

        self._values: np.ndarray = self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> np.ndarray:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {self.csv_path}\n"
                "Run `python data/preprocess.py` first."
            )
        df = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        if self.target_col not in df.columns:
            raise KeyError(
                f"Column '{self.target_col}' not found in {self.csv_path}. "
                f"Available columns: {list(df.columns)}"
            )
        # Forward-fill then back-fill any remaining NaNs
        series = df[self.target_col].ffill().bfill()
        return series.to_numpy(dtype=np.float32)

    def _window(self, idx: int) -> Window:
        start = idx
        mid = start + self.history_len
        end = mid + self.horizon

        history = self._values[start:mid].copy()
        horizon = self._values[mid:end].copy()

        if self.normalize:
            mu = history.mean()
            sigma = history.std() + 1e-8
            history = (history - mu) / sigma
            horizon = (horizon - mu) / sigma

        return history, horizon

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        total = len(self._values) - self.history_len - self.horizon + 1
        if total <= 0:
            return 0
        # Account for stride
        return max(0, (total - 1) // self.stride + 1)

    def __getitem__(self, idx: int) -> Window:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)}).")
        return self._window(idx * self.stride)

    def __iter__(self) -> Iterator[Window]:
        for i in range(len(self)):
            yield self[i]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def values(self) -> np.ndarray:
        """Raw 1-D float32 array of all values in the CSV."""
        return self._values

    def get_batch(
        self, start: int, size: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a batch of *size* windows starting at window index *start*.

        Returns
        -------
        histories : float32 array of shape (size, history_len)
        horizons  : float32 array of shape (size, horizon)
        """
        end = min(start + size, len(self))
        windows = [self[i] for i in range(start, end)]
        histories = np.stack([w[0] for w in windows])
        horizons = np.stack([w[1] for w in windows])
        return histories, horizons

    def summary(self) -> dict:
        """Return a dict with basic dataset statistics."""
        return {
            "csv_path": str(self.csv_path),
            "n_values": len(self._values),
            "n_windows": len(self),
            "history_len": self.history_len,
            "horizon": self.horizon,
            "stride": self.stride,
            "min": float(self._values.min()),
            "max": float(self._values.max()),
            "mean": float(self._values.mean()),
            "std": float(self._values.std()),
        }


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------


def load_split(
    processed_dir: str | Path = PROCESSED_DIR,
    target_col: str = "power_kw",
    history_len: int = 168,
    horizon: int = 24,
    stride: int = 1,
    normalize: bool = False,
) -> tuple[TimeSeriesDataset, TimeSeriesDataset]:
    """
    Load both the train and test splits as TimeSeriesDataset instances.

    Returns
    -------
    train_dataset, test_dataset
    """
    processed_dir = Path(processed_dir)
    train = TimeSeriesDataset(
        processed_dir / "train.csv",
        target_col=target_col,
        history_len=history_len,
        horizon=horizon,
        stride=stride,
        normalize=normalize,
    )
    test = TimeSeriesDataset(
        processed_dir / "test.csv",
        target_col=target_col,
        history_len=history_len,
        horizon=horizon,
        stride=stride,
        normalize=normalize,
    )
    return train, test


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train_ds, test_ds = load_split()
    print("Train dataset:", train_ds.summary())
    print("Test  dataset:", test_ds.summary())
    history, horizon = train_ds[0]
    print(f"history shape: {history.shape}  horizon shape: {horizon.shape}")
    print(f"history[:5]: {history[:5]}")
    print(f"horizon[:5]: {horizon[:5]}")
