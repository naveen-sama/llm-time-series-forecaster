"""
XGBoost-based time series forecaster.

Feature engineering
-------------------
For each target step t, the model uses the following lag features:
  lag_1, lag_2, lag_3    — very short-term autocorrelation
  lag_24                 — same hour yesterday
  lag_48                 — same hour two days ago
  lag_168                — same hour last week (weekly seasonality)

Plus calendar features derived from the internal offset within the history:
  hour_of_day            — cyclic [0, 23]
  sin_hour, cos_hour     — continuous cyclic encoding

A separate XGBoost model is trained for each horizon step (direct
multi-step strategy) to avoid error accumulation.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LAGS = [1, 2, 3, 24, 48, 168]
HORIZON = 24
DEFAULT_XGB_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------


def _build_features(series: np.ndarray, lags: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a feature matrix X and target vector y from a 1-D series.

    Each row of X corresponds to a single time step and contains:
      - lag values specified by *lags*
      - cyclic hour-of-day encodings (assuming series starts at hour 0)

    Only rows where all lags are available are returned.

    Parameters
    ----------
    series : 1-D float array
    lags   : list of positive integers

    Returns
    -------
    X : float32 array of shape (n_samples, n_features)
    y : float32 array of shape (n_samples,)
    """
    max_lag = max(lags)
    n = len(series)
    rows: list[np.ndarray] = []
    targets: list[float] = []

    for t in range(max_lag, n):
        lag_feats = np.array([series[t - lag] for lag in lags], dtype=np.float32)
        hour = t % 24
        sin_h = np.float32(np.sin(2 * np.pi * hour / 24))
        cos_h = np.float32(np.cos(2 * np.pi * hour / 24))
        row = np.concatenate([lag_feats, [np.float32(hour), sin_h, cos_h]])
        rows.append(row)
        targets.append(float(series[t]))

    X = np.stack(rows)
    y = np.array(targets, dtype=np.float32)
    return X, y


def _build_single_row(series: np.ndarray, lags: list[int], offset: int = 0) -> np.ndarray:
    """
    Build a single feature row for predicting the value at position
    ``len(series) + offset`` (where offset ≥ 0 means future steps).

    Uses the tail of *series* for lag look-ups; values not yet observed
    are filled with the last known value (recursive / stubbed).
    """
    # Extend series with placeholder values for recursive feature construction
    extended = list(series)
    max_lag = max(lags)
    # We only need lags relative to the query position
    query_idx = len(series) + offset
    # Pad if needed (for offset > 0 we need prior predictions; caller handles)
    while len(extended) < query_idx:
        extended.append(extended[-1])

    lag_feats = np.array(
        [extended[query_idx - lag] for lag in lags], dtype=np.float32
    )
    hour = query_idx % 24
    sin_h = np.float32(np.sin(2 * np.pi * hour / 24))
    cos_h = np.float32(np.cos(2 * np.pi * hour / 24))
    return np.concatenate([lag_feats, [np.float32(hour), sin_h, cos_h]])


# ---------------------------------------------------------------------------
# XGBoostForecaster
# ---------------------------------------------------------------------------


class XGBoostForecaster:
    """
    Multi-step direct XGBoost forecaster with lag features.

    One model is trained per horizon step so each step can learn its own
    feature-importance profile.

    Parameters
    ----------
    lags : list[int]
        Lag offsets to use as features.
    horizon : int
        Number of future steps.
    xgb_params : dict | None
        XGBoost hyperparameters (see DEFAULT_XGB_PARAMS).
    """

    MODEL_NAME = "XGBoost"

    def __init__(
        self,
        lags: list[int] | None = None,
        horizon: int = HORIZON,
        xgb_params: dict | None = None,
    ) -> None:
        self.lags = sorted(lags or LAGS)
        self.horizon = horizon
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}
        self._models: list | None = None  # list[XGBRegressor], one per step

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> "XGBoostForecaster":
        """
        Train one XGBRegressor per horizon step on the provided history.

        Parameters
        ----------
        history : 1-D float array of length ≥ max(lags) + horizon

        Returns
        -------
        self
        """
        from xgboost import XGBRegressor

        logger.debug(
            "[XGBoost] Fitting %d models on %d observations …",
            self.horizon,
            len(history),
        )
        self._models = []
        max_lag = max(self.lags)

        for step in range(1, self.horizon + 1):
            # For direct multi-step, shift the target by *step*
            if len(history) < max_lag + step:
                logger.warning(
                    "[XGBoost] Not enough data for step %d (need %d, got %d). "
                    "Using naïve repeat.",
                    step,
                    max_lag + step,
                    len(history),
                )
                self._models.append(None)
                continue

            # Build (X, y) where y is the value *step* steps ahead
            X_rows: list[np.ndarray] = []
            y_vals: list[float] = []

            for t in range(max_lag, len(history) - step + 1):
                lag_feats = np.array(
                    [history[t - lag] for lag in self.lags], dtype=np.float32
                )
                hour = t % 24
                sin_h = np.float32(np.sin(2 * np.pi * hour / 24))
                cos_h = np.float32(np.cos(2 * np.pi * hour / 24))
                row = np.concatenate([lag_feats, [np.float32(hour), sin_h, cos_h]])
                X_rows.append(row)
                y_vals.append(float(history[t + step - 1]))

            if len(X_rows) < 5:
                self._models.append(None)
                continue

            X = np.stack(X_rows)
            y = np.array(y_vals, dtype=np.float32)

            model = XGBRegressor(**self.xgb_params)
            model.fit(X, y)
            self._models.append(model)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Fit on *history* then produce a (horizon,) forecast array.

        Parameters
        ----------
        history : np.ndarray of shape (history_len,)

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
        self.fit(history)

        max_lag = max(self.lags)
        forecast = np.empty(self.horizon, dtype=np.float32)

        for step_idx, model in enumerate(self._models):
            step = step_idx + 1

            if model is None:
                # Naïve: use the same-hour value from 24h ago
                naive_idx = len(history) - 24 + (step_idx % 24)
                naive_idx = max(0, min(naive_idx, len(history) - 1))
                forecast[step_idx] = history[naive_idx]
                continue

            # Build query row using the most-recent lags
            t = len(history)
            lag_feats = np.array(
                [
                    history[t - lag] if t - lag >= 0 else history[0]
                    for lag in self.lags
                ],
                dtype=np.float32,
            )
            hour = (t + step - 1) % 24
            sin_h = np.float32(np.sin(2 * np.pi * hour / 24))
            cos_h = np.float32(np.cos(2 * np.pi * hour / 24))
            x_query = np.concatenate([lag_feats, [np.float32(hour), sin_h, cos_h]])

            pred = float(model.predict(x_query.reshape(1, -1))[0])
            forecast[step_idx] = pred

        return np.clip(forecast, 0.0, None)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"XGBoostForecaster("
            f"lags={self.lags}, "
            f"horizon={self.horizon}, "
            f"n_estimators={self.xgb_params.get('n_estimators')})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    rng = np.random.default_rng(1)
    t = np.arange(500)
    synthetic = (
        1.5
        + 0.5 * np.sin(2 * np.pi * t / 24)
        + 0.2 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.05, 500)
    ).astype(np.float32)

    forecaster = XGBoostForecaster(horizon=24)
    preds = forecaster.predict(synthetic)
    print(f"XGBoost forecast ({len(preds)} steps): {preds[:6]} …")
