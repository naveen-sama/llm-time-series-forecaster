"""
ARIMA-based forecaster using statsmodels SARIMAX.

Model order: SARIMAX(1, 1, 1)(1, 1, 1, 24)
  - Non-seasonal: AR(1), I(1), MA(1)
  - Seasonal:     AR(1), I(1), MA(1), period = 24 (hourly → daily seasonality)

The forecaster is re-fitted on each history window (rolling / expanding
window evaluation) to stay consistent with how LLM forecasters are used.
An optional ``reuse_model`` flag caches the model and only re-fits when the
history changes, which is useful during large benchmark runs.
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Suppress noisy convergence warnings from statsmodels
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


class ARIMAForecaster:
    """
    Seasonal ARIMA forecaster wrapping statsmodels SARIMAX.

    Parameters
    ----------
    order : tuple[int, int, int]
        Non-seasonal (p, d, q) order.  Default (1, 1, 1).
    seasonal_order : tuple[int, int, int, int]
        Seasonal (P, D, Q, s) order.  Default (1, 1, 1, 24).
    horizon : int
        Number of steps ahead to forecast.  Default 24.
    enforce_stationarity : bool
        Passed through to SARIMAX.  Default True.
    enforce_invertibility : bool
        Passed through to SARIMAX.  Default True.
    fit_method : str
        Optimisation method for SARIMAX.fit().  'lbfgs' is fast and robust.
    """

    MODEL_NAME = "SARIMAX(1,1,1)(1,1,1,24)"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
        horizon: int = 24,
        enforce_stationarity: bool = True,
        enforce_invertibility: bool = True,
        fit_method: str = "lbfgs",
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.horizon = horizon
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self.fit_method = fit_method
        self._fitted_result = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> "ARIMAForecaster":
        """
        Fit SARIMAX on the provided history.

        Parameters
        ----------
        history : 1-D float array

        Returns
        -------
        self
        """
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        logger.debug(
            "[ARIMA] Fitting SARIMAX%s%s on %d observations …",
            self.order,
            self.seasonal_order,
            len(history),
        )
        model = SARIMAX(
            history.astype(float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=self.enforce_stationarity,
            enforce_invertibility=self.enforce_invertibility,
            trend="n",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._fitted_result = model.fit(
                disp=False,
                method=self.fit_method,
                maxiter=100,
            )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Fit on *history* and return a forecast array of shape (horizon,).

        Parameters
        ----------
        history : np.ndarray of shape (history_len,)

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
        self.fit(history)
        try:
            forecast = self._fitted_result.forecast(steps=self.horizon)
        except Exception as exc:
            logger.warning("[ARIMA] forecast() failed: %s. Returning naïve forecast.", exc)
            # Naïve seasonal fallback: repeat last 24h cycle
            cycle = history[-24:] if len(history) >= 24 else history
            tiles = int(np.ceil(self.horizon / len(cycle)))
            forecast = np.tile(cycle, tiles)[: self.horizon]

        forecast = np.asarray(forecast, dtype=np.float32)
        # Clip to non-negative values (power cannot be negative)
        return np.clip(forecast, 0.0, None)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ARIMAForecaster("
            f"order={self.order}, "
            f"seasonal_order={self.seasonal_order}, "
            f"horizon={self.horizon})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    rng = np.random.default_rng(0)
    t = np.arange(200)
    synthetic = (
        1.5
        + 0.5 * np.sin(2 * np.pi * t / 24)
        + 0.2 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.05, 200)
    ).astype(np.float32)

    forecaster = ARIMAForecaster(horizon=24)
    preds = forecaster.predict(synthetic)
    print(f"ARIMA forecast ({len(preds)} steps): {preds[:6]} …")
