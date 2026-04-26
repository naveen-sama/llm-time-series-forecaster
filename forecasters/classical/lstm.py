"""
LSTM-based time series forecaster using PyTorch.

Architecture
------------
  Input  → LSTM (2 layers, hidden_size=64) → Linear → Output
  Input shape  : (batch, seq_len, 1)
  Output shape : (batch, horizon)

Training
--------
  - Adam optimiser, LR 1e-3, StepLR scheduler (halves every 10 epochs)
  - MSE loss
  - 20 epochs (configurable)
  - Uses the full history array as the single training sequence (sliding
    windows extracted internally).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PyTorch model definition
# ---------------------------------------------------------------------------


def _build_model(input_size: int, hidden_size: int, num_layers: int, horizon: int):
    """Build and return the PyTorch LSTM model."""
    import torch
    import torch.nn as nn

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=0.1 if num_layers > 1 else 0.0,
            )
            self.fc = nn.Linear(hidden_size, horizon)

        def forward(self, x):
            # x : (batch, seq_len, input_size)
            out, _ = self.lstm(x)
            # Use the output at the last time step
            return self.fc(out[:, -1, :])

    return LSTMNet()


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


def _make_windows(
    series: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window over *series* and return (X, y) arrays.

    X : (n_windows, seq_len, 1)   float32
    y : (n_windows, horizon)      float32
    """
    n = len(series)
    xs, ys = [], []
    for i in range(n - seq_len - horizon + 1):
        xs.append(series[i : i + seq_len])
        ys.append(series[i + seq_len : i + seq_len + horizon])
    X = np.stack(xs)[:, :, np.newaxis].astype(np.float32)
    y = np.stack(ys).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# LSTMForecaster
# ---------------------------------------------------------------------------


class LSTMForecaster:
    """
    Two-layer LSTM forecaster trained end-to-end with PyTorch.

    Parameters
    ----------
    seq_len : int
        Number of historical time steps fed to the LSTM.  Default 168.
    horizon : int
        Number of future steps to predict.  Default 24.
    hidden_size : int
        Number of LSTM hidden units.  Default 64.
    num_layers : int
        Number of stacked LSTM layers.  Default 2.
    epochs : int
        Training epochs.  Default 20.
    batch_size : int
        Mini-batch size.  Default 32.
    lr : float
        Initial learning rate.  Default 1e-3.
    device : str | None
        'cpu', 'cuda', or None (auto-detect).
    """

    MODEL_NAME = "LSTM"

    def __init__(
        self,
        seq_len: int = 168,
        horizon: int = 24,
        hidden_size: int = 64,
        num_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        self.seq_len = seq_len
        self.horizon = horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self._model = None
        self._mu: float = 0.0
        self._sigma: float = 1.0

        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        return ((arr - self._mu) / (self._sigma + 1e-8)).astype(np.float32)

    def _denormalise(self, arr: np.ndarray) -> np.ndarray:
        return (arr * (self._sigma + 1e-8) + self._mu).astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, history: np.ndarray) -> "LSTMForecaster":
        """
        Train the LSTM on sliding windows extracted from *history*.

        Parameters
        ----------
        history : 1-D float array

        Returns
        -------
        self
        """
        import torch
        import torch.nn as nn
        from torch.optim import Adam
        from torch.optim.lr_scheduler import StepLR
        from torch.utils.data import DataLoader, TensorDataset

        # z-score normalise
        self._mu = float(history.mean())
        self._sigma = float(history.std())
        norm_history = self._normalise(history)

        min_len = self.seq_len + self.horizon
        if len(norm_history) < min_len:
            logger.warning(
                "[LSTM] History length %d < required %d. Padding.", len(history), min_len
            )
            pad = np.full(min_len - len(norm_history), norm_history[0], dtype=np.float32)
            norm_history = np.concatenate([pad, norm_history])

        X_np, y_np = _make_windows(norm_history, self.seq_len, self.horizon)
        X_t = torch.from_numpy(X_np)
        y_t = torch.from_numpy(y_np)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _build_model(1, self.hidden_size, self.num_layers, self.horizon)
        self._model = self._model.to(self.device)

        optimiser = Adam(self._model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = StepLR(optimiser, step_size=10, gamma=0.5)
        criterion = nn.MSELoss()

        logger.info("[LSTM] Training for %d epochs on %s …", self.epochs, self.device)
        self._model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimiser.zero_grad()
                preds = self._model(xb)
                loss = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimiser.step()
                epoch_loss += loss.item() * len(xb)
            scheduler.step()
            if epoch % 5 == 0 or epoch == 1:
                logger.debug(
                    "[LSTM] Epoch %2d/%d — loss: %.6f",
                    epoch,
                    self.epochs,
                    epoch_loss / len(dataset),
                )

        self._model.eval()
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Fit on *history* then forecast the next *horizon* steps.

        Parameters
        ----------
        history : np.ndarray of shape (history_len,)

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
        import torch

        self.fit(history)

        norm_history = self._normalise(history)
        # Use the last seq_len values as the input sequence
        input_seq = norm_history[-self.seq_len :]
        if len(input_seq) < self.seq_len:
            pad = np.full(self.seq_len - len(input_seq), input_seq[0], dtype=np.float32)
            input_seq = np.concatenate([pad, input_seq])

        x = torch.from_numpy(input_seq[np.newaxis, :, np.newaxis]).to(self.device)

        with torch.no_grad():
            pred_norm = self._model(x).cpu().numpy()[0]

        forecast = self._denormalise(pred_norm)
        return np.clip(forecast, 0.0, None)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"LSTMForecaster("
            f"seq_len={self.seq_len}, "
            f"horizon={self.horizon}, "
            f"hidden={self.hidden_size}x{self.num_layers}, "
            f"epochs={self.epochs})"
        )


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rng = np.random.default_rng(2)
    t = np.arange(500)
    synthetic = (
        1.5
        + 0.5 * np.sin(2 * np.pi * t / 24)
        + 0.2 * np.sin(2 * np.pi * t / 168)
        + rng.normal(0, 0.05, 500)
    ).astype(np.float32)

    forecaster = LSTMForecaster(seq_len=168, horizon=24, epochs=5)
    preds = forecaster.predict(synthetic)
    print(f"LSTM forecast ({len(preds)} steps): {preds[:6]} …")
