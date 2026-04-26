"""
Concrete LLM forecasting strategy classes.

Each class inherits BaseLLMForecaster and implements ``build_prompt`` using
one of the eight templates defined in ``prompts.py``.

Classes
-------
DirectForecaster           – DIRECT prompt (minimal)
ChainOfThoughtForecaster   – COT prompt (step-by-step reasoning)
FewShotForecaster          – FEW_SHOT prompt (3 in-context examples)
StatisticalContextForecaster – STATISTICAL_CONTEXT (summary stats + raw data)
SeasonalDecompForecaster   – SEASONAL_DECOMP (trend + seasonal + residual)
RolePromptForecaster       – ROLE_PROMPT (expert persona)
SelfConsistencyForecaster  – SELF_CONSISTENCY (5 independent samples, averaged)
HybridForecaster           – HYBRID (stats context + chain-of-thought)
"""

from __future__ import annotations

import logging
import re

import numpy as np

from .base_llm import BaseLLMForecaster, _parse_forecast
from .prompts import (
    COT,
    DIRECT,
    FEW_SHOT,
    HYBRID,
    ROLE_PROMPT,
    SEASONAL_DECOMP,
    SELF_CONSISTENCY,
    STATISTICAL_CONTEXT,
)

logger = logging.getLogger(__name__)

# Shared system message for forecasting tasks
_SYSTEM_BASE = (
    "You are a highly accurate time series forecasting assistant. "
    "Follow the instructions precisely and output only what is requested."
)


# ---------------------------------------------------------------------------
# Helper – statistical summary dict
# ---------------------------------------------------------------------------


def _compute_stats(history: np.ndarray, history_len: int) -> dict:
    """Return a dict of summary statistics for use in prompt templates."""
    last_24h = history[-24:] if len(history) >= 24 else history
    last_6h = history[-6:] if len(history) >= 6 else history
    last_change = float(history[-1] - history[-2]) if len(history) >= 2 else 0.0
    return {
        "history_len": history_len,
        "mean": float(history.mean()),
        "std": float(history.std()),
        "min_val": float(history.min()),
        "max_val": float(history.max()),
        "last_24h_mean": float(last_24h.mean()),
        "last_24h_std": float(last_24h.std()),
        "last_6h_str": ", ".join(f"{v:.4f}" for v in last_6h),
        "last_change": last_change,
    }


# ---------------------------------------------------------------------------
# 1. DirectForecaster
# ---------------------------------------------------------------------------


class DirectForecaster(BaseLLMForecaster):
    """Minimal direct-prediction prompt — no reasoning guidance."""

    STRATEGY_NAME = "direct"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        user_prompt = DIRECT.format(
            history_len=self.history_len,
            history_str=history_str,
            horizon=self.horizon,
        )
        return _SYSTEM_BASE, user_prompt


# ---------------------------------------------------------------------------
# 2. ChainOfThoughtForecaster
# ---------------------------------------------------------------------------


class ChainOfThoughtForecaster(BaseLLMForecaster):
    """Elicits step-by-step reasoning (trend → seasonality → recent → forecast)."""

    STRATEGY_NAME = "cot"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        user_prompt = COT.format(
            history_len=self.history_len,
            history_str=history_str,
            horizon=self.horizon,
        )
        return _SYSTEM_BASE, user_prompt


# ---------------------------------------------------------------------------
# 3. FewShotForecaster
# ---------------------------------------------------------------------------


class FewShotForecaster(BaseLLMForecaster):
    """Provides 3 in-context examples before the actual forecasting query."""

    STRATEGY_NAME = "few_shot"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        user_prompt = FEW_SHOT.format(
            history_len=self.history_len,
            history_str=history_str,
            horizon=self.horizon,
        )
        return _SYSTEM_BASE, user_prompt


# ---------------------------------------------------------------------------
# 4. StatisticalContextForecaster
# ---------------------------------------------------------------------------


class StatisticalContextForecaster(BaseLLMForecaster):
    """Includes a statistical summary (mean, std, min, max, recent stats)."""

    STRATEGY_NAME = "statistical_context"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        stats = _compute_stats(history, self.history_len)
        user_prompt = STATISTICAL_CONTEXT.format(
            history_str=history_str,
            horizon=self.horizon,
            **stats,
        )
        return _SYSTEM_BASE, user_prompt


# ---------------------------------------------------------------------------
# 5. SeasonalDecompForecaster
# ---------------------------------------------------------------------------


class SeasonalDecompForecaster(BaseLLMForecaster):
    """Guides the model to decompose signal into trend, seasonal, and residual."""

    STRATEGY_NAME = "seasonal_decomp"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        user_prompt = SEASONAL_DECOMP.format(
            history_len=self.history_len,
            history_str=history_str,
            horizon=self.horizon,
        )
        return _SYSTEM_BASE, user_prompt


# ---------------------------------------------------------------------------
# 6. RolePromptForecaster
# ---------------------------------------------------------------------------


class RolePromptForecaster(BaseLLMForecaster):
    """Assigns an expert energy-analyst persona before forecasting."""

    STRATEGY_NAME = "role_prompt"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        system_prompt = (
            "You are Dr. Elena Vasquez, a world-leading expert in smart-grid "
            "energy analytics. Respond precisely and concisely."
        )
        user_prompt = ROLE_PROMPT.format(
            history_len=self.history_len,
            history_str=history_str,
            horizon=self.horizon,
        )
        return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# 7. SelfConsistencyForecaster
# ---------------------------------------------------------------------------


class SelfConsistencyForecaster(BaseLLMForecaster):
    """
    Self-consistency: sample 5 independent forecasts, return the element-wise
    mean (which acts as the ensemble estimate).

    The prompt asks the model to output all 5 forecasts in a single response.
    If parsing one sample fails it is silently skipped; the result is averaged
    over whichever samples parsed successfully.
    """

    STRATEGY_NAME = "self_consistency"
    N_SAMPLES = 5

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        user_prompt = SELF_CONSISTENCY.format(
            history_len=self.history_len,
            history_str=history_str,
            horizon=self.horizon,
            n_samples=self.N_SAMPLES,
        )
        return _SYSTEM_BASE, user_prompt

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Override predict to parse N_SAMPLES lists from a single response and
        average them.
        """
        if history.ndim != 1:
            raise ValueError(f"history must be 1-D, got shape {history.shape}.")
        if len(history) < self.history_len:
            pad = np.full(
                self.history_len - len(history), history[0], dtype=np.float32
            )
            history = np.concatenate([pad, history])
        history = history[-self.history_len :]

        system_prompt, user_prompt = self.build_prompt(history)
        raw_response = self._call_with_retry(system_prompt, user_prompt)

        # Extract all bracketed lists from the response
        bracket_pattern = re.compile(r"\[([^\[\]]+)\]")
        samples: list[np.ndarray] = []
        for match in bracket_pattern.finditer(raw_response):
            try:
                candidate = _parse_forecast(f"[{match.group(1)}]", self.horizon)
                samples.append(candidate)
            except (ValueError, Exception):
                continue

        if not samples:
            # Fallback: try to parse the whole response as one forecast
            logger.warning(
                "[SelfConsistency] Could not find %d samples; falling back to "
                "single-forecast parse.",
                self.N_SAMPLES,
            )
            forecast = _parse_forecast(raw_response, self.horizon)
            return np.clip(forecast, 0.0, None)

        stacked = np.stack(samples[:self.N_SAMPLES])
        averaged = stacked.mean(axis=0).astype(np.float32)
        return np.clip(averaged, 0.0, None)


# ---------------------------------------------------------------------------
# 8. HybridForecaster
# ---------------------------------------------------------------------------


class HybridForecaster(BaseLLMForecaster):
    """
    Combines a statistical summary header with a structured chain-of-thought
    reasoning protocol.
    """

    STRATEGY_NAME = "hybrid"

    def build_prompt(self, history: np.ndarray) -> tuple[str, str]:
        history_str = self.format_series(history)
        stats = _compute_stats(history, self.history_len)
        user_prompt = HYBRID.format(
            history_str=history_str,
            horizon=self.horizon,
            **stats,
        )
        return _SYSTEM_BASE, user_prompt


# ---------------------------------------------------------------------------
# Registry — maps strategy name → class
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[BaseLLMForecaster]] = {
    "direct": DirectForecaster,
    "cot": ChainOfThoughtForecaster,
    "few_shot": FewShotForecaster,
    "statistical_context": StatisticalContextForecaster,
    "seasonal_decomp": SeasonalDecompForecaster,
    "role_prompt": RolePromptForecaster,
    "self_consistency": SelfConsistencyForecaster,
    "hybrid": HybridForecaster,
}


def get_strategy(
    strategy_name: str,
    model_name: str = "gpt-4o",
    history_len: int = 168,
    horizon: int = 24,
    **kwargs,
) -> BaseLLMForecaster:
    """
    Instantiate a forecaster by strategy name.

    Parameters
    ----------
    strategy_name : str
        One of the keys in STRATEGY_REGISTRY.
    model_name : str
        LLM model identifier.
    history_len, horizon : int
        Window sizes.
    **kwargs
        Additional keyword args forwarded to the forecaster constructor.

    Returns
    -------
    BaseLLMForecaster instance
    """
    key = strategy_name.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {sorted(STRATEGY_REGISTRY.keys())}"
        )
    cls = STRATEGY_REGISTRY[key]
    return cls(
        model_name=model_name,
        history_len=history_len,
        horizon=horizon,
        **kwargs,
    )
