"""
BaseLLMForecaster — shared infrastructure for all LLM-based forecasters.

Responsibilities
----------------
1. Format a numpy history array as a readable text string.
2. Build the full prompt by delegating to the active strategy's
   ``build_prompt`` method.
3. Dispatch the API call to the correct provider
   (OpenAI / Anthropic / Groq), handling retries and rate-limit back-off.
4. Parse the 24 numeric predictions from the raw model response.
5. Expose a ``predict(history)`` method that ties 1–4 together.
"""

from __future__ import annotations

import ast
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HORIZON = 24
DEFAULT_HISTORY_LEN = 168
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # seconds


# ---------------------------------------------------------------------------
# Provider dispatch helpers
# ---------------------------------------------------------------------------


def _call_openai(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Call the OpenAI Chat Completions API and return the text reply."""
    import openai  # lazy import so other providers work without openai installed

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _call_anthropic(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Call the Anthropic Messages API and return the text reply."""
    import anthropic  # lazy import

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return message.content[0].text if message.content else ""


def _call_groq(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """Call the Groq Chat Completions API and return the text reply."""
    import groq  # lazy import

    client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


# Map provider prefix → callable
_PROVIDER_MAP: dict[str, Any] = {
    "gpt": _call_openai,
    "o1": _call_openai,
    "o3": _call_openai,
    "text-": _call_openai,
    "claude": _call_anthropic,
    "llama": _call_groq,
    "mixtral": _call_groq,
    "gemma": _call_groq,
}


def _get_provider_fn(model_name: str):
    model_lower = model_name.lower()
    for prefix, fn in _PROVIDER_MAP.items():
        if model_lower.startswith(prefix):
            return fn
    raise ValueError(
        f"Cannot determine provider for model '{model_name}'. "
        "Supported prefixes: gpt, o1, o3, text-, claude, llama, mixtral, gemma."
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_forecast(text: str, horizon: int) -> np.ndarray:
    """
    Extract exactly *horizon* floats from a model response string.

    Tries three strategies in order:
    1. Find a Python list literal with ast.literal_eval.
    2. Extract all floating-point tokens with a regex.
    3. Raise a ValueError.
    """
    # Strategy 1 – Look for a bracketed list, optionally prefixed by a label
    list_match = re.search(r"\[([^\[\]]+)\]", text)
    if list_match:
        try:
            candidates = ast.literal_eval(f"[{list_match.group(1)}]")
            floats = [float(x) for x in candidates]
            if len(floats) >= horizon:
                return np.array(floats[:horizon], dtype=np.float32)
        except (ValueError, SyntaxError):
            pass

    # Strategy 2 – Regex for sequences of numbers (possibly comma-separated)
    tokens = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", text)
    floats = [float(t) for t in tokens]
    if len(floats) >= horizon:
        return np.array(floats[:horizon], dtype=np.float32)

    # If we have some but not enough, pad with the last observed value
    if floats:
        logger.warning(
            "Only %d values parsed (expected %d). Padding with last value.",
            len(floats),
            horizon,
        )
        arr = np.array(floats, dtype=np.float32)
        pad = np.full(horizon - len(arr), arr[-1], dtype=np.float32)
        return np.concatenate([arr, pad])

    raise ValueError(
        f"Could not parse {horizon} numeric values from model response:\n{text[:500]}"
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseLLMForecaster(ABC):
    """
    Abstract base for LLM forecasters.

    Subclasses must implement ``build_prompt`` which returns a
    ``(system_prompt, user_prompt)`` tuple given the serialised history.

    Parameters
    ----------
    model_name : str
        The exact model identifier, e.g. 'gpt-4o', 'claude-3-5-sonnet-20241022'.
    history_len : int
        Expected number of history time steps.
    horizon : int
        Number of future steps to forecast.
    temperature : float
        Sampling temperature passed to the API.
    max_tokens : int
        Maximum tokens for the model response.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        history_len: int = DEFAULT_HISTORY_LEN,
        horizon: int = DEFAULT_HORIZON,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ) -> None:
        self.model_name = model_name
        self.history_len = history_len
        self.horizon = horizon
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._provider_fn = _get_provider_fn(model_name)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_prompt(
        self, history: np.ndarray
    ) -> tuple[str, str]:
        """
        Build (system_prompt, user_prompt) from a history array.

        Parameters
        ----------
        history : np.ndarray of shape (history_len,)

        Returns
        -------
        system_prompt : str
        user_prompt   : str
        """

    # ------------------------------------------------------------------
    # Text formatting
    # ------------------------------------------------------------------

    @staticmethod
    def format_series(values: np.ndarray, decimals: int = 4) -> str:
        """
        Render a 1-D numpy array as a compact, human-readable string.

        Groups values in rows of 24 to make daily patterns visible.
        """
        rounded = np.round(values.astype(float), decimals)
        rows: list[str] = []
        row_size = 24
        for i in range(0, len(rounded), row_size):
            chunk = rounded[i : i + row_size]
            rows.append(", ".join(f"{v:.{decimals}f}" for v in chunk))
        return "\n".join(rows)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def _call_with_retry(
        self, system_prompt: str, user_prompt: str
    ) -> str:
        """Call the provider API with exponential back-off on rate-limit errors."""
        last_exc: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return self._provider_fn(
                    self.model_name,
                    system_prompt,
                    user_prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as exc:
                last_exc = exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    "API call failed (attempt %d/%d): %s. Retrying in %.1fs …",
                    attempt,
                    MAX_RETRIES,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"API call failed after {MAX_RETRIES} attempts."
        ) from last_exc

    def predict(self, history: np.ndarray) -> np.ndarray:
        """
        Forecast the next ``horizon`` values given a ``history`` array.

        Parameters
        ----------
        history : np.ndarray of shape (history_len,)
            Historical time series values (most-recent last).

        Returns
        -------
        np.ndarray of shape (horizon,)
        """
        if history.ndim != 1:
            raise ValueError(f"history must be 1-D, got shape {history.shape}.")
        if len(history) < self.history_len:
            # Pad left with the first observed value if needed
            pad = np.full(self.history_len - len(history), history[0], dtype=np.float32)
            history = np.concatenate([pad, history])
        history = history[-self.history_len :]

        system_prompt, user_prompt = self.build_prompt(history)
        logger.debug(
            "[%s] Sending prompt (%d chars) to %s …",
            self.__class__.__name__,
            len(user_prompt),
            self.model_name,
        )
        raw_response = self._call_with_retry(system_prompt, user_prompt)
        logger.debug("[%s] Raw response:\n%s", self.__class__.__name__, raw_response[:300])

        forecast = _parse_forecast(raw_response, self.horizon)
        # Clip to a physically plausible range (no negative power)
        forecast = np.clip(forecast, 0.0, None)
        return forecast

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model={self.model_name!r}, "
            f"history_len={self.history_len}, "
            f"horizon={self.horizon})"
        )
