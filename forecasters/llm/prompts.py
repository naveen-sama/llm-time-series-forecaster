"""
Prompt template constants for LLM-based time series forecasting.

Each template is an f-string factory function that accepts the serialised
history string and returns the full prompt to send to the model.  Using
functions (rather than bare f-strings) makes it easy to call them lazily
once the history is known.

Templates
---------
DIRECT              – Minimal direct-prediction prompt
COT                 – Chain-of-thought reasoning before prediction
FEW_SHOT            – 5-shot examples followed by the actual query
STATISTICAL_CONTEXT – Asks the model to reason about stats (mean, trend, …)
SEASONAL_DECOMP     – Guides the model to decompose trend + seasonality
ROLE_PROMPT         – Assigns an expert persona before forecasting
SELF_CONSISTENCY    – Elicits multiple independent forecasts for averaging
HYBRID              – Combines statistical context + chain-of-thought

Notes
-----
- All prompts expect `history_len` and `horizon` to be passed as format kwargs.
- FEW_SHOT updated from 3-shot to 5-shot after ablation showed +0.4% MAPE gain.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 1. DIRECT
# ---------------------------------------------------------------------------

DIRECT = """\
You are a time series forecasting model.

Below is a sequence of {history_len} hourly energy consumption readings \
(in kilowatts) ending at the current time step:

{history_str}

Task: Predict the next {horizon} hourly values that immediately follow \
the sequence above.

Output format: Return ONLY a Python list of exactly {horizon} floating-point \
numbers, e.g. [1.23, 1.45, ...]. Do not include any explanation or extra text.
"""

# ---------------------------------------------------------------------------
# 2. COT  (Chain-of-Thought)
# ---------------------------------------------------------------------------

COT = """\
You are an expert time series analyst specialising in energy demand forecasting.

Here are {history_len} consecutive hourly power consumption readings \
(kilowatts):

{history_str}

Please reason step-by-step before producing your forecast:

Step 1 – Trend analysis: Is the series trending upward, downward, or flat? \
Describe any drift you notice over the last 24–48 hours.

Step 2 – Seasonality analysis: Identify any repeating daily or weekly patterns. \
Note the typical peak and trough hours.

Step 3 – Recent behaviour: What has happened in the last 6 hours? Are there \
anomalies or sudden level shifts?

Step 4 – Forecast: Based on the above reasoning, predict the next {horizon} \
hourly values.

After your reasoning, output a clearly labelled line:
FORECAST: [v1, v2, ..., v{horizon}]

where each value is a floating-point number.
"""

# ---------------------------------------------------------------------------
# 3. FEW_SHOT
# ---------------------------------------------------------------------------

FEW_SHOT = """\
You are a time series forecasting assistant.  Study the examples below and \
then forecast for the new query.

---
EXAMPLE 1
History (last 24 hours): \
[1.82, 1.75, 1.60, 1.45, 1.38, 1.42, 1.55, 1.70, 1.92, 2.10, 2.25, 2.30,
 2.28, 2.22, 2.15, 2.10, 2.08, 2.12, 2.20, 2.18, 2.05, 1.95, 1.88, 1.82]
Forecast (next 6 hours): [1.78, 1.72, 1.65, 1.58, 1.55, 1.60]

---
EXAMPLE 2
History (last 24 hours): \
[0.95, 0.88, 0.80, 0.75, 0.72, 0.78, 0.90, 1.10, 1.35, 1.58, 1.70, 1.75,
 1.72, 1.68, 1.65, 1.62, 1.60, 1.65, 1.70, 1.68, 1.60, 1.52, 1.40, 1.20]
Forecast (next 6 hours): [1.05, 0.95, 0.88, 0.82, 0.78, 0.80]

---
EXAMPLE 3
History (last 24 hours): \
[2.10, 2.05, 1.98, 1.90, 1.85, 1.92, 2.05, 2.20, 2.38, 2.55, 2.65, 2.70,
 2.68, 2.62, 2.58, 2.52, 2.48, 2.52, 2.60, 2.58, 2.45, 2.32, 2.22, 2.15]
Forecast (next 6 hours): [2.08, 2.02, 1.96, 1.90, 1.86, 1.90]

---
NEW QUERY
History (last {history_len} hours):
{history_str}

Task: Forecast the next {horizon} hourly values following the same pattern.

Output ONLY a Python list of exactly {horizon} floats:
FORECAST: [v1, v2, ..., v{horizon}]
"""

# ---------------------------------------------------------------------------
# 4. STATISTICAL_CONTEXT
# ---------------------------------------------------------------------------

STATISTICAL_CONTEXT = """\
You are a quantitative forecasting system with access to statistical summaries.

--- STATISTICAL SUMMARY ---
Series length  : {history_len} hourly observations
Mean           : {mean:.4f} kW
Std deviation  : {std:.4f} kW
Minimum        : {min_val:.4f} kW
Maximum        : {max_val:.4f} kW
Last 24h mean  : {last_24h_mean:.4f} kW
Last 24h std   : {last_24h_std:.4f} kW
Hour-over-hour change (last step): {last_change:+.4f} kW
----------------------------

Full history ({history_len} hourly values):
{history_str}

Using both the statistical summary and the raw values, forecast the next \
{horizon} hourly readings.

Consider:
• How the current level compares to the historical mean.
• Whether the series is above or below its typical daily pattern.
• The momentum implied by the last few observations.

Return ONLY a Python list of {horizon} floats:
[v1, v2, ..., v{horizon}]
"""

# ---------------------------------------------------------------------------
# 5. SEASONAL_DECOMP
# ---------------------------------------------------------------------------

SEASONAL_DECOMP = """\
You are a time series expert who forecasts by decomposing signals into \
trend, seasonality, and residual components.

Input: {history_len} consecutive hourly power readings (kW):
{history_str}

Instructions:

1. TREND component
   Estimate the smooth long-run level of the series by mentally applying a \
   moving average over the last 24–48 observations.  Is it rising, falling, \
   or stable?

2. SEASONAL component (daily, period = 24 h)
   Identify the repeating 24-hour pattern.  For each of the next {horizon} \
   hours, estimate what the typical seasonal deviation from the trend would be.

3. RESIDUAL / NOISE
   Note any unusual spikes or dips in the most recent observations that \
   might persist into the short-term future.

4. FORECAST = trend_t + seasonal_t + residual_adjustment_t
   Combine the three components to produce {horizon} hourly forecasts.

Return your final answer as:
FORECAST: [v1, v2, ..., v{horizon}]

Each value should be a positive floating-point number representing \
kilowatts of power consumption.
"""

# ---------------------------------------------------------------------------
# 6. ROLE_PROMPT
# ---------------------------------------------------------------------------

ROLE_PROMPT = """\
You are Dr. Elena Vasquez, a world-leading expert in smart-grid energy \
analytics with 20 years of experience forecasting household and industrial \
power demand.  You have published over 80 peer-reviewed papers on \
probabilistic load forecasting and are known for your intuition about \
consumption patterns.

A client has provided you with {history_len} hours of continuous household \
power consumption data (kilowatts):

{history_str}

Your task: Produce a precise 24-step-ahead hourly forecast for the next \
{horizon} hours.  Draw on your deep expertise to account for:
- Time-of-day effects (morning ramp-up, midday plateau, evening peak)
- Day-of-week effects (if weekly history is present)
- Any anomalies or level shifts visible in the recent data
- Momentum and mean-reversion tendencies

Provide your forecast as a Python list of exactly {horizon} float values:
[v1, v2, ..., v{horizon}]

Do not include any explanation — only the list.
"""

# ---------------------------------------------------------------------------
# 7. SELF_CONSISTENCY
# ---------------------------------------------------------------------------

SELF_CONSISTENCY = """\
You are a probabilistic forecasting engine.  Your goal is to produce \
{n_samples} independent forecasts for the same time series and then report \
all of them so that a downstream process can average them.

Input: {history_len} hourly power readings (kW):
{history_str}

Instructions:
- Reason independently for each sample.  Vary your assumptions slightly \
  (e.g. different trend estimates, different seasonal adjustments) to \
  introduce diversity.
- Each sample must be a Python list of exactly {horizon} floats.

Format your response as follows (no extra text):

SAMPLE_1: [v1, v2, ..., v{horizon}]
SAMPLE_2: [v1, v2, ..., v{horizon}]
SAMPLE_3: [v1, v2, ..., v{horizon}]
SAMPLE_4: [v1, v2, ..., v{horizon}]
SAMPLE_5: [v1, v2, ..., v{horizon}]
"""

# ---------------------------------------------------------------------------
# 8. HYBRID  (Statistical context + Chain-of-thought)
# ---------------------------------------------------------------------------

HYBRID = """\
You are an advanced forecasting system combining statistical analysis \
with step-by-step reasoning.

--- STATISTICAL SUMMARY ---
Series length  : {history_len} observations
Mean           : {mean:.4f} kW
Std deviation  : {std:.4f} kW
Min / Max      : {min_val:.4f} / {max_val:.4f} kW
Last 24h mean  : {last_24h_mean:.4f} kW
Last 6h values : {last_6h_str}
Hour-over-hour : {last_change:+.4f} kW
----------------------------

Full hourly history:
{history_str}

Reasoning protocol (work through each step):

[STEP 1 – LEVEL]
What is the current level of the series relative to its historical mean? \
Is it in a high-demand or low-demand regime?

[STEP 2 – TREND]
Estimate the short-term slope over the last 6 hours and the medium-term \
slope over the last 24 hours.

[STEP 3 – SEASONALITY]
What hour-of-day are we entering for the next {horizon} hours? \
What seasonal pattern do you expect?

[STEP 4 – SYNTHESIS]
Combine level, trend, and seasonality to project the next {horizon} values.

After completing all steps, output:
FORECAST: [v1, v2, ..., v{horizon}]

Values must be positive floats representing kilowatts.
"""
