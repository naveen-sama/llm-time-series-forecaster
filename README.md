# LLM Time Series Forecaster 📈

A research project benchmarking **LLM-based prompting strategies** against classical ML models for short-term electricity load forecasting. Evaluates 24 forecasters across 3 LLMs and 8 prompting strategies.

## Motivation

Recent work shows LLMs have emergent zero-shot forecasting ability. This project systematically benchmarks:
- **Can LLMs outperform classical models without training data?**
- **Which prompting strategy works best for time series?**
- **How does model size affect forecasting accuracy?**

## Experiment Design

**3 LLMs tested:**
- GPT-4o (OpenAI)
- Claude 3.5 Sonnet (Anthropic)
- LLaMA 3.1 70B (Groq)

**8 Prompting Strategies:**
1. Direct numerical prompt
2. Chain-of-thought reasoning
3. Few-shot examples (5-shot)
4. Statistical context injection (mean, std, trend)
5. Seasonal decomposition context
6. Role prompting ("You are an expert energy analyst...")
7. Self-consistency (majority vote over 5 samples)
8. Hybrid: statistical context + chain-of-thought

**Classical baselines:**
- ARIMA, SARIMA
- XGBoost, LightGBM
- Prophet
- LSTM (PyTorch)

## Results

| Forecaster | MAE (kWh) | MAPE (%) | RMSE |
|------------|-----------|----------|------|
| SARIMA | 142.3 | 4.81 | 198.4 |
| XGBoost | 118.7 | 3.94 | 167.2 |
| Prophet | 127.4 | 4.23 | 181.3 |
| LSTM | 109.2 | 3.61 | 154.8 |
| GPT-4o (direct) | 198.4 | 6.74 | 261.3 |
| GPT-4o (hybrid) | **104.1** | **3.41** | **148.2** |
| Claude 3.5 (hybrid) | 107.8 | 3.58 | 152.6 |
| LLaMA 3.1 70B (hybrid) | 121.3 | 4.01 | 169.4 |

**Key finding:** LLMs with statistical context + CoT prompting outperform all classical baselines.

## Dataset

- **Source:** UCI ML Repository — Individual Household Electric Power Consumption
- **Period:** 2006–2010, 1-minute resolution
- **Resampled to:** Hourly for forecasting
- **Task:** Predict next 24 hours given 7-day history

## Quick Start

```bash
git clone https://github.com/naveen-sama/llm-time-series-forecaster.git
cd llm-time-series-forecaster

pip install -r requirements.txt
cp .env.example .env  # Add OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY

# Download and preprocess dataset
python data/preprocess.py

# Run all benchmarks
python run_benchmark.py --all

# Run a specific strategy
python run_benchmark.py --model gpt-4o --strategy hybrid

# Generate results report
python evaluate.py --output results/report.html
```

## Project Structure

```
llm-time-series-forecaster/
├── data/
│   ├── preprocess.py
│   └── loaders.py
├── forecasters/
│   ├── llm/
│   │   ├── base_llm.py
│   │   ├── strategies.py    # 8 prompting strategies
│   │   └── prompts.py
│   └── classical/
│       ├── arima.py
│       ├── xgboost_model.py
│       └── lstm.py
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Results_Analysis.ipynb
├── run_benchmark.py
├── evaluate.py
└── requirements.txt
```

---

*Part of my AI/ML portfolio — [github.com/naveen-sama](https://github.com/naveen-sama)*
