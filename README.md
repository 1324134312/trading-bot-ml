# Trading Bot ML

An educational Python project that fetches UK and US stock data, builds technical-indicator features, trains machine-learning models, and backtests simple trading strategies.

It is a learning project, not a live trading system, and not financial advice. There is no warranty that any strategy or model will make a profit. Use it for research, backtests and paper trading only.

This project was built with AI assistance. Read the code before you run it, and keep API keys out of git.

## What it does

- Downloads historical data for US tickers (Apple, Microsoft, and other defaults in `config/settings.py`) and UK tickers (FTSE names in `config/uk_config.py`), mainly through Yahoo Finance (`yfinance`)
- Can also pull intraday bars from [Twelve Data](https://twelvedata.com/) when you set an API key
- Optional helpers for IEX Cloud market data and Tradier paper-trading accounts
- Builds features including RSI, MACD, Bollinger Bands, moving averages, Stochastic, ATR, ADX and related indicators
- Trains classifiers with XGBoost and Random Forest, plus an optional ensemble that adds LightGBM, Optuna hyperparameter tuning and SMOTE
- Generates day-trading and swing-trading signals and runs them through a backtester with position limits, stop-loss and take-profit

This repository does **not** include neural-network models, Alpaca integration, or a Finnhub client. Those live in a separate earlier (non-ML) trading-bot project. No trained model files or market-data dumps are included; fetch and train locally.

## Setup

Python 3.10 or 3.11 is recommended (the original project was run on 3.11).

```bash
cd trading-bot-ml-github
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate with `source venv/bin/activate`.

Yahoo Finance scripts work without an API key. For Twelve Data, IEX Cloud or Tradier, copy `.env.example` to `.env` and fill in the keys you need:

```bash
copy .env.example .env
```

The code calls `load_dotenv()` so a `.env` file in the project root is picked up automatically. You can also set the variables in your shell instead.

## Environment variables

| Variable | Used by | Required? |
|----------|---------|-----------|
| `TWELVE_DATA_API_KEY` | `src/fetch_21days_twelve.py`, `src/fetch_21days_with_rate_limit.py`, `src/fetch_with_twelvedata.py`, `src/get_today_5min_data.py` | Yes, for those scripts |
| `IEX_CLOUD_API_KEY` | `src/iex_cloud_fetcher.py` | Optional |
| `TRADIER_ACCESS_TOKEN` | `src/tradier_integration.py`, `src/tradier_real.py` | Optional |
| `TRADIER_ACCOUNT_ID` | Same Tradier scripts | Optional |

Do not commit `.env`. It is already in `.gitignore`.

## How to run

Run commands from the project root so `src` and `config` imports resolve.

Fetch and validate the default US daily dataset (5 years, daily bars) into `data/raw/`:

```bash
python src/main.py
```

Build feature CSVs into `data/processed/`:

```bash
python src/process_features.py
```

Train a model (XGBoost by default; Random Forest is also supported):

```bash
python src/train_model.py
```

Train the ensemble (XGBoost, LightGBM and Random Forest, with optional Optuna tuning):

```bash
python src/train_advanced.py
```

UK data path:

```bash
python src/uk_data_collector.py
python src/uk_feature_engineering.py
python src/train_uk_day_trading.py
```

Twelve Data (needs `TWELVE_DATA_API_KEY`):

```bash
python src/fetch_with_twelvedata.py
python src/fetch_21days_twelve.py
python src/fetch_21days_with_rate_limit.py
```

Other scripts in `src/` cover further data fetches, backtests, model comparisons and settings experiments. Exploratory notebooks are in `notebooks/`. Downloaded data lands in `data/` and trained models in `models/` (both gitignored).

## Project structure

```
config/
  settings.py              # US tickers, paths, default period and interval
  uk_config.py             # UK tickers and intraday settings
src/
  main.py                  # Default US daily data fetch and validation
  data_collection/         # yfinance fetcher and data quality checks
  feature_engineering/     # Indicators and feature builder
  models/                  # Training, ensemble training and prediction
  strategies/              # Day-trading and swing-trading strategies
  backtesting/             # Backtester
  fetch_*.py, uk_*.py      # Extra US/UK and Twelve Data fetch scripts
  train_*.py               # Training entry points
  *backtest*.py, test_*.py # Backtests and experiment scripts
  iex_cloud_fetcher.py     # Optional IEX Cloud client
  tradier_*.py             # Optional Tradier paper-trading helpers
notebooks/                 # Small exploration notebooks
example_kaggle_usage.py    # Optional Kaggle dataset example (US data)
```

## Disclaimer

This is an educational project. It is not financial advice, and it is not a product for live trading. Backtests are simplified, can overfit, and omit many real-world costs and constraints. Paper-trade at most, and never risk money you cannot afford to lose. You are responsible for how you use this code and for complying with any data provider's terms.
