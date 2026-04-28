# Stock Price Prediction System

Production-ready stock price prediction system with end-to-end ingestion, validation, feature engineering, experiment tracking, model training, backtesting, live API serving, scheduled retraining, and a real-time dashboard.

## Features

- Historical and near real-time OHLCV ingestion using `yfinance`
- Custom validation for schema integrity, timestamp validity, and OHLCV consistency
- Leakage-aware feature engineering with moving averages, RSI, MACD, lag features, returns, and volatility
- Model suite covering Linear Regression, Random Forest, XGBoost, and LSTM
- Walk-forward backtesting with RMSE, MAE, MAPE, and directional accuracy
- Probabilistic forecasts with confidence bands, quantiles, and sampled paths
- Explainability through feature importance for tree-based models
- MLflow-compatible experiment tracking with file-based fallback logs
- FastAPI backend with `/train`, `/predict/{ticker}`, `/metrics/{ticker}`, `/insights/{ticker}`, and monitoring endpoints
- API key authentication, rate limiting, structured JSON logging, and latency monitoring
- APScheduler-based automated retraining for configured ticker sets
- Streamlit dashboard for live forecasts, confidence bands, backtests, retraining trends, and API monitoring
- Docker packaging plus GitHub Actions CI/CD hooks for testing and deployment
- Render Blueprint and Railway config included for cloud deployment

## Project Structure

```text
.
|-- app/
|   |-- api/
|   |-- core/
|   |-- services/
|   `-- main.py
|-- dashboard/
|   `-- streamlit_app.py
|-- src/
|   |-- data/
|   |-- features/
|   `-- modeling/
|-- tests/
|-- .github/workflows/
|-- data/
|-- logs/
|-- models/
|-- monitoring/
|-- train.py
|-- requirements.txt
`-- Dockerfile
```

## Architecture

1. `train.py` triggers the full training pipeline.
2. `src/data/ingestion.py` fetches market data and passes it through `src/data/validation.py`.
3. `src/features/engineering.py` builds technical indicators and supervised learning features.
4. `src/modeling/pipeline.py` trains candidate models, performs walk-forward backtests, stores drift artifacts, and tracks experiments.
5. `app/main.py` starts the FastAPI app with monitoring middleware, rate limiting, and scheduled retraining.
6. `dashboard/streamlit_app.py` consumes the live API to show predictions, uncertainty, metrics, backtests, and operational health.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Important environment variables:

- `API_KEY`: secret required by protected API endpoints
- `ENABLE_SCHEDULER`: enables automated retraining
- `RETRAIN_INTERVAL_MINUTES`: retraining cadence
- `SCHEDULED_TICKERS`: comma-separated tickers for recurring retrains
- `STREAMLIT_API_BASE_URL`: dashboard target API URL

## Training

```bash
python train.py --ticker AAPL --period 5y --horizon 5
```

Saved artifacts:

- `models/{ticker}_best_bundle.joblib`
- `models/{ticker}_metrics.json`
- `models/{ticker}_feature_importance.csv`
- `models/{ticker}_backtest.csv`
- `models/{ticker}_drift.json`
- `monitoring/mlruns/` or `monitoring/experiment_runs.jsonl`

## API

```bash
uvicorn app.main:app --reload
```

Key endpoints:

- `GET /health`
- `POST /train`
- `GET /predict/{ticker}?days_ahead=5`
- `GET /insights/{ticker}`
- `GET /metrics/{ticker}`
- `GET /monitoring/summary`
- `GET /monitoring/requests`
- `GET /monitoring/training-history?ticker=AAPL`

Protected endpoints require the `x-api-key` header.

## Dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

The dashboard includes:

- live forecast curves with confidence and quantile bands
- sampled probabilistic future paths
- predicted-vs-actual backtesting charts
- retraining metric trends
- latency and request monitoring snapshots

## Testing

```bash
pytest
python -m compileall app src dashboard train.py
```

## Docker

```bash
docker build -t stock-predictor .
docker run -p 8000:8000 --env-file .env stock-predictor
```

## CI/CD

- `.github/workflows/ci.yml` installs dependencies, runs compile checks, and executes API smoke tests
- `.github/workflows/deploy-render.yml` triggers a Render deploy hook using the `RENDER_DEPLOY_HOOK_URL` GitHub secret

## Deployment

### Render / Railway

- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Configure environment variables from `.env.example`
- Set a strong `API_KEY`
- Use persistent storage for `models/`, `monitoring/`, and `logs/` if you retrain in production
- If the dashboard is deployed separately, set `STREAMLIT_API_BASE_URL` to the public API URL and reuse the same `API_KEY`
- For GitHub Actions based deployment, add `RENDER_DEPLOY_HOOK_URL` as a repository secret

### Render Blueprint

- `render.yaml` defines two web services:
- `stock-price-prediction-api` for FastAPI
- `stock-price-prediction-dashboard` for Streamlit
- The API service uses `healthCheckPath: /health`
- Keep `API_KEY` and `STREAMLIT_API_BASE_URL` as unsynced secrets in Render

### Railway

- `railway.toml` configures Dockerfile-based deploys and a `uvicorn` start command bound to `0.0.0.0:$PORT`
- Railway injects the `PORT` variable at runtime, so do not hardcode a cloud port
- Add service variables from `.env.example` in the Railway Variables tab

### Manual Cloud Steps

1. Push this repository to GitHub.
2. In Render, create a new Blueprint or Web Service from the repository and allow auto-deploy on push.
3. In Railway, create a new project from the repository and import the Docker service.
4. Set runtime secrets such as `API_KEY`, scheduler settings, and dashboard API URL in the cloud dashboard.
5. After deploy, open `/health` on the public API URL to verify the service is healthy.
6. Open the dashboard URL and confirm ticker inputs return predictions.

## Notes

- This project is for analytics and education, not financial advice.
- A truly always-on public deployment still requires your cloud account, secrets, and a successful deploy from this repository.
- `yfinance` is the default data source; you can extend `src/data/ingestion.py` to support Alpha Vantage or other providers.
