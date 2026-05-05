import logging

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse, JSONResponse

from app.api.schemas import (
    ErrorResponse,
    InsightResponse,
    ModelMetricsResponse,
    MonitoringSummaryResponse,
    MonitoringTimeseriesResponse,
    PredictionResponse,
    TrainingHistoryResponse,
    TrainRequest,
    TrainResponse,
)
from app.security import require_api_key
from app.services.monitoring import monitoring_service
from app.services.predictor import predictor_service
from app.services.trainer import trainer_service

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/", response_class=HTMLResponse)
def landing_page() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Stock Price Prediction Live App</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --card: #ffffff;
      --text: #1c2434;
      --muted: #6b7280;
      --primary: #0f766e;
      --accent: #f59e0b;
      --border: #d7deea;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: radial-gradient(circle at top, #e8f8f4, #f6f8fb 45%);
      color: var(--text);
    }
    .wrap {
      max-width: 960px;
      margin: 0 auto;
      padding: 40px 20px 80px;
    }
    .hero {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 18px 50px rgba(15, 23, 42, 0.08);
    }
    h1 { margin: 0 0 10px; font-size: 2.2rem; }
    p { color: var(--muted); line-height: 1.6; }
    .controls {
      display: grid;
      grid-template-columns: 1.4fr 1fr auto auto;
      gap: 12px;
      margin-top: 24px;
    }
    input, select, button {
      border-radius: 14px;
      border: 1px solid var(--border);
      padding: 14px 16px;
      font-size: 1rem;
    }
    button {
      border: none;
      cursor: pointer;
      font-weight: 600;
    }
    .predict { background: var(--primary); color: white; }
    .train { background: var(--accent); color: #1f2937; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 22px;
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
    }
    .label { color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }
    .value { font-size: 1.6rem; font-weight: 700; }
    .section {
      margin-top: 22px;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 18px;
    }
    .status {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #eefbf6;
      color: #0f5132;
      display: none;
    }
    .error {
      background: #fff1f2;
      color: #9f1239;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10px;
    }
    th, td {
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      font-size: 0.95rem;
    }
    .links {
      margin-top: 18px;
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }
    .links a {
      color: var(--primary);
      text-decoration: none;
      font-weight: 600;
    }
    @media (max-width: 760px) {
      .controls { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="hero">
      <h1>Stock Price Prediction Live App</h1>
      <p>Enter a stock ticker, optionally train or refresh the model, and generate a live forecast with uncertainty bands from this deployed service.</p>

      <div class="controls">
        <input id="ticker" value="AAPL" placeholder="Ticker symbol, e.g. AAPL" />
        <select id="days">
          <option value="3">3 days</option>
          <option value="5" selected>5 days</option>
          <option value="7">7 days</option>
          <option value="10">10 days</option>
        </select>
        <button class="predict" onclick="runPrediction()">Predict</button>
        <button class="train" onclick="trainModel()">Train / Refresh</button>
      </div>

      <div id="status" class="status"></div>

      <div class="grid">
        <div class="card">
          <div class="label">Model</div>
          <div id="modelName" class="value">-</div>
        </div>
        <div class="card">
          <div class="label">Last Close</div>
          <div id="lastClose" class="value">-</div>
        </div>
        <div class="card">
          <div class="label">First Forecast</div>
          <div id="firstForecast" class="value">-</div>
        </div>
        <div class="card">
          <div class="label">Uncertainty</div>
          <div id="uncertainty" class="value">-</div>
        </div>
      </div>

      <div class="section">
        <h3>Forecast Table</h3>
        <table>
          <thead>
            <tr>
              <th>Day</th>
              <th>Prediction</th>
              <th>Lower</th>
              <th>Upper</th>
            </tr>
          </thead>
          <tbody id="forecastBody"></tbody>
        </table>
      </div>

      <div class="section">
        <h3>Model Metrics</h3>
        <pre id="metricsBox" style="white-space: pre-wrap; color: #334155;">Run a prediction to load metrics.</pre>
      </div>

      <div class="links">
        <a href="/health" target="_blank">Health Check</a>
        <a href="/docs" target="_blank">API Docs</a>
        <a href="https://github.com/singhradhika26986-ops/Stock-Price-Prediction" target="_blank">GitHub Repo</a>
      </div>
    </div>
  </div>

  <script>
    function setStatus(message, isError = false) {
      const box = document.getElementById("status");
      box.style.display = "block";
      box.className = isError ? "status error" : "status";
      box.textContent = message;
    }

    async function trainModel() {
      const ticker = document.getElementById("ticker").value.trim().toUpperCase();
      if (!ticker) return setStatus("Please enter a ticker symbol.", true);
      setStatus("Training started. This can take 1-3 minutes depending on the service load.");
      try {
        const response = await fetch("/public/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ticker, period: "5y", interval: "1d", horizon: Number(document.getElementById("days").value) })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || "Training failed.");
        setStatus(`Training completed. Best model: ${data.best_model}.`);
        await runPrediction();
      } catch (error) {
        setStatus(error.message, true);
      }
    }

    async function runPrediction() {
      const ticker = document.getElementById("ticker").value.trim().toUpperCase();
      const days = document.getElementById("days").value;
      if (!ticker) return setStatus("Please enter a ticker symbol.", true);
      setStatus("Fetching live prediction...");
      try {
        const response = await fetch(`/public/predict/${ticker}?days_ahead=${days}`);
        const data = await response.json();
        if (!response.ok) throw new Error(data.message || "Prediction failed.");

        document.getElementById("modelName").textContent = data.model_name;
        document.getElementById("lastClose").textContent = data.last_close.toFixed(2);
        document.getElementById("firstForecast").textContent = data.predictions[0].toFixed(2);
        document.getElementById("uncertainty").textContent = data.uncertainty_score.toFixed(4);

        const body = document.getElementById("forecastBody");
        body.innerHTML = "";
        data.predictions.forEach((value, index) => {
          const row = document.createElement("tr");
          row.innerHTML = `<td>${index + 1}</td><td>${value.toFixed(2)}</td><td>${data.confidence_lower[index].toFixed(2)}</td><td>${data.confidence_upper[index].toFixed(2)}</td>`;
          body.appendChild(row);
        });

        const metricsResponse = await fetch(`/public/metrics/${ticker}`);
        const metricsData = await metricsResponse.json();
        if (metricsResponse.ok) {
          document.getElementById("metricsBox").textContent = JSON.stringify(metricsData.metrics, null, 2);
        }

        setStatus(`Prediction ready for ${ticker}.`);
      } catch (error) {
        setStatus(error.message + " If the model is missing, click Train / Refresh first.", true);
      }
    }
  </script>
</body>
</html>
"""


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/train", response_model=TrainResponse, responses={500: {"model": ErrorResponse}}, dependencies=[Depends(require_api_key)])
def train_model(request: TrainRequest):
    try:
        result = trainer_service.train(
            ticker=request.ticker,
            period=request.period,
            interval=request.interval,
            horizon=request.horizon,
        )
        return TrainResponse(**result)
    except Exception as exc:
        logger.exception("Private training endpoint failed for %s", request.ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.get("/predict/{ticker}", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}}, dependencies=[Depends(require_api_key)])
def predict(ticker: str, days_ahead: int = 5):
    try:
        result = predictor_service.predict(ticker=ticker, days_ahead=days_ahead)
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Private prediction endpoint failed for %s", ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.post("/public/train", response_model=TrainResponse, responses={500: {"model": ErrorResponse}})
def public_train_model(request: TrainRequest):
    try:
        result = trainer_service.train(
            ticker=request.ticker,
            period=request.period,
            interval=request.interval,
            horizon=request.horizon,
        )
        return TrainResponse(**result)
    except Exception as exc:
        logger.exception("Public training endpoint failed for %s", request.ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.get("/public/predict/{ticker}", response_model=PredictionResponse, responses={500: {"model": ErrorResponse}})
def public_predict(ticker: str, days_ahead: int = 5):
    try:
        result = predictor_service.predict(ticker=ticker, days_ahead=days_ahead)
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Public prediction endpoint failed for %s", ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.get("/insights/{ticker}", response_model=InsightResponse, responses={500: {"model": ErrorResponse}}, dependencies=[Depends(require_api_key)])
def insights(ticker: str):
    try:
        result = predictor_service.insights(ticker=ticker)
        return InsightResponse(**result)
    except Exception as exc:
        logger.exception("Insights endpoint failed for %s", ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.get("/metrics/{ticker}", response_model=ModelMetricsResponse, responses={500: {"model": ErrorResponse}}, dependencies=[Depends(require_api_key)])
def model_metrics(ticker: str):
    try:
        result = predictor_service.insights(ticker=ticker)
        return ModelMetricsResponse(
            ticker=result["ticker"],
            model_name=result["model_name"],
            metrics=result["metrics"],
            drift_summary=result["drift_summary"],
        )
    except Exception as exc:
        logger.exception("Metrics endpoint failed for %s", ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.get("/public/metrics/{ticker}", response_model=ModelMetricsResponse, responses={500: {"model": ErrorResponse}})
def public_model_metrics(ticker: str):
    try:
        result = predictor_service.insights(ticker=ticker)
        return ModelMetricsResponse(
            ticker=result["ticker"],
            model_name=result["model_name"],
            metrics=result["metrics"],
            drift_summary=result["drift_summary"],
        )
    except Exception as exc:
        logger.exception("Public metrics endpoint failed for %s", ticker)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


@router.get("/monitoring/summary", response_model=MonitoringSummaryResponse, dependencies=[Depends(require_api_key)])
def monitoring_summary() -> MonitoringSummaryResponse:
    return MonitoringSummaryResponse(**monitoring_service.summary())


@router.get("/monitoring/requests", response_model=MonitoringTimeseriesResponse, dependencies=[Depends(require_api_key)])
def monitoring_requests() -> MonitoringTimeseriesResponse:
    return MonitoringTimeseriesResponse(records=monitoring_service.request_timeseries())


@router.get("/monitoring/training-history", response_model=TrainingHistoryResponse, dependencies=[Depends(require_api_key)])
def training_history(ticker: str | None = None) -> TrainingHistoryResponse:
    return TrainingHistoryResponse(records=monitoring_service.training_history(ticker=ticker))
