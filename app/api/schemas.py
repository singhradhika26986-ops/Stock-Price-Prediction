from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str


class SuccessResponse(BaseModel):
    status: str = "success"
    message: str


class TrainRequest(BaseModel):
    ticker: str = Field(..., description="Ticker symbol")
    period: str = Field(default="5y")
    interval: str = Field(default="1d")
    horizon: int = Field(default=5, ge=1, le=30)


class TrainResponse(SuccessResponse):
    ticker: str
    best_model: str
    model: str
    source: str = "live_or_fallback"
    last_close: float | None = None
    forecast: list[float] = []
    uncertainty: dict = {}
    metrics: dict
    artifacts: dict
    retrained_at: str


class PredictionResponse(SuccessResponse):
    ticker: str
    model_name: str
    model: str
    generated_at: str
    last_close: float
    horizon: int
    forecast: list[float]
    predictions: list[float]
    confidence_lower: list[float]
    confidence_upper: list[float]
    quantiles: dict[str, list[float]]
    probabilistic_paths: list[list[float]]
    uncertainty_score: float
    uncertainty: dict
    metrics: dict


class InsightResponse(BaseModel):
    ticker: str
    model_name: str
    metrics: dict
    feature_importance: list[dict]
    backtest_points: list[dict]
    drift_summary: dict


class MonitoringSummaryResponse(BaseModel):
    total_requests: int
    average_latency_ms: float
    error_rate: float
    top_paths: list[dict]


class MonitoringTimeseriesResponse(BaseModel):
    records: list[dict]


class TrainingHistoryResponse(BaseModel):
    records: list[dict]


class ModelMetricsResponse(BaseModel):
    ticker: str
    model_name: str
    metrics: dict
    drift_summary: dict
