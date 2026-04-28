from fastapi import APIRouter, Depends, HTTPException

from app.api.schemas import (
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


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/train", response_model=TrainResponse, dependencies=[Depends(require_api_key)])
def train_model(request: TrainRequest) -> TrainResponse:
    result = trainer_service.train(
        ticker=request.ticker,
        period=request.period,
        interval=request.interval,
        horizon=request.horizon,
    )
    return TrainResponse(**result)


@router.get("/predict/{ticker}", response_model=PredictionResponse, dependencies=[Depends(require_api_key)])
def predict(ticker: str, days_ahead: int = 5) -> PredictionResponse:
    try:
        result = predictor_service.predict(ticker=ticker, days_ahead=days_ahead)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PredictionResponse(**result)


@router.get("/insights/{ticker}", response_model=InsightResponse, dependencies=[Depends(require_api_key)])
def insights(ticker: str) -> InsightResponse:
    try:
        result = predictor_service.insights(ticker=ticker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return InsightResponse(**result)


@router.get("/metrics/{ticker}", response_model=ModelMetricsResponse, dependencies=[Depends(require_api_key)])
def model_metrics(ticker: str) -> ModelMetricsResponse:
    try:
        result = predictor_service.insights(ticker=ticker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ModelMetricsResponse(
        ticker=result["ticker"],
        model_name=result["model_name"],
        metrics=result["metrics"],
        drift_summary=result["drift_summary"],
    )


@router.get("/monitoring/summary", response_model=MonitoringSummaryResponse, dependencies=[Depends(require_api_key)])
def monitoring_summary() -> MonitoringSummaryResponse:
    return MonitoringSummaryResponse(**monitoring_service.summary())


@router.get("/monitoring/requests", response_model=MonitoringTimeseriesResponse, dependencies=[Depends(require_api_key)])
def monitoring_requests() -> MonitoringTimeseriesResponse:
    return MonitoringTimeseriesResponse(records=monitoring_service.request_timeseries())


@router.get("/monitoring/training-history", response_model=TrainingHistoryResponse, dependencies=[Depends(require_api_key)])
def training_history(ticker: str | None = None) -> TrainingHistoryResponse:
    return TrainingHistoryResponse(records=monitoring_service.training_history(ticker=ticker))
