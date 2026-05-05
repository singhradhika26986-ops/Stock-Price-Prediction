from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.core.config import settings
from src.data.ingestion import StockDataIngestor
from src.features.engineering import FeatureEngineer
from src.modeling.sequence import build_inference_sequence

logger = logging.getLogger(__name__)


class PredictorService:
    def __init__(self) -> None:
        self.ingestor = StockDataIngestor()
        self.engineer = FeatureEngineer()

    def _bundle_path(self, ticker: str) -> Path:
        return settings.model_path / f"{ticker.upper()}_best_bundle.joblib"

    def _metrics_path(self, ticker: str) -> Path:
        return settings.model_path / f"{ticker.upper()}_metrics.json"

    def _importance_path(self, ticker: str) -> Path:
        return settings.model_path / f"{ticker.upper()}_feature_importance.csv"

    def _backtest_path(self, ticker: str) -> Path:
        return settings.model_path / f"{ticker.upper()}_backtest.csv"

    def _drift_path(self, ticker: str) -> Path:
        return settings.model_path / f"{ticker.upper()}_drift.json"

    def _load_bundle(self, ticker: str) -> dict:
        bundle_path = self._bundle_path(ticker)
        if not bundle_path.exists():
            raise FileNotFoundError(f"Model bundle not found for ticker {ticker.upper()}. Train the model first.")
        try:
            bundle = joblib.load(bundle_path)
            if bundle.get("model_name") == "lstm" and "model" not in bundle:
                from tensorflow.keras.models import load_model

                bundle["model"] = load_model(bundle["model_path"])
            return bundle
        except Exception as exc:
            logger.exception("Model bundle load failed for %s", ticker)
            raise RuntimeError(f"Failed to load model for ticker {ticker.upper()}: {exc}") from exc

    def predict(self, ticker: str, days_ahead: int) -> dict:
        try:
            bundle = self._load_bundle(ticker)
            history = self.ingestor.fetch(
                ticker=ticker.upper(),
                period=settings.default_live_lookback_period,
                interval=settings.default_interval,
            )
            feature_frame = self.engineer.transform(history, horizon=bundle["horizon"])
            feature_frame = feature_frame.dropna(subset=["Date", "Close"]).copy()
            if feature_frame.empty:
                feature_frame = history.dropna(subset=["Date", "Close"]).copy()
                if feature_frame.empty:
                    raise FileNotFoundError(f"Not enough recent data available for {ticker.upper()} to generate a prediction.")

            feature_columns = bundle["feature_columns"]
            model_name = bundle["model_name"]
            residual_std = float(bundle.get("residual_std", 0.0))
            last_close = float(feature_frame["Close"].iloc[-1])

            if model_name == "moving_average_fallback":
                base_pred = float(bundle.get("moving_average", last_close))
            elif model_name == "lstm":
                sequence = build_inference_sequence(
                    feature_frame[feature_columns],
                    bundle["feature_scaler"],
                    bundle["sequence_length"],
                )
                preds_scaled = bundle["model"].predict(sequence, verbose=0).reshape(-1)
                base_pred = bundle["target_scaler"].inverse_transform(preds_scaled.reshape(-1, 1)).reshape(-1)[0]
            else:
                if not set(feature_columns).issubset(feature_frame.columns):
                    feature_frame = self.engineer.transform(history, horizon=bundle["horizon"])
                latest = feature_frame[feature_columns].iloc[[-1]]
                if bundle.get("use_scaler", False):
                    latest = bundle["feature_scaler"].transform(latest)
                base_pred = float(bundle["model"].predict(latest)[0])

            quantiles = {"p05": [], "p25": [], "p50": [], "p75": [], "p95": []}
            predictions = []
            lower = []
            upper = []
            probabilistic_paths = []
            residual_quantiles = bundle.get("residual_quantiles", {})
            uncertainty_score = round(float(bundle.get("residual_std", 0.0) / max(last_close, 1e-8)), 6)

            for step in range(1, days_ahead + 1):
                drifted = base_pred + (step - 1) * (base_pred - last_close) * 0.15
                spread = residual_std * np.sqrt(step) if residual_std else abs(drifted - last_close) * 0.1
                predictions.append(round(float(drifted), 4))
                lower.append(round(float(drifted - 1.96 * spread), 4))
                upper.append(round(float(drifted + 1.96 * spread), 4))
                quantiles["p05"].append(round(float(drifted + residual_quantiles.get("q05", -1.96 * spread) * np.sqrt(step)), 4))
                quantiles["p25"].append(round(float(drifted + residual_quantiles.get("q25", -0.67 * spread) * np.sqrt(step)), 4))
                quantiles["p50"].append(round(float(drifted + residual_quantiles.get("q50", 0.0) * np.sqrt(step)), 4))
                quantiles["p75"].append(round(float(drifted + residual_quantiles.get("q75", 0.67 * spread) * np.sqrt(step)), 4))
                quantiles["p95"].append(round(float(drifted + residual_quantiles.get("q95", 1.96 * spread) * np.sqrt(step)), 4))

            rng = np.random.default_rng(42)
            for _ in range(10):
                path = []
                for step, prediction in enumerate(predictions, start=1):
                    sample = prediction + rng.normal(0, residual_std if residual_std else max(prediction * 0.01, 0.1)) * np.sqrt(step)
                    path.append(round(float(sample), 4))
                probabilistic_paths.append(path)

            metrics_payload = self.insights(ticker).get("metrics", {})
            return {
                "status": "success",
                "message": "Prediction generated successfully.",
                "ticker": ticker.upper(),
                "model_name": model_name,
                "model": model_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "last_close": round(last_close, 4),
                "horizon": days_ahead,
                "forecast": predictions,
                "predictions": predictions,
                "confidence_lower": lower,
                "confidence_upper": upper,
                "quantiles": quantiles,
                "probabilistic_paths": probabilistic_paths,
                "uncertainty_score": uncertainty_score,
                "uncertainty": {
                    "score": uncertainty_score,
                    "lower_band": lower,
                    "upper_band": upper,
                    "quantiles": quantiles,
                },
                "metrics": metrics_payload,
            }
        except Exception as exc:
            logger.exception("Prediction failed for %s", ticker)
            raise RuntimeError(f"Prediction failed for ticker {ticker.upper()}: {exc}") from exc

    def insights(self, ticker: str) -> dict:
        try:
            bundle = self._load_bundle(ticker)
            metrics_path = self._metrics_path(ticker)
            importance_path = self._importance_path(ticker)
            backtest_path = self._backtest_path(ticker)
            drift_path = self._drift_path(ticker)

            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
            feature_importance = []
            if importance_path.exists():
                feature_importance = pd.read_csv(importance_path).to_dict(orient="records")
            backtest_points = pd.read_csv(backtest_path).to_dict(orient="records") if backtest_path.exists() else []
            drift_summary = json.loads(drift_path.read_text(encoding="utf-8")) if drift_path.exists() else {}

            return {
                "ticker": ticker.upper(),
                "model_name": bundle["model_name"],
                "metrics": metrics,
                "feature_importance": feature_importance,
                "backtest_points": backtest_points,
                "drift_summary": drift_summary,
            }
        except Exception as exc:
            logger.exception("Insights fetch failed for %s", ticker)
            raise RuntimeError(f"Failed to load insights for ticker {ticker.upper()}: {exc}") from exc


predictor_service = PredictorService()
