from __future__ import annotations

import json
from pathlib import Path

from app.core.config import settings

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


class ExperimentTracker:
    def __init__(self) -> None:
        self.tracking_dir = settings.monitoring_path / "mlruns"
        self.tracking_dir.mkdir(parents=True, exist_ok=True)
        if mlflow:
            mlflow.set_tracking_uri(self.tracking_dir.resolve().as_uri())
            mlflow.set_experiment("stock-price-prediction")

    def log_run(self, ticker: str, model_name: str, params: dict, metrics: dict, artifacts: dict) -> None:
        if mlflow:
            with mlflow.start_run(run_name=f"{ticker}-{model_name}"):
                mlflow.log_params({"ticker": ticker, **params})
                flat_metrics = self._flatten_metrics(metrics)
                mlflow.log_metrics(flat_metrics)
                for artifact_name, artifact_path in artifacts.items():
                    if Path(artifact_path).exists():
                        mlflow.log_artifact(artifact_path, artifact_path=f"artifacts/{artifact_name}")
        else:
            fallback = {
                "ticker": ticker,
                "model_name": model_name,
                "params": params,
                "metrics": metrics,
                "artifacts": artifacts,
            }
            target = settings.monitoring_path / "experiment_runs.jsonl"
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(fallback) + "\n")

    @staticmethod
    def _flatten_metrics(metrics: dict) -> dict:
        flattened = {}
        for key, value in metrics.items():
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float)):
                        flattened[f"{key}_{nested_key}"] = float(nested_value)
            elif isinstance(value, (int, float)):
                flattened[key] = float(value)
        return flattened
