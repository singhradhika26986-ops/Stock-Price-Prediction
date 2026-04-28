from __future__ import annotations

import json
from collections import Counter
from statistics import mean

import pandas as pd

from app.core.config import settings


class MonitoringService:
    def _jsonl_path(self):
        return settings.monitoring_path / "api_requests.jsonl"

    def _training_path(self):
        return settings.monitoring_path / "training_history.jsonl"

    def summary(self) -> dict:
        path = self._jsonl_path()
        if not path.exists():
            return {
                "total_requests": 0,
                "average_latency_ms": 0.0,
                "error_rate": 0.0,
                "top_paths": [],
            }

        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            return {
                "total_requests": 0,
                "average_latency_ms": 0.0,
                "error_rate": 0.0,
                "top_paths": [],
            }

        status_codes = [row["status_code"] for row in rows]
        path_counts = Counter(row["path"] for row in rows)
        return {
            "total_requests": len(rows),
            "average_latency_ms": round(mean(row["latency_ms"] for row in rows), 2),
            "error_rate": round(sum(code >= 400 for code in status_codes) / len(status_codes), 4),
            "top_paths": [{"path": path, "count": count} for path, count in path_counts.most_common(10)],
        }

    def request_timeseries(self) -> list[dict]:
        path = self._jsonl_path()
        if not path.exists():
            return []
        frame = pd.read_json(path, lines=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values("timestamp")
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        return frame.to_dict(orient="records")

    def training_history(self, ticker: str | None = None) -> list[dict]:
        path = self._training_path()
        if not path.exists():
            return []
        frame = pd.read_json(path, lines=True)
        if ticker:
            frame = frame[frame["ticker"].str.upper() == ticker.upper()]
        if frame.empty:
            return []
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.sort_values("timestamp")
        frame["timestamp"] = frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        return frame.to_dict(orient="records")


monitoring_service = MonitoringService()
