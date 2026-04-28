from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RequestBucket:
    timestamps: deque[float] = field(default_factory=deque)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.buckets: dict[str, RequestBucket] = defaultdict(RequestBucket)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/health":
            return await call_next(request)

        client_host = request.client.host if request.client else "unknown"
        key = f"{client_host}:{request.url.path}"
        now = time.time()
        bucket = self.buckets[key]

        while bucket.timestamps and now - bucket.timestamps[0] > settings.rate_limit_window_seconds:
            bucket.timestamps.popleft()

        if len(bucket.timestamps) >= settings.rate_limit_requests:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Please retry later."},
            )

        bucket.timestamps.append(now)
        return await call_next(request)


class MonitoringMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            client_host = request.client.host if request.client else "unknown"
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": request.url.path,
                "method": request.method,
                "status_code": status_code,
                "latency_ms": latency_ms,
                "client_host": client_host,
            }
            logger.info("api_request", extra=record)
            metrics_file = settings.monitoring_path / "api_requests.jsonl"
            with metrics_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
