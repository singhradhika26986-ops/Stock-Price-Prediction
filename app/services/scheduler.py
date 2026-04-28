from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.core.config import settings
from app.services.trainer import trainer_service

logger = logging.getLogger(__name__)


class RetrainingScheduler:
    def __init__(self) -> None:
        self.scheduler = BackgroundScheduler(timezone="UTC")
        self.started = False

    def _scheduled_retrain(self) -> None:
        for ticker in settings.scheduled_ticker_list:
            try:
                logger.info("scheduled_retraining_started", extra={"ticker": ticker})
                trainer_service.train(
                    ticker=ticker,
                    period=settings.default_period,
                    interval=settings.default_interval,
                    horizon=settings.forecast_horizon,
                )
                logger.info("scheduled_retraining_finished", extra={"ticker": ticker})
            except Exception as exc:
                logger.exception("scheduled_retraining_failed", extra={"ticker": ticker, "error": str(exc)})

    def start(self) -> None:
        if not settings.enable_scheduler or self.started:
            return
        self.scheduler.add_job(
            self._scheduled_retrain,
            trigger="interval",
            minutes=settings.retrain_interval_minutes,
            id="scheduled_retraining",
            replace_existing=True,
        )
        self.scheduler.start()
        self.started = True
        logger.info("scheduler_started", extra={"tickers": settings.scheduled_ticker_list})

    def stop(self) -> None:
        if self.started:
            self.scheduler.shutdown(wait=False)
            self.started = False


retraining_scheduler = RetrainingScheduler()
