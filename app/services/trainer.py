import logging

from src.modeling.pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class TrainerService:
    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()

    def train(self, ticker: str, period: str, interval: str, horizon: int) -> dict:
        try:
            logger.info("Training requested for %s", ticker)
            result = self.pipeline.run(
                ticker=ticker.upper(),
                period=period,
                interval=interval,
                horizon=horizon,
            )
            result["status"] = "success"
            result["message"] = "Training completed successfully."
            result["model"] = result["best_model"]
            result["source"] = "live_or_fallback"
            result["last_close"] = None
            result["forecast"] = []
            result["uncertainty"] = {}
            return result
        except Exception as exc:
            logger.exception("Training failed for %s", ticker)
            raise RuntimeError(f"Training failed for ticker {ticker.upper()}: {exc}") from exc


trainer_service = TrainerService()
