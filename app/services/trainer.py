from src.modeling.pipeline import TrainingPipeline


class TrainerService:
    def __init__(self) -> None:
        self.pipeline = TrainingPipeline()

    def train(self, ticker: str, period: str, interval: str, horizon: int) -> dict:
        return self.pipeline.run(
            ticker=ticker.upper(),
            period=period,
            interval=interval,
            horizon=horizon,
        )


trainer_service = TrainerService()
