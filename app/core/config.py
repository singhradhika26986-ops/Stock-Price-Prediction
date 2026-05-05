from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Stock Price Prediction API"
    environment: str = "development"
    log_level: str = "INFO"
    default_period: str = "5y"
    default_interval: str = "1d"
    model_dir: str = "models"
    data_dir: str = "data"
    log_dir: str = "logs"
    monitoring_dir: str = "monitoring"
    forecast_horizon: int = 5
    sequence_length: int = 20
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_api_base_url: str = "http://localhost:8000"
    api_key: str = "change-me"
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60
    enable_scheduler: bool = True
    retrain_interval_minutes: int = 360
    scheduled_tickers: str = "AAPL,MSFT,GOOGL"
    default_live_lookback_period: str = "6mo"
    alpha_vantage_api_key: str = "demo"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def model_path(self) -> Path:
        return Path(self.model_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def monitoring_path(self) -> Path:
        return Path(self.monitoring_dir)

    @property
    def scheduled_ticker_list(self) -> list[str]:
        return [ticker.strip().upper() for ticker in self.scheduled_tickers.split(",") if ticker.strip()]


settings = Settings()
