from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from app.core.config import settings
from src.modeling.pipeline import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stock prediction models.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, for example AAPL")
    parser.add_argument("--period", default=settings.default_period, help="Historical lookback period")
    parser.add_argument("--interval", default=settings.default_interval, help="Candlestick interval")
    parser.add_argument("--horizon", type=int, default=settings.forecast_horizon, help="Forecast horizon in days")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
    args = parse_args()
    pipeline = TrainingPipeline()
    result = pipeline.run(
        ticker=args.ticker.upper(),
        period=args.period,
        interval=args.interval,
        horizon=args.horizon,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
