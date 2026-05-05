from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from src.data.validation import StockDataValidator

logger = logging.getLogger(__name__)


class StockDataIngestor:
    def __init__(self) -> None:
        self.validator = StockDataValidator()

    def fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        logger.info("Fetching data for %s with period=%s interval=%s", ticker, period, interval)
        try:
            data: Any = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        except Exception as exc:
            logger.exception("Stock data download failed for %s", ticker)
            raise ValueError(f"Failed to fetch stock data for ticker {ticker}: {exc}") from exc

        if data is None or data.empty:
            logger.warning("Empty stock data response for %s", ticker)
            raise ValueError(f"No data returned for ticker {ticker}.")

        data = data.reset_index()
        if "Date" not in data.columns and "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "Date"})
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date").reset_index(drop=True)
        try:
            return self.validator.validate(data)
        except Exception as exc:
            logger.exception("Stock data validation failed for %s", ticker)
            raise ValueError(f"Stock data validation failed for ticker {ticker}: {exc}") from exc
