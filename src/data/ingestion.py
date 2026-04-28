from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

from src.data.validation import StockDataValidator

logger = logging.getLogger(__name__)


class StockDataIngestor:
    def __init__(self) -> None:
        self.validator = StockDataValidator()

    def fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        logger.info("Fetching data for %s", ticker)
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        if data.empty:
            raise ValueError(f"No data returned for ticker {ticker}.")

        data = data.reset_index()
        if "Date" not in data.columns and "Datetime" in data.columns:
            data = data.rename(columns={"Datetime": "Date"})
        data["Date"] = pd.to_datetime(data["Date"])
        data = data.sort_values("Date").reset_index(drop=True)
        return self.validator.validate(data)
