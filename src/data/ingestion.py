from __future__ import annotations

import logging
import time
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd
import yfinance as yf

from app.core.config import settings
from src.data.validation import StockDataValidator

logger = logging.getLogger(__name__)


class StockDataIngestor:
    def __init__(self) -> None:
        self.validator = StockDataValidator()
        self.sample_data_path = Path(__file__).resolve().parent / "sample_stock_data.csv"

    def fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        normalized_ticker = self._normalize_ticker(ticker)
        errors: list[str] = []

        for source_name, fetcher in [
            ("yfinance", lambda: self._fetch_yfinance(normalized_ticker)),
            ("alpha_vantage", lambda: self._fetch_alpha_vantage(normalized_ticker)),
            ("sample_cache", self._fetch_sample_cache),
        ]:
            try:
                frame = fetcher()
                validated = self._prepare_and_validate(frame, normalized_ticker, source_name)
                logger.info("Fetched %s rows for %s from %s", len(validated), normalized_ticker, source_name)
                return validated
            except Exception as exc:
                message = f"{source_name} failed: {exc}"
                errors.append(message)
                logger.warning(message)

        error_message = f"All stock data sources failed for {normalized_ticker}. " + " | ".join(errors)
        logger.error(error_message)
        raise ValueError(error_message)

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("Ticker cannot be empty.")
        return normalized

    def _fetch_yfinance(self, ticker: str) -> pd.DataFrame:
        last_error: Exception | None = None
        for attempt in range(1, 3):
            try:
                logger.info("Fetching %s from yfinance attempt=%s period=2y interval=1d", ticker, attempt)
                data: Any = yf.download(
                    ticker,
                    period="2y",
                    interval="1d",
                    auto_adjust=False,
                    progress=False,
                    session=self._build_yfinance_session(),
                    threads=False,
                )
                logger.info("yfinance raw shape for %s attempt=%s: %s", ticker, attempt, getattr(data, "shape", None))
                if data is not None and not data.empty:
                    logger.info("yfinance preview for %s attempt=%s:\n%s", ticker, attempt, data.head().to_string())
                if data is not None and not data.empty:
                    return data
                last_error = ValueError("yfinance returned empty data.")
            except Exception as exc:
                last_error = exc
                logger.warning("yfinance attempt %s failed for %s: %s", attempt, ticker, exc)
            time.sleep(attempt)
        raise ValueError(last_error or "yfinance returned no usable data.")

    @staticmethod
    def _build_yfinance_session():
        try:
            from curl_cffi import requests

            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                }
            )
            return session
        except Exception:
            logger.debug("curl_cffi session unavailable; falling back to yfinance default session.", exc_info=True)
            return None

    def _fetch_alpha_vantage(self, ticker: str) -> pd.DataFrame:
        logger.info("Fetching %s from Alpha Vantage fallback", ticker)
        query = urlencode(
            {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": ticker,
                "apikey": settings.alpha_vantage_api_key,
                "outputsize": "compact",
            }
        )
        with urlopen(f"https://www.alphavantage.co/query?{query}", timeout=30.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        series = payload.get("Time Series (Daily)")
        if not series:
            note = payload.get("Note") or payload.get("Information") or payload.get("Error Message") or "empty Alpha Vantage response"
            raise ValueError(note)

        rows = []
        for date_value, values in series.items():
            rows.append(
                {
                    "Date": date_value,
                    "Open": values.get("1. open"),
                    "High": values.get("2. high"),
                    "Low": values.get("3. low"),
                    "Close": values.get("4. close"),
                    "Volume": values.get("6. volume"),
                }
            )
        return pd.DataFrame(rows)

    def _fetch_sample_cache(self) -> pd.DataFrame:
        logger.info("Loading cached sample stock data from %s", self.sample_data_path)
        if not self.sample_data_path.exists():
            raise FileNotFoundError(f"Cached sample CSV not found at {self.sample_data_path}")
        return pd.read_csv(self.sample_data_path)

    def _prepare_and_validate(self, data: pd.DataFrame, ticker: str, source_name: str) -> pd.DataFrame:
        if data is None or data.empty:
            raise ValueError(f"{source_name} returned empty data for {ticker}.")

        frame = data.copy()
        logger.info("%s raw dataframe shape for %s: %s", source_name, ticker, frame.shape)
        logger.info("%s raw dataframe head for %s:\n%s", source_name, ticker, frame.head().to_string())
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                next((part for part in column if str(part).lower() in {"date", "datetime", "open", "high", "low", "close", "adj close", "volume"}), column[0])
                if isinstance(column, tuple)
                else column
                for column in frame.columns
            ]

        frame = frame.reset_index()
        frame = self._standardize_columns(frame)
        frame = frame.dropna(subset=["Date", "Open", "High", "Low", "Close"])

        validated = self.validator.validate(frame)
        logger.info("%s validated dataframe shape for %s: %s", source_name, ticker, validated.shape)
        logger.info("%s validated dataframe head for %s:\n%s", source_name, ticker, validated.head().to_string())
        return validated

    @staticmethod
    def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
        normalized_map = {str(column).strip().lower().replace("_", " "): column for column in frame.columns}
        aliases = {
            "Date": ["date", "datetime", "index"],
            "Open": ["open", "1. open"],
            "High": ["high", "2. high"],
            "Low": ["low", "3. low"],
            "Close": ["close", "adj close", "4. close", "5. adjusted close"],
            "Volume": ["volume", "6. volume"],
        }

        renamed = frame.copy()
        for target, candidates in aliases.items():
            if target in renamed.columns:
                continue
            source = next((normalized_map[candidate] for candidate in candidates if candidate in normalized_map), None)
            if source is not None:
                renamed = renamed.rename(columns={source: target})

        return renamed
