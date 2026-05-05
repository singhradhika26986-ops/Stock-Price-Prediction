from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    def transform(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        frame = df.copy()
        frame["return_1d"] = frame["Close"].pct_change()
        frame["ma_3"] = frame["Close"].rolling(window=3, min_periods=1).mean()
        frame["ma_5"] = frame["Close"].rolling(window=5, min_periods=1).mean()
        frame["volatility_5"] = frame["return_1d"].rolling(window=5, min_periods=2).std()
        frame["volume_ma_5"] = frame["Volume"].rolling(window=5, min_periods=1).mean()
        frame["rsi_7"] = self._rsi(frame["Close"], window=7)
        macd, signal = self._macd(frame["Close"])
        frame["macd"] = macd
        frame["macd_signal"] = signal
        for lag in [1, 2, 3, 5]:
            frame[f"close_lag_{lag}"] = frame["Close"].shift(lag)

        frame["target"] = frame["Close"].shift(-horizon)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        logger.info("Feature engineering NaN count before fill/drop: %s", frame.isna().sum().to_dict())
        frame = frame.ffill().bfill()
        essential_columns = ["Date", "Close", "target"]
        frame = frame.dropna(subset=essential_columns).reset_index(drop=True)
        logger.info("Final feature-engineered data shape: %s", frame.shape)
        return frame

    @staticmethod
    def _rsi(series: pd.Series, window: int = 7) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
        ema_fast = series.ewm(span=12, adjust=False).mean()
        ema_slow = series.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
