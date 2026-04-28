from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    def transform(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        frame = df.copy()
        frame["return_1d"] = frame["Close"].pct_change()
        frame["ma_5"] = frame["Close"].rolling(window=5).mean()
        frame["ma_10"] = frame["Close"].rolling(window=10).mean()
        frame["ma_20"] = frame["Close"].rolling(window=20).mean()
        frame["volatility_10"] = frame["return_1d"].rolling(window=10).std()
        frame["volume_ma_10"] = frame["Volume"].rolling(window=10).mean()
        frame["rsi_14"] = self._rsi(frame["Close"], window=14)
        macd, signal = self._macd(frame["Close"])
        frame["macd"] = macd
        frame["macd_signal"] = signal
        for lag in [1, 2, 3, 5, 10]:
            frame[f"close_lag_{lag}"] = frame["Close"].shift(lag)
            frame[f"volume_lag_{lag}"] = frame["Volume"].shift(lag)

        frame["target"] = frame["Close"].shift(-horizon)
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.ffill().dropna().reset_index(drop=True)
        return frame

    @staticmethod
    def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=window).mean()
        loss = (-delta.clip(upper=0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
        ema_fast = series.ewm(span=12, adjust=False).mean()
        ema_slow = series.ewm(span=26, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
