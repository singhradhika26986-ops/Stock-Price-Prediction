from __future__ import annotations

import pandas as pd


class DataValidationError(ValueError):
    pass


class StockDataValidator:
    required_columns = {"Date", "Open", "High", "Low", "Close", "Volume"}
    min_rows = 51

    def validate(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing_columns = self.required_columns.difference(frame.columns)
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {sorted(missing_columns)}")

        validated = frame.copy()
        validated["Date"] = pd.to_datetime(validated["Date"], errors="coerce")
        if validated["Date"].isna().any():
            raise DataValidationError("Date column contains invalid timestamps.")

        numeric_columns = ["Open", "High", "Low", "Close", "Volume"]
        for column in numeric_columns:
            validated[column] = pd.to_numeric(validated[column], errors="coerce")

        if validated[numeric_columns].isna().any().any():
            raise DataValidationError("Numeric market fields contain invalid values.")

        if (validated["Volume"] < 0).any():
            raise DataValidationError("Volume cannot be negative.")

        invalid_price_rows = validated[
            (validated["High"] < validated["Low"])
            | (validated["Open"] <= 0)
            | (validated["High"] <= 0)
            | (validated["Low"] <= 0)
            | (validated["Close"] <= 0)
        ]
        if not invalid_price_rows.empty:
            raise DataValidationError("Detected invalid OHLC price relationships.")

        validated = validated.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
        if len(validated) < self.min_rows:
            raise DataValidationError(f"Not enough stock data rows. Expected at least {self.min_rows}, got {len(validated)}.")
        return validated
