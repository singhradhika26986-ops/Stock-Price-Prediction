from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from app.core.config import settings
from src.data.ingestion import StockDataIngestor
from src.features.engineering import FeatureEngineer
from src.modeling.experiment_tracking import ExperimentTracker
from src.modeling.metrics import regression_metrics
from src.modeling.sequence import build_lstm_sequences, transform_lstm_sequences

logger = logging.getLogger(__name__)


class TrainingPipeline:
    def __init__(self) -> None:
        settings.model_path.mkdir(parents=True, exist_ok=True)
        settings.data_path.mkdir(parents=True, exist_ok=True)
        self.ingestor = StockDataIngestor()
        self.engineer = FeatureEngineer()
        self.tracker = ExperimentTracker()

    def run(self, ticker: str, period: str, interval: str, horizon: int) -> dict:
        raw = self.ingestor.fetch(ticker=ticker, period=period, interval=interval)
        engineered = self.engineer.transform(raw, horizon=horizon)
        raw.to_csv(settings.data_path / f"{ticker}_raw.csv", index=False)
        engineered.to_csv(settings.data_path / f"{ticker}_features.csv", index=False)

        feature_columns = [
            column
            for column in engineered.columns
            if column not in {"Date", "target"}
        ]

        split_index = int(len(engineered) * 0.8)
        train_frame = engineered.iloc[:split_index].copy()
        test_frame = engineered.iloc[split_index:].copy()

        x_train = train_frame[feature_columns]
        y_train = train_frame["target"]
        x_test = test_frame[feature_columns]
        y_test = test_frame["target"]

        model_results = {}
        bundle_candidates = {}
        backtest_frames = {}

        for model_name in ["linear_regression", "random_forest", "xgboost"]:
            metrics, bundle, backtest_frame = self._train_tabular_model(
                model_name,
                x_train,
                y_train,
                x_test,
                y_test,
                feature_columns,
                test_frame["Date"],
                horizon,
            )
            model_results[model_name] = metrics
            bundle_candidates[model_name] = bundle
            backtest_frames[model_name] = backtest_frame
            self.tracker.log_run(
                ticker=ticker,
                model_name=model_name,
                params={"period": period, "interval": interval, "horizon": horizon},
                metrics=metrics,
                artifacts={},
            )

        lstm_metrics, lstm_bundle, lstm_backtest = self._train_lstm(train_frame, test_frame, feature_columns, horizon)
        model_results["lstm"] = lstm_metrics
        bundle_candidates["lstm"] = lstm_bundle
        backtest_frames["lstm"] = lstm_backtest
        self.tracker.log_run(
            ticker=ticker,
            model_name="lstm",
            params={"period": period, "interval": interval, "horizon": horizon},
            metrics=lstm_metrics,
            artifacts={},
        )

        best_model = min(model_results, key=lambda name: model_results[name]["rmse"])
        best_bundle = bundle_candidates[best_model]
        best_bundle["model_name"] = best_model
        best_bundle["horizon"] = horizon
        best_bundle["feature_columns"] = feature_columns
        best_bundle["trained_at"] = datetime.now(timezone.utc).isoformat()

        if best_model == "lstm":
            keras_path = settings.model_path / f"{ticker}_lstm.keras"
            best_bundle["model"].save(keras_path)
            best_bundle["model_path"] = str(keras_path)
            del best_bundle["model"]

        bundle_path = settings.model_path / f"{ticker}_best_bundle.joblib"
        joblib.dump(best_bundle, bundle_path)

        metrics_path = settings.model_path / f"{ticker}_metrics.json"
        metrics_path.write_text(json.dumps(model_results, indent=2), encoding="utf-8")

        backtest_path = settings.model_path / f"{ticker}_backtest.csv"
        backtest_frames[best_model].to_csv(backtest_path, index=False)

        drift_summary = self._build_drift_summary(raw, backtest_frames[best_model])
        drift_path = settings.model_path / f"{ticker}_drift.json"
        drift_path.write_text(json.dumps(drift_summary, indent=2), encoding="utf-8")
        self._append_training_history(ticker, best_model, model_results[best_model], drift_summary)

        if "feature_importance" in best_bundle:
            importance_frame = pd.DataFrame(best_bundle["feature_importance"])
            importance_frame.to_csv(settings.model_path / f"{ticker}_feature_importance.csv", index=False)

        self.tracker.log_run(
            ticker=ticker,
            model_name=best_model,
            params={"period": period, "interval": interval, "horizon": horizon},
            metrics=model_results[best_model],
            artifacts={
                "bundle_path": str(bundle_path),
                "metrics_path": str(metrics_path),
                "backtest_path": str(backtest_path),
                "drift_path": str(drift_path),
            },
        )

        return {
            "ticker": ticker,
            "best_model": best_model,
            "metrics": model_results,
            "artifacts": {
                "bundle_path": str(bundle_path),
                "metrics_path": str(metrics_path),
                "backtest_path": str(backtest_path),
                "drift_path": str(drift_path),
            },
            "retrained_at": best_bundle["trained_at"],
        }

    def _train_tabular_model(self, model_name, x_train, y_train, x_test, y_test, feature_columns, test_dates, horizon):
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        if model_name == "linear_regression":
            model = LinearRegression()
            model.fit(x_train_scaled, y_train)
            preds = model.predict(x_test_scaled)
            metrics = regression_metrics(y_test, preds)
        elif model_name == "random_forest":
            metrics, model, preds = self._grid_search(
                estimator_cls=RandomForestRegressor,
                grid={"n_estimators": [100, 200], "max_depth": [4, 8, None]},
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                random_state=42,
            )
        else:
            metrics, model, preds = self._grid_search(
                estimator_cls=XGBRegressor,
                grid={"n_estimators": [150, 250], "max_depth": [3, 5], "learning_rate": [0.03, 0.08]},
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                objective="reg:squarederror",
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )

        if model_name == "linear_regression":
            residual_std = float(np.std(y_test - preds))
            bundle = {
                "model": model,
                "feature_scaler": scaler,
                "residual_std": residual_std,
                "residual_quantiles": self._residual_quantiles(y_test - preds),
                "use_scaler": True,
            }
        else:
            residual_std = float(np.std(y_test - preds))
            feature_importance = [
                {"feature": feature, "importance": float(importance)}
                for feature, importance in sorted(
                    zip(feature_columns, model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )
            ]
            bundle = {
                "model": model,
                "feature_scaler": scaler,
                "residual_std": residual_std,
                "residual_quantiles": self._residual_quantiles(y_test - preds),
                "feature_importance": feature_importance,
                "use_scaler": False,
            }

        backtest_metrics, backtest_frame = self._walk_forward_backtest(model_name, x_train, y_train, x_test, y_test, test_dates)
        metrics["backtest"] = backtest_metrics
        return metrics, bundle, backtest_frame

    def _grid_search(self, estimator_cls, grid, x_train, y_train, x_test, y_test, **kwargs):
        best_score = None
        best_model = None
        best_preds = None
        best_metrics = None
        for params in ParameterGrid(grid):
            model = estimator_cls(**params, **kwargs)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            metrics = regression_metrics(y_test, preds)
            if best_score is None or metrics["rmse"] < best_score:
                best_score = metrics["rmse"]
                best_model = model
                best_preds = preds
                best_metrics = metrics
        return best_metrics, best_model, best_preds

    def _train_lstm(self, train_frame, test_frame, feature_columns, horizon):
        from tensorflow.keras import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam

        sequence_length = settings.sequence_length
        x_seq_train, y_seq_train, feature_scaler, target_scaler = build_lstm_sequences(
            train_frame[feature_columns],
            train_frame["target"],
            sequence_length,
        )
        combined = pd.concat([train_frame.tail(sequence_length), test_frame], ignore_index=True)
        x_seq_test, y_seq_test = transform_lstm_sequences(
            combined[feature_columns],
            combined["target"],
            sequence_length,
            feature_scaler,
            target_scaler,
        )

        model = Sequential(
            [
                LSTM(64, return_sequences=True, input_shape=(x_seq_train.shape[1], x_seq_train.shape[2])),
                Dropout(0.2),
                LSTM(32),
                Dense(16, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        model.fit(x_seq_train, y_seq_train, epochs=15, batch_size=16, verbose=0)

        preds_scaled = model.predict(x_seq_test, verbose=0)
        preds = target_scaler.inverse_transform(preds_scaled).reshape(-1)
        y_true = target_scaler.inverse_transform(y_seq_test.reshape(-1, 1)).reshape(-1)
        metrics = regression_metrics(y_true, preds)
        metrics["backtest"] = metrics.copy()
        backtest_frame = pd.DataFrame(
            {
                "Date": test_frame["Date"].tail(len(preds)).astype(str).tolist(),
                "actual": y_true,
                "predicted": preds,
                "absolute_error": np.abs(y_true - preds),
            }
        )

        bundle = {
            "model": model,
            "feature_scaler": feature_scaler,
            "target_scaler": target_scaler,
            "sequence_length": sequence_length,
            "residual_std": float(np.std(y_true - preds)),
            "residual_quantiles": self._residual_quantiles(y_true - preds),
            "use_scaler": False,
        }
        return metrics, bundle, backtest_frame

    def _walk_forward_backtest(self, model_name, x_train, y_train, x_test, y_test, test_dates):
        history_x = x_train.copy()
        history_y = y_train.copy()
        predictions = []

        for index in range(len(x_test)):
            current_x = x_test.iloc[[index]]
            current_y = y_test.iloc[index]

            if model_name == "linear_regression":
                scaler = StandardScaler()
                hist_scaled = scaler.fit_transform(history_x)
                cur_scaled = scaler.transform(current_x)
                model = LinearRegression()
                model.fit(hist_scaled, history_y)
                pred = model.predict(cur_scaled)[0]
            elif model_name == "random_forest":
                model = RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42)
                model.fit(history_x, history_y)
                pred = model.predict(current_x)[0]
            else:
                model = XGBRegressor(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    objective="reg:squarederror",
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42,
                )
                model.fit(history_x, history_y)
                pred = model.predict(current_x)[0]

            predictions.append(pred)
            history_x = pd.concat([history_x, current_x], ignore_index=True)
            history_y = pd.concat([history_y, pd.Series([current_y])], ignore_index=True)

        metrics = regression_metrics(y_test, predictions)
        backtest_frame = pd.DataFrame(
            {
                "Date": test_dates.astype(str).tolist(),
                "actual": y_test.to_numpy(),
                "predicted": np.asarray(predictions),
            }
        )
        backtest_frame["absolute_error"] = (backtest_frame["actual"] - backtest_frame["predicted"]).abs()
        return metrics, backtest_frame

    @staticmethod
    def _residual_quantiles(residuals) -> dict[str, float]:
        values = np.asarray(residuals, dtype=float)
        return {
            "q05": float(np.quantile(values, 0.05)),
            "q25": float(np.quantile(values, 0.25)),
            "q50": float(np.quantile(values, 0.50)),
            "q75": float(np.quantile(values, 0.75)),
            "q95": float(np.quantile(values, 0.95)),
        }

    @staticmethod
    def _build_drift_summary(raw_frame: pd.DataFrame, backtest_frame: pd.DataFrame) -> dict:
        recent_window = min(20, len(raw_frame))
        rolling_volatility = float(raw_frame["Close"].pct_change().tail(recent_window).std())
        mean_absolute_error = float(backtest_frame["absolute_error"].mean()) if not backtest_frame.empty else 0.0
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "recent_price_volatility": rolling_volatility,
            "backtest_mean_absolute_error": mean_absolute_error,
            "data_points_seen": int(len(raw_frame)),
        }

    @staticmethod
    def _append_training_history(ticker: str, best_model: str, metrics: dict, drift_summary: dict) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticker": ticker,
            "best_model": best_model,
            "rmse": metrics.get("rmse"),
            "mae": metrics.get("mae"),
            "mape": metrics.get("mape"),
            "directional_accuracy": metrics.get("directional_accuracy"),
            "recent_price_volatility": drift_summary.get("recent_price_volatility"),
            "backtest_mean_absolute_error": drift_summary.get("backtest_mean_absolute_error"),
        }
        path = settings.monitoring_path / "training_history.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
