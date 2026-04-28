from __future__ import annotations

import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "change-me")

st.set_page_config(page_title="Stock Prediction Dashboard", layout="wide")
st.title("Stock Price Prediction Dashboard")
st.caption("Live forecasting, uncertainty, backtesting, and API monitoring in one place.")

ticker = st.sidebar.text_input("Ticker", value="AAPL").upper()
days_ahead = st.sidebar.slider("Forecast horizon", min_value=1, max_value=15, value=5)
should_train = st.sidebar.button("Train / Refresh Model")
auto_refresh = st.sidebar.checkbox("Auto refresh every 30s", value=False)

if auto_refresh:
    components.html(
        """
        <script>
        setTimeout(function() {
          window.parent.location.reload();
        }, 30000);
        </script>
        """,
        height=0,
    )


def call_api(method: str, path: str, payload: dict | None = None) -> dict:
    with httpx.Client(timeout=120.0) as client:
        response = client.request(
            method,
            f"{API_BASE_URL}{path}",
            json=payload,
            headers={"x-api-key": API_KEY},
        )
        response.raise_for_status()
        return response.json()


if should_train:
    with st.spinner("Training model pipeline..."):
        train_data = call_api("POST", "/train", {"ticker": ticker, "period": "5y", "interval": "1d", "horizon": days_ahead})
    st.success(f"Best model: {train_data['best_model']}")
    st.json(train_data["metrics"])

try:
    prediction_data = call_api("GET", f"/predict/{ticker}?days_ahead={days_ahead}")
    insight_data = call_api("GET", f"/insights/{ticker}")
    monitoring_summary = call_api("GET", "/monitoring/summary")
    monitoring_requests = call_api("GET", "/monitoring/requests")
    training_history = call_api("GET", f"/monitoring/training-history?ticker={ticker}")
except Exception as exc:
    st.warning(f"API call failed: {exc}")
    st.stop()

pred_df = pd.DataFrame(
    {
        "Step": list(range(1, prediction_data["horizon"] + 1)),
        "Prediction": prediction_data["predictions"],
        "Lower": prediction_data["confidence_lower"],
        "Upper": prediction_data["confidence_upper"],
        "P25": prediction_data["quantiles"]["p25"],
        "P75": prediction_data["quantiles"]["p75"],
    }
)

backtest_df = pd.DataFrame(insight_data["backtest_points"])
training_df = pd.DataFrame(training_history["records"])
request_df = pd.DataFrame(monitoring_requests["records"])

col1, col2 = st.columns([2, 1])

with col1:
    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=pred_df["Step"], y=pred_df["Prediction"], mode="lines+markers", name="Forecast"))
    forecast_fig.add_trace(go.Scatter(x=pred_df["Step"], y=pred_df["Upper"], mode="lines", name="Upper", line=dict(dash="dash")))
    forecast_fig.add_trace(go.Scatter(x=pred_df["Step"], y=pred_df["Lower"], mode="lines", name="Lower", line=dict(dash="dash"), fill="tonexty"))
    forecast_fig.add_trace(go.Scatter(x=pred_df["Step"], y=pred_df["P75"], mode="lines", name="P75", line=dict(color="#f59e0b")))
    forecast_fig.add_trace(go.Scatter(x=pred_df["Step"], y=pred_df["P25"], mode="lines", name="P25", line=dict(color="#f59e0b"), fill="tonexty"))
    for index, path in enumerate(prediction_data["probabilistic_paths"][:5], start=1):
        forecast_fig.add_trace(
            go.Scatter(
                x=pred_df["Step"],
                y=path,
                mode="lines",
                name=f"Sample {index}",
                opacity=0.18,
                line=dict(width=1),
                showlegend=False,
            )
        )
    forecast_fig.update_layout(title=f"{ticker} Forecast", xaxis_title="Days ahead", yaxis_title="Predicted close")
    st.plotly_chart(forecast_fig, use_container_width=True)

with col2:
    st.metric("Model", prediction_data["model_name"])
    st.metric("Last Close", f"{prediction_data['last_close']:.2f}")
    st.metric("First Forecast", f"{prediction_data['predictions'][0]:.2f}")
    st.metric("Uncertainty Score", f"{prediction_data['uncertainty_score']:.4f}")
    st.metric("API Requests", f"{monitoring_summary['total_requests']}")
    st.metric("Avg Latency (ms)", f"{monitoring_summary['average_latency_ms']:.2f}")

importance = pd.DataFrame(insight_data["feature_importance"])
if not importance.empty:
    st.subheader("Feature Importance")
    st.dataframe(importance.head(15), use_container_width=True)
else:
    st.info("Feature importance is only available for tree-based best models.")

st.subheader("Metrics")
st.json(insight_data["metrics"])

if not backtest_df.empty:
    st.subheader("Backtesting: Predicted vs Actual")
    backtest_fig = go.Figure()
    backtest_fig.add_trace(go.Scatter(x=backtest_df["Date"], y=backtest_df["actual"], mode="lines", name="Actual"))
    backtest_fig.add_trace(go.Scatter(x=backtest_df["Date"], y=backtest_df["predicted"], mode="lines", name="Predicted"))
    backtest_fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(backtest_fig, use_container_width=True)

    error_fig = go.Figure()
    error_fig.add_trace(go.Bar(x=backtest_df["Date"], y=backtest_df["absolute_error"], name="Absolute Error"))
    error_fig.update_layout(title="Backtest Error", xaxis_title="Date", yaxis_title="Absolute Error")
    st.plotly_chart(error_fig, use_container_width=True)

if not training_df.empty:
    st.subheader("Retraining Performance Trend")
    training_df["timestamp"] = pd.to_datetime(training_df["timestamp"])
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=training_df["timestamp"], y=training_df["rmse"], mode="lines+markers", name="RMSE"))
    trend_fig.add_trace(go.Scatter(x=training_df["timestamp"], y=training_df["mae"], mode="lines+markers", name="MAE"))
    trend_fig.add_trace(go.Scatter(x=training_df["timestamp"], y=training_df["directional_accuracy"], mode="lines+markers", name="Directional Accuracy"))
    trend_fig.update_layout(xaxis_title="Retrain timestamp", yaxis_title="Metric")
    st.plotly_chart(trend_fig, use_container_width=True)

st.subheader("Drift Summary")
st.json(insight_data["drift_summary"])

if not request_df.empty:
    st.subheader("API Latency Timeline")
    request_df["timestamp"] = pd.to_datetime(request_df["timestamp"])
    latency_fig = go.Figure()
    latency_fig.add_trace(go.Scatter(x=request_df["timestamp"], y=request_df["latency_ms"], mode="lines+markers", name="Latency"))
    latency_fig.update_layout(xaxis_title="Timestamp", yaxis_title="Latency (ms)")
    st.plotly_chart(latency_fig, use_container_width=True)
