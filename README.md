# Stock Price Prediction System

A full-stack web application that predicts future stock prices based on historical data using machine learning.

---

# Live Demo
  https://stock-price-prediction-api-soqn.onrender.com

---

# Project Overview

This project allows users to enter a stock ticker (like AAPL, MSFT) and get a short-term price prediction based on historical data.

The system fetches stock data, processes it, trains a machine learning model, and displays predictions along with uncertainty.

---

# Features

-  Real-time stock data fetching (yfinance)
-  Machine learning based prediction (Linear Regression)
-  Train / Refresh model option
-  Future price forecasting (multi-day)
-  Uncertainty estimation
-  Live deployed API (Render)
-  Interactive web interface (Streamlit)
-  Robust error handling (data fallback, validation)

---

# Tech Stack

- **Backend:** FastAPI  
- **Frontend:** Streamlit  
- **ML Model:** Linear Regression  
- **Data:** yfinance API  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Deployment:** Render  
- **Version Control:** Git & GitHub  

---

# How It Works

1. User enters stock ticker (e.g., AAPL)
2. Backend fetches historical stock data
3. Data is cleaned and processed
4. Machine learning model is trained
5. Future stock prices are predicted
6. Results are displayed on UI

---

# API Endpoints

- `GET /health` → Check API status  
- `POST /train` → Train model  
- `GET /predict/{ticker}` → Get prediction  

---

# Run Locally

```bash
git clone https://github.com/your-username/Stock-Price-Prediction
cd Stock-Price-Prediction

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

uvicorn app.main:app --reload
