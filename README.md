# Macro-Aware Stock Forecasting Using Bi-GRU Ensemble and FinBERT Sentiment Integration

> **Version:** V1.3 &nbsp;|&nbsp; **Model:** Bi-GRU Ensemble &nbsp;|&nbsp; **Avg MAPE:** 1.0658% &nbsp;|&nbsp; **Avg R²:** 0.9582 &nbsp;|&nbsp; **Coverage:** 58 NSE Stocks

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Tech Stack](#2-tech-stack)
3. [Dataset](#3-dataset)
4. [Model Architecture](#4-model-architecture)
5. [Feature Engineering](#5-feature-engineering)
6. [Sentiment Layer](#6-sentiment-layer-newsapi--finbert)
7. [Results & Performance](#7-results--performance)
8. [Web Application](#8-web-application)
9. [File Structure](#9-file-structure)
10. [Setup & Execution](#10-setup--execution)
11. [Risk Assessment Module](#11-risk-assessment-module)
12. [Known Limitations](#12-known-limitations)
13. [Future Work](#13-future-work)

---

## 1. Project Overview

**Macro-Aware Stock Forecasting Using Bi-GRU Ensemble and FinBERT Sentiment Integration** is a deep learning–based stock price prediction system built for the National Stock Exchange of India (NSE). It uses a **Bidirectional GRU ensemble** trained on 23 years of historical price data (2003–2026) combined with a real-time **NLP sentiment layer** (NewsAPI + FinBERT) to predict next-day closing prices for 58 NSE stocks and ETFs.

### The Core Problem

Traditional technical analysis relies on lagging indicators and manual interpretation. MarketMind addresses this by:

- Automating feature extraction from raw OHLCV data
- Incorporating real-time market sentiment via NLP
- Providing probabilistic next-day price predictions with built-in risk metrics

### Key Highlights

| Metric | Value |
|---|---|
| Average MAPE | 1.0658% |
| Average R² | 0.9582 |
| Stocks below 1.25% MAPE | 46 / 58 |
| Stocks below 1.00% MAPE | 27 / 58 |
| Training data range | 2015 – 2026 |
| Prediction horizon | 1 trading day ahead |
| Ensemble size | 3 Bi-GRU models per stock |

---

## 2. Tech Stack

### Core Framework

| Component | Library | Version | Role |
|---|---|---|---|
| Deep Learning | TensorFlow / Keras | 2.x | Model training and inference |
| Data Processing | Pandas, NumPy | latest | Feature engineering, data pipelines |
| Market Data | yfinance | latest | Historical OHLCV data fetching |
| Scaling | scikit-learn (RobustScaler) | latest | Feature normalization |
| Web App | Streamlit | latest | Interactive dashboard |
| NLP Model | FinBERT (ProsusAI) | latest | Financial sentiment scoring |
| News API | NewsAPI | v2 | Real-time financial news |
| Tunneling | pyngrok | latest | Colab → public URL |
| Scheduling | APScheduler | latest | Automated daily updates |

### Training Hardware

| Resource | Specification |
|---|---|
| Platform | Google Colab |
| GPU | NVIDIA T4 / A100 (Colab) |
| Storage | Google Drive (MyDrive) |
| Training time | ~4–6 hours (all 58 stocks) |

---

## 3. Dataset

### Price Data

- **Source:** Yahoo Finance via `yfinance`
- **Coverage:** 58 NSE stocks + ETFs
- **Range:** January 2003 – March 2026
- **Frequency:** Daily (trading days only)
- **Split:** 85% train / 15% test

### Sentiment Data

- **Source:** NewsAPI (live) + pre-built historical sentiment CSV
- **File:** `unified_sentiment_layer_CLAMPED.csv`
- **Rows:** 403,920 rows × 11 columns
- **Tickers:** 48 stocks with stock-specific sentiment
- **Macro coverage:** All 58 tickers via Nifty/NSE macro batch

### Stock Coverage — 49 Individual Stocks

```
ADANIENT, ADANIPORTS, APOLLOHOSP, ASIANPAINT, AXISBANK,
BAJAJFINSV, BAJFINANCE, BHARTIARTL, BPCL, BRITANNIA,
CIPLA, COALINDIA, DIVISLAB, DRREDDY, EICHERMOT,
GRASIM, HAVELLS, HCLTECH, HDFCBANK, HDFCLIFE,
HEROMOTOCO, HINDALCO, HINDUNILVR, ICICIBANK, INDUSINDBK,
INFY, ITC, JSWSTEEL, KOTAKBANK, LT, LTIM, M&M, MARUTI,
NESTLEIND, NTPC, ONGC, POWERGRID, RELIANCE, SBILIFE,
SBIN, SUNPHARMA, TATACONSUM, TATASTEEL, TCS, TECHM,
TITAN, ULTRACEMCO, UPL, WIPRO
```

### 9 ETFs (Macro Sentiment Only)

```
NIFTYBEES, BANKBEES, GOLDBEES, JUNIORBEES, ITBEES,
CPSEETF, AXISNIFTY, HDFCNIFTY, MON100
```

---

## 4. Model Architecture

### Bi-GRU Ensemble

Each stock has **3 independently trained Bi-GRU models** whose predictions are averaged (ensemble) for final output.

```
Input: (15 timesteps × 10 features)
         │
         ▼
┌─────────────────────────────────┐
│   Bidirectional GRU (64 units)  │
│   kernel_regularizer = L2(0.001)│
│   return_sequences = True       │
└──────────────┬──────────────────┘
               │
               ▼
         Dropout (0.25)
               │
               ▼
┌─────────────────────────────────┐
│       GRU (32 units)            │
│   kernel_regularizer = L2(0.001)│
└──────────────┬──────────────────┘
               │
               ▼
         Dense (1)  →  Target_Log_Returns
```

### Hyperparameters

| Parameter | Value |
|---|---|
| Lookback window | 15 days |
| Ensemble members | 3 |
| Bi-GRU units | 64 |
| GRU units | 32 |
| Dropout | 0.25 |
| L2 regularization | 0.001 |
| Optimizer | Nadam |
| Batch size | 32 |
| Max epochs | 25 |
| Early stopping patience | 8 |
| Train/Test split | 85% / 15% |

### Custom Loss Function

A **profit-focused loss** that penalizes directional errors 5× more than magnitude errors:

```python
def profit_focused_loss(y_true, y_pred):
    direction_error = cast(sign(y_true) != sign(y_pred), float32)
    magnitude_error = abs(y_true - y_pred)
    size_weight = where(abs(y_true) > 0.01, 2.0, 1.0)
    return mean((direction_error * 5.0 + 1.0) * magnitude_error * size_weight)
```

### Transductive KNN Adjustment

After ensemble prediction, a **KNN-based pattern matching** step (k=20, cosine similarity) blends base predictions with historical neighbor patterns:

```
enhanced_pred = 0.7 × model_pred + 0.3 × mean(knn_neighbor_targets)
```

---

## 5. Feature Engineering

### 10 Model Input Features

| Feature | Description | Type |
|---|---|---|
| `Close` | Closing price | Price |
| `Log_Returns` | log(Close / Close_prev) | Returns |
| `RSI` | Relative Strength Index (14-day) | Momentum |
| `SMA_50` | 50-day Simple Moving Average | Trend |
| `Volatility_20D` | 20-day rolling std of log returns | Risk |
| `USD_INR` | USD/INR exchange rate | Macro |
| `Macro_Sentiment_7D` | 7-day rolling macro news sentiment | Sentiment |
| `Stock_Sentiment_7D` | 7-day rolling stock-specific sentiment | Sentiment |
| `Sentiment_Momentum` | Diff of Stock_Sentiment_7D | Sentiment |
| `News_Shock` | 1 if 5+ articles on a date, else 0 | Sentiment |

### Additional CSV Features (25 total in stock files)

```
Close, High, Low, Open, Volume
RSI, MACD, MACD_Signal, BB_High, BB_Low, ATR
SMA_50, SMA_200, Log_Returns
Volatility_5D, Volatility_20D, Volatility_Ratio
Momentum_5, Momentum_10, Price_Acceleration, Price_Range
SMA_20, Above_SMA_50, SMA_Distance, USD_INR
```

### Scaler

**RobustScaler** (median + IQR) — robust to price outliers and market crashes, fitted per stock on training data only.

---

## 6. Sentiment Layer (NewsAPI + FinBERT)

### Pipeline

```
NewsAPI fetch (6 requests/day)
        │
        ▼
Keyword matching → assign articles to tickers
        │
        ▼
FinBERT scoring (ProsusAI/finbert)
title + description → {positive, negative, neutral} → [-1, +1]
        │
        ▼
Daily aggregation per ticker
        │
        ▼
7-day rolling average → Stock_Sentiment_7D
                      → Macro_Sentiment_7D
                      → Sentiment_Momentum
                      → News_Shock (≥5 articles/day)
        │
        ▼
Appended to unified_sentiment_layer_CLAMPED.csv
```

### Quota Efficiency

```
49 stocks ÷ 10 per batch = 5 requests
1 macro batch             = 1 request
─────────────────────────────────────
Total: 6 requests / 100 daily free quota
```

---

## 7. Results & Performance

### Overall Metrics

| Metric | Value |
|---|---|
| Average MAPE | **1.0658%** |
| Average R² | **0.9582** |
| Average RMSE | Stock-dependent (₹16–₹94) |
| Stocks < 1.00% MAPE | 27 / 58 (46.6%) |
| Stocks < 1.25% MAPE | 46 / 58 (79.3%) |

### Best Performing Stocks (MAPE < 0.70%)

```
NIFTYBEES, AXISNIFTY, HDFCNIFTY, JUNIORBEES, HDFCBANK,
ITBEES, COALINDIA, NESTLEIND, APOLLOHOSP, BRITANNIA
```

### Model Color Legend (Dashboard)

| Color | MAPE Range |
|---|---|
| 🟢 Green | < 1.25% — Excellent |
| 🔵 Blue | 1.25% – 1.50% — Good |
| 🟠 Orange | > 1.50% — Acceptable |
| 🔴 Red | Outlier |

---

## 8. Web Application

### Pages

| Page | Description |
|---|---|
| Login | SHA256-hashed auth, 5-attempt lockout |
| Dashboard | Model performance across all 58 stocks — MAPE bar chart, distribution, scatter |
| Predict | Live next-day prediction with BUY/SELL/HOLD signal + Risk Assessment |
| Admin | Stock update, News fetch, Scheduler, Cache management, System info |

### Admin Panel Tabs

| Tab | Function |
|---|---|
| 🔄 Stock Data Update | Fetch OHLCV + recompute all 25 features via yfinance |
| 📰 News & Sentiment | NewsAPI fetch + FinBERT scoring + sentiment CSV update |
| ⏰ Auto Schedule | APScheduler CronTrigger — Mon-Fri, configurable IST time |
| 🗄️ Cache Management | Clear Streamlit prediction + results cache |
| ℹ️ System Info | File stats, user accounts, sentiment CSV stats |

### Predict Page Output

```
Signal:          BUY / SELL / HOLD
Last known:      ₹XXXX.XX  on DD Mon YYYY
Predicting for:  Tue, DD Mon YYYY
Predicted price: ₹XXXX.XX  (+X.XXX%)

Metrics:  MAPE | RMSE | MAE | R²

Risk Assessment:
  Entry Price | Stop Loss | Take Profit | Predicted Move | R/R Ratio | RMSE
```

---

## 9. File Structure

```
StockPrediction_2024_FINAL/
│
├── streamlit_app/
│   ├── app.py                        # Main Streamlit application (~2300 lines)
│   ├── requirements.txt
│   ├── users.json                    # Hashed user credentials
│   └── README.md
│
├── data_final/
│   └── stocks/
│       ├── RELIANCE.NS.csv           # 58 stock CSVs (25 features each)
│       ├── TCS.NS.csv
│       └── ...
│
├── data_final/
│   └── unified_sentiment_layer_CLAMPED.csv   # 403,920 rows sentiment data
│
├── saved_models_v1_f/
│   ├── RELIANCE.NS_ensemble_0.keras  # 3 models × 58 stocks = 174 .keras files
│   ├── RELIANCE.NS_ensemble_1.keras
│   ├── RELIANCE.NS_ensemble_2.keras
│   ├── RELIANCE.NS_scaler.pkl        # 58 RobustScaler objects
│   └── ...
│
├── v1_f_metrics.csv                  # Training results for all 58 stocks
├── validation_plots_v1_f/            # Actual vs Predicted plots
├── loss_plots_v1_f/                  # Training loss curves
└── training_history_v1_f/            # Per-epoch history CSVs
```

---

## 10. Setup & Execution

### Prerequisites

```python
# Cell 1 — Install dependencies
!pip install streamlit pyngrok apscheduler pytz yfinance \
            transformers torch newsapi-python scikit-learn \
            tensorflow joblib plotly -q
```

### Running the App (Google Colab)

```python
# Cell 2 — Copy app from Drive to Colab runtime
import shutil, os
dest = '/content/streamlit_app'
if os.path.exists(dest): shutil.rmtree(dest)
shutil.copytree(
    '/content/drive/MyDrive/StockPrediction_2024_FINAL/streamlit_app',
    dest
)

# Cell 3 — Start Streamlit + ngrok
import subprocess, threading, time
from pyngrok import ngrok

def run():
    subprocess.run([
        "streamlit", "run", "/content/streamlit_app/app.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])

threading.Thread(target=run, daemon=True).start()
time.sleep(5)

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
tunnel = ngrok.connect(8501)
print("✅ MarketMind is live at:", tunnel.public_url)
```

### Default Login Credentials

| Username | Password | Role |
|---|---|---|
| admin | admin123 | Administrator |
| analyst | predict2024 | Analyst |

### Daily Update Workflow

```
1. NSE closes at 3:30 PM IST
2. Admin → Stock Data Update → 7d → Run  (fetches today's OHLCV)
3. Admin → News & Sentiment → 1 day → Quick Fetch → Run
4. Predict → Select stock → Run Prediction  (now uses today's data)
```

---

## 11. Risk Assessment Module

The Predict page includes an RMSE-based risk assessment box:

### Formula

```
Stop Loss   = Entry Price − (1.5 × RMSE)
Take Profit = Entry Price + (1.0 × RMSE)
R/R Ratio   = Predicted Move / (1.5 × RMSE)
```

### Signal Labels

| Label | Condition | Meaning |
|---|---|---|
| ✅ Good Trade | R/R ≥ 1.0 | Predicted gain exceeds model error |
| 🟡 Marginal | R/R < 1.0 | Predicted gain smaller than risk |

### Rationale

RMSE is in rupee terms — it directly represents the model's typical prediction error on any given day. Using 1.5× RMSE as the stop loss buffer ensures the trade isn't stopped out by normal prediction variance alone.

---

## 12. Known Limitations

| Limitation | Description |
|---|---|
| 1-day horizon only | Model predicts only the next trading day — not suitable for swing/positional trading |
| No intraday data | Uses daily closing prices — intraday fluctuations not captured |
| News coverage | NewsAPI free tier — 100 requests/day, 1 month historical, English only |
| ETF sentiment | 9 ETFs use only macro sentiment (no stock-specific news) |
| Colab dependency | App requires active Colab session — no persistent 24/7 deployment |
| ngrok URL | Changes on every Colab restart — not a permanent URL |
| Model staleness | Models trained up to Feb 2026 — no retraining on new data |
| Market gaps | Weekends, holidays result in missing dates in sentiment CSV |

---

## 13. Future Work

| Enhancement | Impact | Notes |
|---|---|---|
| Multi-day forecasting | High | Predict 3/5/10 days ahead |
| Transformer architecture | High | Replace Bi-GRU with Temporal Fusion Transformer |
| Options & derivatives data | Medium | Add open interest, IV as features |
| Continuous retraining | High | Auto-retrain on new 30 days of data monthly |
| Permanent cloud deployment | Medium | Migrate from Colab+ngrok to Hugging Face Spaces |
| Mobile-responsive UI | Low | CSS overhaul for phone/tablet |
| Portfolio optimization | High | Multi-stock signal aggregation with Markowitz weights |
| Backtesting engine | High | Simulate trades using historical signals + R/R filter |

---

## Quick Reference

### Run Prediction

```
1. Open app → Login (admin / admin123)
2. Predict → Select stock from dropdown
3. Click "Run Prediction"
4. View: Signal | Forecast card | Risk Assessment | Chart
```

### Key Metrics Summary

```
Model          : Bi-GRU Ensemble (3 members per stock)
Features       : 10 (6 technical + 1 macro + 3 sentiment)
Lookback       : 15 trading days
Training data  : 2003 – Feb 2026 (~5,700 trading days)
Avg MAPE       : 1.0658%
Avg R²         : 0.9582
Stocks covered : 58 (49 stocks + 9 ETFs)
App name       : MarketMind (demo interface)
```

---

*Macro-Aware Stock Forecasting Using Bi-GRU Ensemble and FinBERT Sentiment Integration — V1.3 — March 2026*  
*Developed as part of thesis project — NSE Stock Prediction*
