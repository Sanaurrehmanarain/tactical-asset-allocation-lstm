# Deep Learning for Finance — Project #2 (Tasks)

## Scenario
Most institutional investors allocate across multiple asset classes and/or geographies. This can help exploit variations in rewards for risk across market regimes. Strategic Asset Allocation (SAA) and Tactical Asset Allocation (TAA) are different approaches to multi-asset investment strategies.

---

## Step 1 — Data Gathering & Exploratory Data Analysis (EDA)
We will work with 5 ETFs (one per asset class) downloaded with `yfinance`:

- Equity: SPDR S&P 500 ETF — **SPY**
- Fixed Income: iShares 20+ Year Treasury Bond ETF — **TLT**
- Cash-like: iShares 1–3 Year Treasury Bond ETF — **SHY**
- Gold: SPDR Gold Shares — **GLD**
- Crude Oil: Invesco DB Oil Fund — **DBO**

### 1a) Download data
- Download **daily prices** for each ETF.
- Use **January 1, 2018 to December 30, 2022** as the **test sample**.
- Define **training** and **validation** periods at your discretion.

### 1b) Exploratory data analysis (brief, focus on key characteristics)
Use visualization tools to describe main dynamics/characteristics:
- summary statistics
- seasonality
- stationarity
- you may focus on prices and/or daily returns

EDA must cover:
- **individual distributions**
- **joint distributions**, including **correlations/covariances**

---

## Step 2 — Single-output RNN models (one model per asset)
Build RNN-based deep learning models to predict **25-day ahead return** of each ETF.

### 2a) Build 5 separate LSTM models
For each ETF:
- use **LSTM** architecture (can combine LSTM + Dense layers)
- predict **25-day ahead return**
- input features are past information (choose lags you find appropriate)
- you may compare architectures and choose best performer

### 2b) Train models (in-sample)
- Train each of the 5 models
- Comment on **in-sample predictive performance**
- Compare performance across asset classes

### 2c) Test models (out-of-sample)
- Test each model out-of-sample
- Comment on **out-of-sample predictive performance**
- Compare across asset classes

### 2d) Build trading strategy from out-of-sample predictions
- Use predictions from **at least 4 out of 5** asset classes
- Rebalance **every 25 days**
- Example strategy: long top 2–3 predicted assets and short bottom 2–3 predicted assets
- You may choose a different strategy

### 2e) Backtest strategy in test period
- Backtest during the test period
- Comment on results
- Compare vs **buy-and-hold** return of an **equally weighted** portfolio of the 5 ETFs

---

## Step 3 — Multi-output deep learning model
Implement a model that predicts all 5 ETFs jointly (multi-output regression).

### 3a) Create multi-output architecture
- Blend all information from Step 2 inputs
- Use all inputs together to predict **25-day ahead returns** for all 5 ETFs
- Model has **5 outputs** (one per ETF)

### 3b) Train & test multi-output model
- Comment on in-sample and out-of-sample performance
- Compare with Step 2 (single-output models)

### 3c) Trading strategy from multi-output predictions
- Same instructions as Step 2d apply

### 3d) Backtest multi-output strategy
- Backtest during the test period
- Compare vs:
  (i) equally weighted buy-and-hold portfolio
  (ii) Step 2 trading strategy

---

## Step 4 — Group discussion / interpretation
Discuss:
- Predictability implications of multi-output vs single-output architectures
- Backtesting performance implications
- Whether the same information is captured by both architectures

---

## Step 5 — Submission
Submit all files that reproduce the assignment:
- Python file(s) for all steps
- A clear, organized report answering all questions (without code), as if presenting to your boss