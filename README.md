# Cointegration-Based Pairs Trading Bot (NXPI & AMAT)

This project implements a **statistical arbitrage trading bot** using **cointegration-based pairs trading**. It identifies, validates, and backtests mean-reverting equity pairs using historical data. The selected pair in this repository is **NXPI–AMAT**, chosen through a screening process based on cointegration metrics.

---

## Repository Contents

- **`screener.py`** – Screens 2000+ U.S. equity pairs across sectors using correlation and Augmented Dickey-Fuller (ADF) tests to identify cointegrated candidates.
- **`chosen_pair_spread_plot.py`** – Plots the **spread**, **rolling mean**, **rolling standard deviation**, and **z-score** for the selected NXPI–AMAT pair.
- **`NXPI_AMAT_plot.py`** – Additional visual analysis and exploratory plots for the NXPI–AMAT pair.
- **`backtest.py`** – Runs a realistic backtest on the pair from 2021–2025, calculating **CAGR**, **Sharpe ratio**, and **max drawdown** using $10,000 starting capital and z-score thresholds.
- **`README.md`** – You’re reading it! Explains project structure and usage.

---

## Strategy Overview

- **Target**: Find equity pairs with strong cointegration (ADF p-value < 0.001, correlation > 0.95).
- **Pair**: `NXPI` and `AMAT` selected based on statistical tests (see `screener.py`).
- **Execution Logic**: 
  - Track spread and z-score
  - Enter **long-short positions** when z-score > 1.0 or < –1.0
  - Exit when z-score reverts to 0
  - Simulated capital: **$10,000**
  - Simple position sizing, daily PnL tracking

---

## Backtest Results

- **CAGR**: ~24%  
- **Sharpe Ratio**: ~1.8  
- **Max Drawdown**: ~15%  

---

## Dependencies

- Python 3.13
- `pandas`, `numpy`, `yfinance`, `statsmodels`, `matplotlib`, `tqdm`

