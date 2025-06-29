import yfinance as yf
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt

tickers = ["NXPI", "AMAT"]
prices = (
    yf.download(tickers, start="2021-01-01", auto_adjust=True)["Close"]
    .ffill()
    .dropna()
)

prices["NXPI_INV"] = -prices["NXPI"]

Y = prices["NXPI_INV"]
X = add_constant(prices["AMAT"])
alpha, beta = OLS(Y, X).fit().params

spread = Y - (alpha + beta * prices["AMAT"])

window = 252
rolling_mean = spread.rolling(window).mean()
rolling_std = spread.rolling(window).std()
z_score = (spread - rolling_mean) / rolling_std

plt.figure(figsize=(12, 6))
plt.plot(spread.index, spread, label="spread", color="steelblue")
plt.plot(rolling_mean.index, rolling_mean, label="rolling mean", color="orange")
plt.plot(rolling_std.index, rolling_std, label="rolling std", color="green")
plt.plot(z_score.index, z_score, label="z-score", color="red")
plt.title("NXPI / AMAT (Even Strategy) — Spread and Rolling Metrics")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()