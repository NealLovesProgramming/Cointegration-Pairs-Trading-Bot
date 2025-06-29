import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
from datetime import datetime
from tqdm import tqdm

from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

tickers = [
    "AAPL","MSFT","NVDA","AVGO","ADBE","CRM","AMD","CSCO","NFLX","INTU",
    "QCOM","ORCL","ACN","TXN","AMAT","IBM","ADP","LRCX","MU","NOW","PANW",
    "INTC","KLAC","SNPS","ANET","CDNS","MSI","FTNT","MCHP","PAYX","CTSH",
    "APH","ADI","NXPI","AKAM","HPE","STX","TEL","KEYS","GLW","HPQ","ZBRA",
    "TER","WDAY","TYL","EPAM","GPN","GRMN","DXC","FFIV","CDW","PTC","IT",
    "JKHY","ON","SWKS","QRVO","ENPH","SEDG","RNG","DOCU","OKTA","DDOG",
    "ZS","CRWD"
]

prices = (
    yf.download(
        tickers,
        start="2021-01-01",
        end=datetime.today().strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True
    )["Close"]
    .ffill()                
    .dropna(axis=1, how="all")
)

def analyse_pair(y_sym, x_sym):
    y = prices[y_sym].values
    X = add_constant(prices[x_sym].values)
    beta0, beta1 = OLS(y, X).fit().params     
    resid = y - (beta0 + beta1 * X[:, 1])

    adf_stat, p_val, *_ = adfuller(resid)
    corr_val = prices[y_sym].corr(prices[x_sym])  

    return corr_val, adf_stat, p_val, beta1, beta0

records = []
pairs = combinations(prices.columns, 2)
total_pairs = len(prices.columns) * (len(prices.columns) - 1) // 2

for s1, s2 in tqdm(pairs, total=total_pairs):
    corr, adf_stat, p_val, beta, alpha = analyse_pair(s1, s2)
    records.append((s1, s2, corr, adf_stat, p_val, beta, alpha))

cols = ["Stock1", "Stock2", "corr", "ADF_stat", "p_value", "beta", "alpha"]
results = (
    pd.DataFrame(records, columns=cols)
      .sort_values("p_value")             
      .reset_index(drop=True)
)

print(f"\nTotal pairs analysed: {len(results):,}")
print("\nTop 10 by lowest p-value:")
print(results.head(10))
