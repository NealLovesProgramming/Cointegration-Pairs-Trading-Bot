import pandas as pd
import numpy as np
import yfinance as yf
from itertools import combinations
from datetime import datetime
from tqdm import tqdm

# Light-weight statsmodels imports (safe on Py 3.13)
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

import pandas as pd
import yfinance as yf
import sys

# List of available sectors
sector_names = [
    'technology', 'healthcare', 'financial-services', 'consumer-cyclical', 
    'industrials', 'communication-services', 'consumer-defensive', 
    'energy', 'utilities', 'real-estate', 'basic-materials'
]

def parse_args():
    print(sys.argv)
    return sys.argv[1:]

class Sectors:
    def __init__(self, name):

        self.names = []
        self.symbols = {}

        if type(name) == list:

            if len(name):
                for sector in name:
                    if sector in sector_names:
                        self.names.append(sector)
                        print("accessing", sector + "...")
                    else:
                        print(sector, "is an invalid type")
            else:
                for n in sector_names:
                    self.names.append(n)

        elif type(name) == str and name in sector_names:
            self.names = name
            print(self.names)
        else:
            print(name, "is an invalid type")

    def show(self, filter=None):

        names = self.names

        if filter != None:
            names = []
            if len(filter):
                for f in filter:
                    print(f)
                    if f in self.names:
                        names.append(f)
            elif filter in sector_names:
                names.append(filter)

        for sector_name in names:
            print(f"\n{'='*50}")
            print(f"SECTOR: {sector_name.upper().replace('-', ' ')}")
            print(f"{'='*50}")

            try:
                sector = yf.Sector(sector_name)
                industries = sector.industries
                # 
                print(f"Number of industries: {len(industries)}")
                print("\nIndustries:")
                # 
                # Display each industry
                for index, (key, row) in enumerate(industries.iterrows(), 1):
                    print(f"{index:2d}. {row['name']:<40} | Symbol: {row['symbol']:<15} | Weight: {row['market weight']:.4f}")
                    # 
            except Exception as e:
                print(f"Error accessing {sector_name}: {e}")

            print("-" * 50)
    
    def get_tickers(self, industries=None):
        if industries == None:
            for n in self.names:
                for i, (k, r) in enumerate(yf.Sector(n).industries.iterrows(), 1):
                    
                    #print(f"\nIndustry: {k}")

                    try:
                        industry = yf.Industry(k)
                        top_companies = industry.top_companies
    
                        # Check if DataFrame is not empty
                        if not top_companies.empty:
                            # The symbols are in the index, not a column!
                            symbols = top_companies.index.tolist()
                            self.symbols[k] = symbols
                            #print(f"Company symbols: {symbols}")
                            
                        else:
                            print("No companies found for this industry")

                    except Exception as e:
                        print(f"Error getting companies for industry {r['name']}: {e}")
        else:
            for i in industries:
                for s in self.names:
                    if i in s.industries:
                        pass
        return self.symbols


a = Sectors(parse_args())
tickers = a.get_tickers()

for i in tickers:
    for t in tickers[i]:
        print(t)


# ---------- XLK ticker list ----------
tickers = [
    "AAPL","MSFT","NVDA","AVGO","ADBE","CRM","AMD","CSCO","NFLX","INTU",
    "QCOM","ORCL","ACN","TXN","AMAT","IBM","ADP","LRCX","MU","NOW","PANW",
    "INTC","KLAC","SNPS","ANET","CDNS","MSI","FTNT","MCHP","PAYX","CTSH",
    "APH","ADI","NXPI","AKAM","HPE","STX","TEL","KEYS","GLW","HPQ","ZBRA",
    "TER","WDAY","TYL","EPAM","GPN","GRMN","DXC","FFIV","CDW","PTC","IT",
    "JKHY","ON","SWKS","QRVO","ENPH","SEDG","RNG","DOCU","OKTA","DDOG",
    "ZS","CRWD"
]

# ---------- Download adjusted-close prices (2021-01-01 → today) ----------
prices = (
    yf.download(
        tickers,
        start="2021-01-01",
        end=datetime.today().strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        threads=True
    )["Close"]
    .ffill()                 # forward-fill any gap
    .dropna(axis=1, how="all")
)

# ---------- Helper: regression + ADF + correlation ----------
def analyse_pair(y_sym, x_sym):
    y = prices[y_sym].values
    X = add_constant(prices[x_sym].values)
    beta0, beta1 = OLS(y, X).fit().params      # alpha, beta
    resid = y - (beta0 + beta1 * X[:, 1])

    adf_stat, p_val, *_ = adfuller(resid)
    corr_val = prices[y_sym].corr(prices[x_sym])  # Pearson

    return corr_val, adf_stat, p_val, beta1, beta0

# ---------- Iterate through all pairs ----------
records = []
pairs = combinations(prices.columns, 2)
total_pairs = len(prices.columns) * (len(prices.columns) - 1) // 2

for s1, s2 in tqdm(pairs, total=total_pairs):
    corr, adf_stat, p_val, beta, alpha = analyse_pair(s1, s2)
    records.append((s1, s2, corr, adf_stat, p_val, beta, alpha))

# ---------- Build results DataFrame ----------
cols = ["Stock1", "Stock2", "corr", "ADF_stat", "p_value", "beta", "alpha"]
results = (
    pd.DataFrame(records, columns=cols)
      .sort_values("p_value")              # sort by stationarity strength
      .reset_index(drop=True)
)

# ---------- Display ----------
print(f"\nTotal pairs analysed: {len(results):,}")
print("\nTop 10 by lowest p-value:")
print(results.head(10))
