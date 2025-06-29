# backtest_realistic.py
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

CAPITAL_START   = 10_000    
LEG_FRACTION    = 0.10      
ENTRY_Z         = 1.0
EXIT_Z          = 0.5
WINDOW          = 252       
REFRESH_BETA    = 21        
MAX_HOLD        = 10        
COST_BP         = 10       

px = (
    yf.download(["NXPI", "AMAT"], start="2021-01-01",
                auto_adjust=True, progress=False)["Close"]
      .ffill()
      .dropna()
)
px["NXPI_INV"] = -px["NXPI"] 

idx   = px.index
cols  = ["alpha", "beta", "spread", "roll_mean", "roll_std", "z",
         "signal", "position", "days_held", "pnl$", "equity$"]
out   = pd.DataFrame(index=idx, columns=cols, dtype=float)

equity = CAPITAL_START
position = 0        
days_held = 0

for i, today in enumerate(idx):

    if i >= WINDOW and (i - WINDOW) % REFRESH_BETA == 0:
        win = px.iloc[i-WINDOW : i]
        Y = win["NXPI_INV"]
        X = add_constant(win["AMAT"])
        alpha, beta = OLS(Y, X).fit().params
    elif i < WINDOW:
        out.loc[today, "equity$"] = equity
        continue

    spread = px.loc[today, "NXPI_INV"] - (alpha + beta*px.loc[today, "AMAT"])
    hist_spread = px["NXPI_INV"].iloc[i-WINDOW:i] - (
        alpha + beta*px["AMAT"].iloc[i-WINDOW:i]
    )
    roll_mean = hist_spread.mean()
    roll_std  = hist_spread.std()
    z         = (spread - roll_mean) / roll_std if roll_std else 0.0

    out.loc[today, ["alpha","beta","spread","roll_mean","roll_std","z"]] = \
        [alpha, beta, spread, roll_mean, roll_std, z]

    if i > WINDOW:
        z_yday = out["z"].iloc[i-1]
        if position == 0 and abs(z_yday) > ENTRY_Z:
            signal = np.sign(-z_yday)  
        elif position != 0 and (abs(z_yday) < EXIT_Z or days_held >= MAX_HOLD):
            signal = 0              
        else:
            signal = position         
    else:
        signal = 0

    trade_change = signal - position
    if trade_change != 0:
        notional_leg = equity * LEG_FRACTION
        cost = abs(trade_change) * notional_leg * 2 * (COST_BP / 10_000)
        equity -= cost

    position = signal
    days_held = days_held + 1 if position != 0 else 0

    if i > 0:
        ret_nxpi = px["NXPI"].pct_change().iloc[i]
        ret_amat = px["AMAT"].pct_change().iloc[i]
        leg_notional = equity * LEG_FRACTION

        pnl = (
            (-position) * leg_notional * ret_nxpi +
            position * beta * leg_notional * ret_amat
        )
        equity += pnl
    else:
        pnl = 0.0

    out.loc[today, ["signal","position","days_held","pnl$","equity$"]] = \
        [signal, position, days_held, pnl, equity]

trade_count = (out["signal"].diff().abs() > 0).sum()
returns = out["equity$"].pct_change().dropna()
cagr = (out["equity$"].iloc[-1]/CAPITAL_START)**(252/len(returns)) - 1
sharpe = returns.mean()/returns.std() * np.sqrt(252)
max_dd = 1 - out["equity$"].div(out["equity$"].cummax()).min()

print(f"Trades executed : {int(trade_count)}")
print(f"CAGR            : {cagr:6.2%}")
print(f"Sharpe ratio    : {sharpe:5.2f}")
print(f"Max drawdown    : {max_dd:5.2%}")

plt.figure(figsize=(10,4))
out["equity$"].plot()
plt.title("NXPI / AMAT – Even Pair Strategy (Realistic)  |  Equity Curve")
plt.xlabel("Date"); plt.ylabel("Equity ($)")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
