import yfinance as yf
import matplotlib.pyplot as plt

# ---------------- pair + hedge params ----------------
s1, s2  = "AMAT", "NXPI"
beta    = 1.127745
alpha   = -76.817099

# ------------- download adjusted closes --------------
raw = yf.download([s1, s2],
                  start="2021-01-01",
                  auto_adjust=True,
                  progress=False)          # default group_by="column"

# Slice the 'Close' level -> columns are now the tickers
prices = raw['Close']                     # <--- key change

# ------------- build & plot spread -------------------
spread = prices[s1] - (alpha + beta * prices[s2])

plt.figure(figsize=(12, 5))
plt.plot(spread, label="Spread")
plt.axhline(spread.mean(), color="red", ls="--", label="Mean")
plt.title(f"Cointegrated Spread: {s1} − (α + β·{s2})")
plt.xlabel("Date"); plt.ylabel("Spread ($)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
