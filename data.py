"""Download & cache data harian dari Yahoo Finance."""
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).parent / "data_cache"

# kelas aset -> (daftar ticker, biaya per sisi transaksi, hari per tahun, boleh short?)
UNIVERSE = {
    "saham": {
        "tickers": ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "SPY", "AAPL"],
        "cost": 0.0020,   # ~0.15% beli / 0.25% jual di IDX, dirata-rata
        "periods": 252,
        "short": False,
    },
    "crypto": {
        "tickers": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
        "cost": 0.0010,   # fee taker exchange ~0.1%
        "periods": 365,
        "short": False,
    },
    "forex": {
        "tickers": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X"],
        "cost": 0.0001,   # spread ~1 pip
        "periods": 260,
        "short": True,
    },
}


def load(ticker: str, start: str = "2010-01-01", refresh: bool = False) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"{ticker.replace('=', '_')}.csv"
    fresh = f.exists() and (time.time() - f.stat().st_mtime) < 12 * 3600
    if fresh and not refresh:
        return pd.read_csv(f, index_col=0, parse_dates=True)
    df = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Close"] > 0]
    df.to_csv(f)
    return df
