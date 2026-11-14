"""Unduh, cache, dan bersihkan data panel harian per kelas aset."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import CLASSES, START

CACHE = Path(__file__).resolve().parent.parent / "data_cache"
MAX_AGE_H = 12


def _download(tickers):
    for attempt in range(3):
        try:
            df = yf.download(tickers, start=START, auto_adjust=True, progress=False,
                             group_by="column", threads=True)
            if df is not None and len(df):
                close = df["Close"]
                if isinstance(close, pd.Series):
                    close = close.to_frame(tickers[0])
                return close
        except Exception as e:  # noqa: BLE001
            print("  gagal unduh:", e)
        time.sleep(3 * (attempt + 1))
    return pd.DataFrame()


def download_close(tickers):
    close = _download(tickers)
    missing = [t for t in tickers if t not in close.columns or close[t].dropna().empty]
    for t in missing:  # coba ulang satu per satu
        c = _download([t])
        if not c.empty and t in c.columns and c[t].dropna().size:
            close[t] = c[t]
    return close


def load_class(name, refresh=False):
    cfg = CLASSES[name]
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"quant_{name}.pkl"
    fresh = f.exists() and (time.time() - f.stat().st_mtime) < MAX_AGE_H * 3600
    if fresh and not refresh:
        raw = pd.read_pickle(f)
    else:
        print(f"  unduh {name} ({len(cfg['tickers'])} ticker)")
        raw = download_close(cfg["tickers"])
        bench = cfg["benchmark"]
        if bench and bench not in raw.columns:
            raw["__bench__"] = download_close([bench])[bench]
        elif bench:
            raw["__bench__"] = raw[bench]
        raw.to_pickle(f)
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    bench = raw.pop("__bench__") if "__bench__" in raw.columns else None
    return clean(raw, cfg, bench)


def clean(close, cfg, bench):
    close = close.where(close > 0)
    close = close.loc[:, close.notna().sum() >= 400]
    if cfg["periods"] == 365:
        idx = pd.date_range(close.index.min(), close.index.max(), freq="D")
    else:
        idx = close.index[close.index.dayofweek < 5]
        idx = idx[close.loc[idx].notna().mean(axis=1) >= 0.3]   # buang hari libur bursa
    close = close.reindex(idx)
    traded = close.notna()
    closef = close.ffill(limit=5)
    ret = closef.pct_change(fill_method=None)
    ret = ret.where(ret.abs() <= cfg["bad_ret"])               # buang tick rusak
    # aset yang sudah tidak ada data > 5 hari dianggap tidak bisa diperdagangkan
    tradable = closef.notna() & (traded | traded.shift(1, fill_value=False))
    out = {"close": closef, "ret": ret, "tradable": tradable, "raw_close": close}
    if bench is not None:
        b = bench.reindex(idx).ffill(limit=5)
        out["bench"] = b.pct_change(fill_method=None).where(lambda x: x.abs() <= cfg["bad_ret"]).fillna(0)
    return out


def names(cfg, cols):
    return [c.replace(cfg["strip"], "") for c in cols]


__all__ = ["load_class", "names", "np"]
