"""Pola/strategi. Tiap fungsi mengembalikan sinyal {-1, 0, 1} berdasarkan data
sampai CLOSE hari t. Backtest mengeksekusi posisi di hari t+1 (tanpa look-ahead)."""
import numpy as np
import pandas as pd


def _state(enter_long, exit_long, enter_short=None, exit_short=None):
    """Ubah event masuk/keluar menjadi posisi yang dipertahankan (state machine via ffill)."""
    s = pd.Series(np.nan, index=enter_long.index)
    s[exit_long] = 0
    s[enter_long] = 1
    if enter_short is not None:
        s[exit_short] = 0
        s[enter_short] = -1
    return s.ffill().fillna(0)


def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# ---------- trend following ----------
def sma_cross(df, fast, slow, short=False):
    if fast >= slow:
        return None
    c = df["Close"]
    up = c.rolling(fast).mean() > c.rolling(slow).mean()
    sig = up.astype(float)
    if short:
        sig[~up] = -1.0
    return sig.where(c.rolling(slow).count() == slow, 0)


def donchian(df, n, short=False):
    c, h, l = df["Close"], df["High"], df["Low"]
    hi, lo = h.rolling(n).max().shift(1), l.rolling(n).min().shift(1)
    ex_hi, ex_lo = h.rolling(n // 2).max().shift(1), l.rolling(n // 2).min().shift(1)
    if short:
        return _state(c > hi, c < ex_lo, c < lo, c > ex_hi)
    return _state(c > hi, c < ex_lo)


def momentum(df, lookback, short=False):
    r = df["Close"].pct_change(lookback)
    sig = (r > 0).astype(float)
    if short:
        sig[r < 0] = -1.0
    return sig.where(r.notna(), 0)


# ---------- mean reversion ----------
def rsi_revert(df, n, lo, short=False):
    r = rsi(df["Close"], n)
    hi = 100 - lo
    if short:
        return _state(r < lo, r > 50, r > hi, r < 50)
    return _state(r < lo, r > 50)


def bollinger_revert(df, n, k, short=False):
    c = df["Close"]
    m, sd = c.rolling(n).mean(), c.rolling(n).std()
    if short:
        return _state(c < m - k * sd, c > m, c > m + k * sd, c < m)
    return _state(c < m - k * sd, c > m)


# ---------- candlestick ----------
def engulfing_hold(df, hold, short=False):
    """Bullish engulfing -> beli dan tahan `hold` hari."""
    o, c = df["Open"], df["Close"]
    bull = (c.shift(1) < o.shift(1)) & (c > o) & (c >= o.shift(1)) & (o <= c.shift(1))
    sig = bull.astype(float).replace(0, np.nan)
    if short:
        bear = (c.shift(1) > o.shift(1)) & (c < o) & (c <= o.shift(1)) & (o >= c.shift(1))
        sig[bear] = -1.0
    if hold > 1:
        sig = sig.ffill(limit=hold - 1)
    return sig.fillna(0)


def buy_hold(df, short=False):
    return pd.Series(1.0, index=df.index)


STRATEGIES = {
    "SMA cross":     (sma_cross, {"fast": [5, 10, 20, 50], "slow": [50, 100, 150, 200]}),
    "Donchian":      (donchian, {"n": [20, 40, 55, 100]}),
    "Momentum":      (momentum, {"lookback": [20, 60, 120, 250]}),
    "RSI revert":    (rsi_revert, {"n": [2, 3, 5, 14], "lo": [5, 10, 20, 30]}),
    "Bollinger rev": (bollinger_revert, {"n": [10, 20, 40], "k": [1.5, 2.0, 2.5]}),
    "Engulfing":     (engulfing_hold, {"hold": [1, 3, 5, 10]}),
}
