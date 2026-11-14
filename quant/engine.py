"""Mesin riset: sinyal -> skor cross-sectional -> bobot walk-forward -> portofolio -> backtest.

Aturan anti look-ahead:
- Sinyal dihitung dari harga penutupan hari t.
- Target portofolio dari hari t dieksekusi di penutupan hari t+1.
- Bobot sinyal pada tanggal refit d hanya memakai IC yang return masa depannya sudah
  diketahui sebelum d.
"""
import numpy as np
import pandas as pd

from .config import REFIT, SHRINK, SIGNALS


# ---------------------------------------------------------------- sinyal
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def raw_signals(close, ret):
    vol = ret.rolling(60, min_periods=40).std()
    s = {
        "mom_12_1": (close.shift(21) / close.shift(252) - 1) / (vol * np.sqrt(231)),
        "mom_3m": (close / close.shift(63) - 1) / (vol * np.sqrt(63)),
        "trend_200": (close / close.rolling(200, min_periods=180).mean() - 1) / (vol * np.sqrt(200)),
        "high_52w": close / close.rolling(252, min_periods=200).max() - 1,
        "rev_1w": -(close / close.shift(5) - 1) / (vol * np.sqrt(5)),
        "rsi_rev": 50 - rsi(close, 5),
        "low_vol": -vol,
    }
    return {k: s[k] for k in SIGNALS}, vol


def xs_rank(sig, mask, min_n):
    """Rank cross-sectional -> skor seragam di [-1, 1]. NaN jika aset < min_n."""
    s = sig.where(mask)
    n = s.count(axis=1)
    z = (s.rank(axis=1) - 0.5).div(n, axis=0) * 2 - 1
    return z.where(n >= min_n, np.nan)


# ---------------------------------------------------------------- IC & bobot
def forward_return(ret, h):
    """Return dari close t+1 sampai close t+1+h (selaras dengan eksekusi t+1)."""
    p = (1 + ret.fillna(0)).cumprod()
    return p.shift(-(h + 1)) / p.shift(-1) - 1


def rank_ic(z, fwd, min_n):
    m = z.notna() & fwd.notna()
    a, b = z.where(m).rank(axis=1), fwd.where(m).rank(axis=1)
    a, b = a.sub(a.mean(axis=1), axis=0), b.sub(b.mean(axis=1), axis=0)
    ic = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    return ic.where(m.sum(axis=1) >= min_n)


def walk_forward_weights(ics, start_row, lookback, h, mode="ic"):
    """Bobot sinyal per tanggal refit. mode='ic' (IC-IR + shrink) atau 'equal'."""
    k = len(ics.columns)
    prior = pd.Series(1.0 / k, index=ics.columns)
    rows = {}
    for i in range(start_row, len(ics), REFIT):
        if mode == "equal":
            rows[ics.index[i]] = prior
            continue
        win = ics.iloc[max(0, i - lookback): max(0, i - h - 1)]
        mu, sd = win.mean(), win.std()
        icir = (mu / sd).fillna(0)
        w = icir / icir.abs().sum() if icir.abs().sum() > 0 else prior
        rows[ics.index[i]] = SHRINK * prior + (1 - SHRINK) * w
    return pd.DataFrame(rows).T


# ---------------------------------------------------------------- portofolio
def _pick(s, n, prev, buffer):
    """Ambil n teratas, tapi pertahankan aset lama selama peringkatnya masih <= buffer*n."""
    rank = s.rank(ascending=False, method="first")
    keep = [a for a in prev if a in rank.index and rank[a] <= buffer * n]
    keep = sorted(keep, key=lambda a: rank[a])[:n]
    rest = [a for a in rank.sort_values().index if a not in keep][: n - len(keep)]
    return pd.Index(keep + rest)


def build_targets(score, vol, ret, tradable, cfg, start_row, regime=None, top_n=None, hold=None,
                  buffer=None):
    top_n = top_n or cfg["top_n"]
    hold = hold or cfg["hold"]
    buffer = buffer or cfg.get("buffer", 1.0)
    R = ret.fillna(0)
    targets = {}
    prev_l, prev_s = [], []
    for i in range(start_row, len(score), hold):
        d = score.index[i]
        s = score.iloc[i].where(tradable.iloc[i]).dropna()
        v = vol.iloc[i].reindex(s.index)
        s = s[(v > 0) & v.notna()]
        if len(s) < cfg["min_assets"]:
            continue
        v = v[s.index]
        if cfg["long_short"]:
            longs = _pick(s, top_n, prev_l, buffer)
            shorts = _pick(-s.drop(longs), top_n, prev_s, buffer)
            prev_l, prev_s = list(longs), list(shorts)
            wl = (1 / v[longs]) / (1 / v[longs]).sum() * 0.5
            ws = -(1 / v[shorts]) / (1 / v[shorts]).sum() * 0.5
            w = pd.concat([wl, ws])
        else:
            pick = _pick(s, top_n, prev_l, buffer)
            prev_l = list(pick)
            w = (1 / v[pick]) / (1 / v[pick]).sum()
        cov = R.iloc[max(0, i - 59): i + 1][w.index].cov() * cfg["periods"]
        pv = float(np.sqrt(max(w.values @ cov.values @ w.values, 0)))
        scale = min(cfg["target_vol"] / pv if pv > 0 else 1.0, cfg["max_gross"] / w.abs().sum())
        w = w * scale
        if regime is not None and not bool(regime.iloc[i]):
            w = w * 0.0
        targets[d] = w
    return targets


def backtest(ret, targets, cost):
    """Simulasi harian: bobot ter-drift, target dieksekusi di close hari berikutnya."""
    R = ret.fillna(0)
    cols = list(R.columns)
    pos = {c: i for i, c in enumerate(cols)}
    Rv = R.values
    w = np.zeros(len(cols))
    pending = None
    out, to_list, gross = np.zeros(len(R)), np.zeros(len(R)), np.zeros(len(R))
    for i, d in enumerate(R.index):
        r = Rv[i]
        pr = float(w @ r)
        w = w * (1 + r) / (1 + pr) if 1 + pr > 0 else np.zeros_like(w)
        tc = 0.0
        if pending is not None:
            tgt = np.zeros(len(cols))
            for c, x in pending.items():
                tgt[pos[c]] = x
            to = float(np.abs(tgt - w).sum())
            tc = to * cost
            to_list[i] = to
            w = tgt
            pending = None
        out[i] = (1 + pr) * (1 - tc) - 1
        gross[i] = np.abs(w).sum()
        if d in targets:
            pending = targets[d]
    return pd.DataFrame({"ret": out, "turnover": to_list, "gross": gross}, index=R.index)


# ---------------------------------------------------------------- satu kelas, ujung ke ujung
def run_class(data, cfg, mode="ic", top_n=None, hold=None, cost_mult=1.0, use_regime=True, buffer=None):
    close, ret, tradable = data["close"], data["ret"], data["tradable"]
    hold = hold or cfg["hold"]
    raw, vol = raw_signals(close, ret)
    z = {k: xs_rank(v, tradable, cfg["min_assets"]) for k, v in raw.items()}
    fwd = forward_return(ret, hold)
    ics = pd.DataFrame({k: rank_ic(v, fwd, cfg["min_assets"]) for k, v in z.items()})
    # baris pertama dengan cukup aset bersinyal + masa pemanasan
    count = z["mom_12_1"].notna().sum(axis=1)
    first = int(np.argmax((count >= cfg["min_assets"]).values))
    start_row = max(cfg["warmup"], first + 252)
    weights = walk_forward_weights(ics, start_row, cfg["ic_lookback"], hold, mode)
    w_daily = weights.reindex(close.index).ffill()
    zs = sum(z[k].fillna(0).mul(w_daily[k], axis=0) for k in z)
    have = sum(z[k].notna().astype(int) for k in z)
    score = zs.where(have >= 5)
    regime = None
    if use_regime and cfg["regime"] and cfg["regime"] in close.columns:
        b = close[cfg["regime"]]
        regime = b > b.rolling(200, min_periods=180).mean()
    targets = build_targets(score, vol, ret, tradable, cfg, start_row, regime, top_n, hold, buffer)
    bt = backtest(ret, targets, cfg["cost"] * cost_mult)
    bt = bt.iloc[start_row:]
    return dict(bt=bt, targets=targets, weights=weights, ics=ics, z=z, score=score, vol=vol,
                regime=regime, start=close.index[start_row])
