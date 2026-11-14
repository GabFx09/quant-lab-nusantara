"""Jalankan seluruh riset dan kumpulkan hasil untuk website."""
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import CLASSES, REFIT, SHRINK, SIGNALS
from .data import load_class
from .engine import run_class
from .metrics import clean, monthly_table, stats, yearly

warnings.filterwarnings("ignore")


def r5(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 5)


def weekly(series):
    s = series.resample("W-SUN").last().dropna()
    return {"d": [d.strftime("%Y-%m-%d") for d in s.index], "v": [r5(v) for v in s.values]}


def short(cfg, t):
    return t.replace(cfg["strip"], "")


def signal_table(res, cfg):
    ics, h = res["ics"], cfg["hold"]
    live = ics.loc[res["start"]:]
    sub = live.iloc[::h]             # sampel tidak tumpang tindih untuk t-stat
    last_w = res["weights"].iloc[-1]
    rows = []
    for k, (label, desc) in SIGNALS.items():
        x = live[k].dropna()
        xs = sub[k].dropna()
        rows.append(dict(key=k, label=label, desc=desc, ic=r5(x.mean()),
                         icir=r5(x.mean() / x.std() * np.sqrt(cfg["periods"] / h)) if x.std() > 0 else None,
                         t=r5(xs.mean() / xs.std() * np.sqrt(len(xs))) if xs.std() > 0 else None,
                         pos=r5((xs > 0).mean()), w=r5(last_w[k])))
    return rows


def holdings(res, data, cfg):
    targets = res["targets"]
    last_d = max(targets)
    tgt = targets[last_d]
    score = res["score"]
    i = len(score) - 1
    while i > 0 and score.iloc[i].dropna().size < cfg["min_assets"]:
        i -= 1
    sd = score.index[i]
    s = score.iloc[i].where(data["tradable"].iloc[i]).dropna().sort_values(ascending=False)
    close = data["close"]
    ret21 = close.iloc[i] / close.iloc[max(0, i - 21)] - 1
    volann = res["vol"].iloc[i] * np.sqrt(cfg["periods"])
    rank = []
    for n, (t, v) in enumerate(s.items(), 1):
        rank.append(dict(t=short(cfg, t), score=r5(v), rank=n, vol=r5(volann.get(t)), r1m=r5(ret21.get(t)),
                         price=r5(close[t].iloc[i]), w=r5(tgt.get(t, 0.0)),
                         z={k: r5(res["z"][k][t].iloc[i]) for k in SIGNALS}))
    reg = None
    if res["regime"] is not None:
        b = data["close"][cfg["regime"]]
        sma = b.rolling(200, min_periods=180).mean()
        reg = dict(on=bool(res["regime"].iloc[-1]), price=r5(b.iloc[-1]), sma=r5(sma.iloc[-1]))
    return dict(rebalance=last_d.strftime("%Y-%m-%d"), score_date=sd.strftime("%Y-%m-%d"),
                gross=r5(float(np.abs(tgt).sum())), regime=reg,
                weights=[dict(t=short(cfg, t), w=r5(w)) for t, w in tgt.sort_values(ascending=False).items() if abs(w) > 1e-9],
                ranking=rank)


def robustness(data, cfg, base):
    rows = []
    def add(group, label, **kw):
        bt = run_class(data, cfg, **kw)["bt"]
        s = stats(bt.ret, cfg["periods"], bt.turnover, cfg["cost"] * kw.get("cost_mult", 1))
        rows.append(dict(group=group, label=label, Sharpe=r5(s["Sharpe"]), CAGR=r5(s["CAGR"]), MaxDD=r5(s["MaxDD"]),
                         Turnover=r5(s["TurnoverTahunan"]), base=False))
    rows.append(dict(group="Utama", label="Konfigurasi utama", Sharpe=r5(base["Sharpe"]), CAGR=r5(base["CAGR"]),
                     MaxDD=r5(base["MaxDD"]), Turnover=r5(base["TurnoverTahunan"]), base=True))
    add("Bobot sinyal", "Bobot sama (tanpa IC)", mode="equal")
    add("Biaya", "Biaya 2×", cost_mult=2.0)
    add("Biaya", "Biaya 3×", cost_mult=3.0)
    n = cfg["top_n"]
    for m in (max(2, n // 2), n * 2):
        add("Jumlah aset", f"{'Long/short ' if cfg['long_short'] else 'Top '}{m}", top_n=m)
    for b in (1.0, 3.0):
        add("Buffer", f"Buffer {b:g}×", buffer=b)
    for h in (max(1, cfg["hold"] // 2), cfg["hold"] * 2):
        add("Rebalance", f"Tiap {h} hari", hold=h)
    if cfg["regime"]:
        add("Filter", "Tanpa filter tren BTC", use_regime=False)
    return rows


def grid(data, cfg):
    out = []
    for h in sorted({max(1, cfg["hold"] // 2), cfg["hold"], cfg["hold"] * 2, cfg["hold"] * 4}):
        for b in (1.0, 2.0, 3.0):
            bt = run_class(data, cfg, hold=h, buffer=b)["bt"]
            s = stats(bt.ret, cfg["periods"])
            out.append(dict(hold=h, buffer=b, Sharpe=r5(s["Sharpe"])))
    return out


def run_all(full=True):
    out = {"generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
           "params": {"refit": REFIT, "shrink": SHRINK}, "signals": {k: v[0] for k, v in SIGNALS.items()},
           "classes": {}}
    sleeves = {}
    for k, cfg in CLASSES.items():
        print("kelas", k)
        data = load_class(k)
        res = run_class(data, cfg)
        bt = res["bt"]
        idx = bt.index
        st = stats(bt.ret, cfg["periods"], bt.turnover, cfg["cost"])
        ew = data["ret"].where(data["tradable"]).mean(axis=1).reindex(idx).fillna(0)
        series = {"Strategi": bt.ret, "Rata-rata semua aset": ew}
        if "bench" in data:
            series[cfg["benchmark_label"]] = data["bench"].reindex(idx).fillna(0)
        sleeves[k] = bt.ret
        eq = {name: weekly((1 + s).cumprod()) for name, s in series.items()}
        dd = (1 + bt.ret).cumprod()
        yr = pd.DataFrame({n: yearly(s) for n, s in series.items()})
        out["classes"][k] = dict(
            label=cfg["label"], sub=cfg["sub"], currency=cfg["currency"],
            n_assets=int(data["close"].shape[1]), last_date=data["close"].index[-1].strftime("%Y-%m-%d"),
            cfg=dict(top_n=cfg["top_n"], hold=cfg["hold"], buffer=cfg["buffer"], target_vol=cfg["target_vol"],
                     cost=cfg["cost"], long_short=cfg["long_short"], max_gross=cfg["max_gross"],
                     regime=cfg["regime"]),
            stats=clean(st),
            compare={n: clean(stats(s, cfg["periods"])) for n, s in series.items()},
            equity=eq, drawdown=weekly(dd / dd.cummax() - 1),
            gross=weekly(bt.gross),
            yearly={"years": [int(y) for y in yr.index], **{n: [r5(v) for v in yr[n]] for n in yr.columns}},
            signals=signal_table(res, cfg),
            weights={"d": [d.strftime("%Y-%m-%d") for d in res["weights"].index],
                     **{s: [r5(v) for v in res["weights"][s]] for s in SIGNALS}},
            now=holdings(res, data, cfg),
            robust=robustness(data, cfg, st) if full else [],
            grid=grid(data, cfg) if full else [],
        )

    # ---------- gabungan risk parity ----------
    cal = pd.date_range(min(s.index[0] for s in sleeves.values()), max(s.index[-1] for s in sleeves.values()), freq="D")
    S = pd.DataFrame({k: s.reindex(cal) for k, s in sleeves.items()})
    live = S.notna().cumsum() > 0
    S = S.fillna(0)
    vol = S.rolling(365, min_periods=120).std().where(live)
    month_end = S.index[S.index.is_month_end]
    W = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for d in month_end:
        v = vol.loc[d].dropna()
        v = v[v > 0]
        if len(v):
            W.loc[d, v.index] = (1 / v) / (1 / v).sum()
            W.loc[d] = W.loc[d].fillna(0)
    W = W.shift(1).ffill().fillna(0)          # dipakai mulai hari berikutnya
    start = W.index[(W.sum(axis=1) > 0).values.argmax()]
    comb = (W * S).sum(axis=1).loc[start:]
    W = W.loc[start:]
    eqc = (1 + comb).cumprod()
    roll = comb.rolling(365, min_periods=300)
    rsh = (roll.mean() / roll.std() * np.sqrt(365)).dropna()
    yrc = pd.DataFrame({"Gabungan": yearly(comb), **{CLASSES[k]["label"]: yearly(S[k].loc[start:]) for k in S}})
    wm = W.resample("ME").last()
    out["combined"] = dict(
        stats=clean(stats(comb, 365)),
        equity=weekly(eqc), drawdown=weekly(eqc / eqc.cummax() - 1), rolling_sharpe=weekly(rsh),
        monthly=monthly_table(comb),
        yearly={"years": [int(y) for y in yrc.index], **{n: [r5(v) for v in yrc[n]] for n in yrc.columns}},
        weights={"d": [d.strftime("%Y-%m-%d") for d in wm.index], **{k: [r5(v) for v in wm[k]] for k in wm}},
        weights_now={k: r5(W[k].iloc[-1]) for k in W},
        sleeves_since={k: clean(stats(S[k].loc[start:], 365)) for k in S},
        corr={a: {b: r5(S.loc[start:, a].corr(S.loc[start:, b])) for b in S} for a in S},
        start=start.strftime("%Y-%m-%d"),
    )
    return out
