"""Ekspor hasil backtest + contoh grafik per pola ke JSON, lalu sisipkan ke web/template.html."""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import UNIVERSE, load
from strategies import STRATEGIES, rsi

ROOT = Path(__file__).parent
BARS = 160


def cfg_of(cls):
    return UNIVERSE[cls]


def r(x, d=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), d)


def series(s):
    return [r(v, 6) for v in s]


def example(name, row):
    fn, _ = STRATEGIES[name]
    p = ast.literal_eval(row["params"])
    cfg = cfg_of(row["kelas"])
    df = load(row["aset"])
    sig = fn(df, short=cfg["short"], **p)
    pos = sig.shift(1).fillna(0)
    c, h, l = df["Close"], df["High"], df["Low"]
    over, sub = {}, None
    if name == "SMA cross":
        over = {f"SMA {p['fast']}": c.rolling(p["fast"]).mean(), f"SMA {p['slow']}": c.rolling(p["slow"]).mean()}
    elif name == "Donchian":
        over = {f"High {p['n']}h": h.rolling(p["n"]).max().shift(1), f"Low {p['n'] // 2}h": l.rolling(p["n"] // 2).min().shift(1)}
    elif name == "Bollinger rev":
        m, sd = c.rolling(p["n"]).mean(), c.rolling(p["n"]).std()
        over = {f"MA {p['n']}": m, f"+{p['k']}σ": m + p["k"] * sd, f"−{p['k']}σ": m - p["k"] * sd}
    elif name == "RSI revert":
        sub = {"name": f"RSI({p['n']})", "values": rsi(c, p["n"]), "lines": [p["lo"], 50], "range": [0, 100]}
    elif name == "Momentum":
        sub = {"name": f"Return {p['lookback']} hari (%)", "values": c.pct_change(p["lookback"]) * 100, "lines": [0], "range": None}
    sl = slice(-BARS, None)
    out = {
        "aset": row["aset"], "kelas": row["kelas"], "params": p,
        "dates": [d.strftime("%Y-%m-%d") for d in df.index[sl]],
        "open": series(df["Open"].iloc[sl]), "close": series(c.iloc[sl]),
        "high": series(h.iloc[sl]), "low": series(l.iloc[sl]),
        "pos": [int(v) for v in pos.iloc[sl]],
        "overlays": {k: series(v.iloc[sl]) for k, v in over.items()},
    }
    if sub:
        out["sub"] = {**sub, "values": series(sub["values"].iloc[sl])}
    return out


def main():
    res = pd.read_csv(ROOT / "hasil_backtest.csv")
    cols = ["Sharpe", "CAGR", "MaxDD", "Winrate", "ProfitFactor", "Expectancy", "Trades", "Exposure"]
    bh = res[res.tujuan == "benchmark"].set_index("aset")

    def pack(row):
        d = {"kelas": row["kelas"], "aset": row["aset"], "params": row["params"]}
        for pre in ("IS", "OOS"):
            d[pre] = {k: r(row[f"{pre}_{k}"]) for k in cols}
        d["BH"] = {k: r(bh.loc[row["aset"], f"OOS_{k}"]) for k in ("Sharpe", "CAGR", "MaxDD")}
        return d

    strategies = {}
    for name in STRATEGIES:
        s = res[(res.strategi == name) & (res.tujuan == "max Sharpe")]
        ok = s[s.OOS_Trades >= 10]
        best = (ok if len(ok) else s).sort_values("OOS_Sharpe", ascending=False).iloc[0]
        strategies[name] = {"rows": [pack(x) for _, x in s.iterrows()], "example": example(name, best)}

    hunt = res[res.tujuan == "max Winrate"]
    data = {
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "universe": {k: {"tickers": v["tickers"], "cost": v["cost"], "short": v["short"]} for k, v in UNIVERSE.items()},
        "benchmark": [pack(x) for _, x in res[res.tujuan == "benchmark"].iterrows()],
        "strategies": strategies,
        "hunt": [{"kelas": x.kelas, "aset": x.aset, "strategi": x.strategi, "params": x.params,
                  "isWin": r(x.IS_Winrate), "oosWin": r(x.OOS_Winrate), "isN": int(x.IS_Trades), "oosN": int(x.OOS_Trades),
                  "oosExp": r(x.OOS_Expectancy)} for _, x in hunt.iterrows()],
        "musiman": json.loads((ROOT / "hasil_musiman.json").read_text(encoding="utf-8")),
    }
    tpl = (ROOT / "web" / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False, allow_nan=False))
    (ROOT / "web" / "index.html").write_text(html, encoding="utf-8")
    print("ok", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
