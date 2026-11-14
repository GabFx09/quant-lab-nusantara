"""Cari pola di saham, crypto, forex dengan validasi out-of-sample.

Untuk tiap aset x strategi:
  1. Grid-search parameter di 70% data awal (in-sample / IS)
  2. Ambil parameter terbaik (berdasarkan Sharpe IS, min 10 trade)
  3. Uji parameter itu di 30% data terakhir (out-of-sample / OOS)
Juga dilakukan "berburu winrate": parameter dengan winrate IS tertinggi,
lalu dicek apakah winrate itu bertahan di OOS.
"""
import itertools
import warnings

import pandas as pd

from backtest import metrics, run
from data import UNIVERSE, load
from strategies import STRATEGIES, buy_hold

warnings.filterwarnings("ignore")
SPLIT = 0.7
MIN_TRADES = 10


def evaluate(df, fn, params, cfg, cut):
    sig = fn(df, short=cfg["short"], **params)
    if sig is None:
        return None
    strat, pos = run(df, sig, cfg["cost"], cfg["periods"])
    return (metrics(strat[:cut], pos[:cut], cfg["periods"]),
            metrics(strat[cut:], pos[cut:], cfg["periods"]))


def main():
    rows = []
    for cls, cfg in UNIVERSE.items():
        for tk in cfg["tickers"]:
            df = load(tk)
            cut = df.index[int(len(df) * SPLIT)]
            print(f"{cls:6s} {tk:9s} {df.index[0].date()} .. {df.index[-1].date()}  OOS mulai {cut.date()}")
            bh_is, bh_oos = evaluate(df, buy_hold, {}, cfg, cut)
            rows.append(dict(kelas=cls, aset=tk, strategi="Buy & Hold", params="-", tujuan="benchmark",
                             **{f"IS_{k}": v for k, v in bh_is.items()},
                             **{f"OOS_{k}": v for k, v in bh_oos.items()}))
            for name, (fn, grid) in STRATEGIES.items():
                results = []
                for combo in itertools.product(*grid.values()):
                    p = dict(zip(grid.keys(), combo))
                    r = evaluate(df, fn, p, cfg, cut)
                    if r and r[0]["Trades"] >= MIN_TRADES:
                        results.append((p, *r))
                if not results:
                    continue
                for tujuan, key in (("max Sharpe", "Sharpe"), ("max Winrate", "Winrate")):
                    p, is_m, oos_m = max(results, key=lambda x: x[1][key])
                    rows.append(dict(kelas=cls, aset=tk, strategi=name, params=str(p), tujuan=tujuan,
                                     **{f"IS_{k}": v for k, v in is_m.items()},
                                     **{f"OOS_{k}": v for k, v in oos_m.items()}))
    res = pd.DataFrame(rows)
    res.to_csv("hasil_backtest.csv", index=False)
    return res


if __name__ == "__main__":
    main()
