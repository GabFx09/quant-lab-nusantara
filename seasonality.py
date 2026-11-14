"""Pola siklus kalender + gabungan dengan pola teknikal.

A. Efek kalender murni per aset (dipilih di data lama, diuji di data baru):
   - Turn-of-month (TOM): hanya pegang 1 hari terakhir + 3 hari pertama bulan  [dari literatur, tidak di-fit]
   - Sell in May / Halloween: hanya pegang Nov–Apr                            [dari literatur, tidak di-fit]
   - Bulan terpilih: pegang hanya di bulan yang kuat di data lama               [di-fit]
   - Hari terpilih: pegang hanya di hari (Sen–Jum) yang kuat di data lama       [di-fit]
B. Persistensi: korelasi rata-rata return per bulan (dan per hari) data lama vs data baru.
C. Filter kalender pada 6 pola: pola hanya aktif di bucket kalender (bulan / hari /
   TOM vs bukan) yang return strateginya positif di data lama. Dibandingkan dengan tanpa filter.
D. Filter tren (bukan kalender): mean reversion hanya beli saat harga > SMA200
   (short hanya saat harga < SMA200). Parameter tetap, tidak di-fit.
"""
import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import metrics, run
from data import UNIVERSE, load
from strategies import STRATEGIES

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
SPLIT = 0.7
MIN_DAYS = 20          # bucket dengan data kurang dari ini tidak difilter (bukti kurang)
T_MIN = 1.0            # ambang t-stat untuk memilih bulan/hari pada strategi kalender murni
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
HARI = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]


def calendar_keys(idx):
    s = pd.Series(1, index=idx)
    ym = idx.year * 100 + idx.month
    first = s.groupby(ym).cumsum()
    last = s[::-1].groupby(ym[::-1]).cumsum()[::-1]
    tom = ((first <= 3) | (last <= 1)).astype(int)
    return {"bulan": pd.Series(idx.month, index=idx),
            "hari": pd.Series(idx.dayofweek, index=idx),
            "tom": tom}


def next_day(key):
    """Kalender hari berikutnya sudah diketahui hari ini -> sinyal hari t memakai key hari t+1."""
    return key.shift(-1).ffill()


def tstat(x):
    x = x.dropna()
    return x.mean() / x.std() * np.sqrt(len(x)) if len(x) > 2 and x.std() > 0 else 0.0


def m2d(m):
    return {k: (None if (isinstance(v, float) and (np.isnan(v) or np.isinf(v))) else round(float(v), 4))
            for k, v in m.items()}


def evaluate(df, sig, cfg, cut):
    strat, pos = run(df, sig, cfg["cost"], cfg["periods"])
    return metrics(strat[:cut], pos[:cut], cfg["periods"]), metrics(strat[cut:], pos[cut:], cfg["periods"])


def pick_buckets(ret, key, short):
    """Bucket dengan t-stat > T_MIN -> long; jika boleh short, t < -T_MIN -> short."""
    out = {}
    for b, g in ret.groupby(key):
        t = tstat(g)
        if t > T_MIN:
            out[int(b)] = 1.0
        elif short and t < -T_MIN:
            out[int(b)] = -1.0
    return out


def main():
    res = pd.read_csv(ROOT / "hasil_backtest.csv")
    base = res[res.tujuan == "max Sharpe"]
    pure, persist, filt, monthly, trend = [], [], [], [], []

    for cls, cfg in UNIVERSE.items():
        for tk in cfg["tickers"]:
            df = load(tk)
            cut = df.index[int(len(df) * SPLIT)]
            ret = df["Close"].pct_change()
            K = calendar_keys(df.index)
            nk = {k: next_day(v) for k, v in K.items()}
            is_, oos = df.index < cut, df.index >= cut

            # ---------- A. kalender murni ----------
            cands = {
                "Buy & Hold": pd.Series(1.0, index=df.index),
                "Turn-of-month": nk["tom"].astype(float),
                "Sell in May": nk["bulan"].isin([11, 12, 1, 2, 3, 4]).astype(float),
            }
            for nama, key in (("Bulan terpilih", "bulan"), ("Hari terpilih", "hari")):
                pick = pick_buckets(ret[is_], K[key][is_], cfg["short"])
                cands[nama] = nk[key].map(pick).fillna(0.0)
                cands[nama].attrs["pick"] = pick
            for nama, sig in cands.items():
                a, b = evaluate(df, sig, cfg, cut)
                pick = sig.attrs.get("pick")
                label = None
                if pick is not None:
                    names = BULAN if nama.startswith("Bulan") else HARI
                    label = ", ".join(("short " if v < 0 else "") + names[k - 1 if nama.startswith("Bulan") else k]
                                      for k, v in sorted(pick.items())) or "(tidak ada yang lolos)"
                pure.append(dict(kelas=cls, aset=tk, strategi=nama, pilihan=label, IS=m2d(a), OOS=m2d(b)))

            # ---------- B. persistensi ----------
            mret = (1 + ret).groupby([df.index.year, df.index.month]).prod() - 1
            mret.index.names = ["y", "m"]
            mret = mret.reset_index(name="r")
            mret["oos"] = pd.to_datetime(dict(year=mret.y, month=mret.m, day=1)) >= cut.replace(day=1)
            for _, row in mret.iterrows():
                monthly.append(dict(kelas=cls, aset=tk, m=int(row.m), r=float(row.r), oos=bool(row.oos)))
            row = dict(kelas=cls, aset=tk)
            for key in ("bulan", "hari"):
                a = ret[is_].groupby(K[key][is_]).mean()
                b = ret[oos].groupby(K[key][oos]).mean()
                j = pd.concat([a, b], axis=1).dropna()
                row[key] = round(float(j.iloc[:, 0].rank().corr(j.iloc[:, 1].rank())), 3)  # Spearman
            persist.append(row)

            # ---------- C. filter kalender pada pola ----------
            for _, br in base[base.aset == tk].iterrows():
                fn, _ = STRATEGIES[br.strategi]
                p = ast.literal_eval(br.params)
                sig = fn(df, short=cfg["short"], **p)
                strat, pos = run(df, sig, cfg["cost"], cfg["periods"])
                a0, b0 = metrics(strat[:cut], pos[:cut], cfg["periods"]), metrics(strat[cut:], pos[cut:], cfg["periods"])
                rec = dict(kelas=cls, aset=tk, strategi=br.strategi, params=br.params,
                           tanpa={"IS": m2d(a0), "OOS": m2d(b0)})
                active = is_ & (pos != 0).values
                for key in ("bulan", "hari", "tom"):
                    g = strat[active].groupby(K[key][active])
                    stat = pd.DataFrame({"mean": g.mean(), "n": g.size()})
                    blocked = [int(b) for b, s in stat.iterrows() if s.n >= MIN_DAYS and s["mean"] <= 0]
                    allow = ~nk[key].isin(blocked)
                    a, b = evaluate(df, sig * allow, cfg, cut)
                    rec[key] = {"blok": blocked, "IS": m2d(a), "OOS": m2d(b)}
                filt.append(rec)

                # ---------- D. filter tren SMA200 untuk mean reversion ----------
                if br.strategi in ("RSI revert", "Bollinger rev"):
                    up = df["Close"] > df["Close"].rolling(200).mean()
                    f = sig.where(((sig > 0) & up) | ((sig < 0) & ~up), 0)
                    a, b = evaluate(df, f, cfg, cut)
                    trend.append(dict(kelas=cls, aset=tk, strategi=br.strategi, params=br.params,
                                      tanpa={"IS": m2d(a0), "OOS": m2d(b0)}, trend={"IS": m2d(a), "OOS": m2d(b)}))

    out = {"pure": pure, "persist": persist, "filter": filt, "monthly": monthly, "trend": trend,
           "cfg": {"T_MIN": T_MIN, "MIN_DAYS": MIN_DAYS}}
    (ROOT / "hasil_musiman.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    report(out)


def report(o):
    pd.set_option("display.width", 220)
    P = pd.DataFrame([dict(kelas=x["kelas"], aset=x["aset"], strategi=x["strategi"], pilihan=x["pilihan"],
                           IS_Sharpe=x["IS"]["Sharpe"], OOS_Sharpe=x["OOS"]["Sharpe"], OOS_CAGR=x["OOS"]["CAGR"],
                           OOS_MaxDD=x["OOS"]["MaxDD"], OOS_Win=x["OOS"]["Winrate"]) for x in o["pure"]])
    print("=== A. Kalender murni: rata-rata per kelas ===")
    print(P.groupby(["kelas", "strategi"])[["IS_Sharpe", "OOS_Sharpe", "OOS_CAGR", "OOS_MaxDD"]].mean().round(2).to_string())
    bh = P[P.strategi == "Buy & Hold"].set_index("aset").OOS_Sharpe
    P["beat"] = P.OOS_Sharpe > P.aset.map(bh)
    print("\nKalahkan B&H (Sharpe data baru):")
    print(P[P.strategi != "Buy & Hold"].groupby(["strategi"]).beat.agg(["sum", "count"]).to_string())
    print("\nBulan terpilih:")
    print(P[P.strategi == "Bulan terpilih"][["aset", "pilihan", "IS_Sharpe", "OOS_Sharpe"]].to_string(index=False))
    print("\n=== B. Persistensi (Spearman data lama vs baru) ===")
    Q = pd.DataFrame(o["persist"]); print(Q.to_string(index=False)); print("rata-rata:", Q[["bulan", "hari"]].mean().round(3).to_dict())
    print("\n=== C. Filter kalender pada pola ===")
    rows = []
    for x in o["filter"]:
        for key in ("bulan", "hari", "tom"):
            rows.append(dict(kelas=x["kelas"], strategi=x["strategi"], filter=key,
                             dIS=x[key]["IS"]["Sharpe"] - x["tanpa"]["IS"]["Sharpe"],
                             dOOS=x[key]["OOS"]["Sharpe"] - x["tanpa"]["OOS"]["Sharpe"],
                             dWin=(x[key]["OOS"]["Winrate"] or 0) - (x["tanpa"]["OOS"]["Winrate"] or 0),
                             blok=len(x[key]["blok"])))
    R = pd.DataFrame(rows)
    R["membaik"] = R.dOOS > 0
    print(R.groupby("filter")[["dIS", "dOOS", "dWin", "membaik", "blok"]].mean().round(3).to_string())
    print(R.groupby(["kelas", "filter"])[["dIS", "dOOS", "membaik"]].mean().round(3).to_string())
    print(R.groupby(["strategi", "filter"])[["dOOS", "membaik"]].mean().round(3).unstack().to_string())


if __name__ == "__main__":
    main()
