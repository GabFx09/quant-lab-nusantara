"""Metrik performa."""
import math

import numpy as np
import pandas as pd


def _phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def stats(r, periods, turnover=None, cost=None):
    r = r.dropna()
    if len(r) < 20:
        return {}
    eq = (1 + r).cumprod()
    yrs = len(r) / periods
    mu, sd = r.mean(), r.std()
    down = r[r < 0].std()
    sr = mu / sd if sd > 0 else 0.0
    dd = eq / eq.cummax() - 1
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if eq.iloc[-1] > 0 else -1.0
    sk, ku = float(r.skew()), float(r.kurt()) + 3
    denom = 1 - sk * sr + (ku - 1) / 4 * sr ** 2
    psr = _phi(sr * math.sqrt(len(r) - 1) / math.sqrt(denom)) if denom > 0 else float("nan")
    monthly = (1 + r).resample("ME").prod() - 1
    out = {
        "CAGR": cagr, "Vol": sd * math.sqrt(periods), "Sharpe": sr * math.sqrt(periods),
        "Sortino": mu / down * math.sqrt(periods) if down > 0 else float("nan"),
        "MaxDD": dd.min(), "Calmar": cagr / -dd.min() if dd.min() < 0 else float("nan"),
        "PSR": psr, "tstat": sr * math.sqrt(len(r)),
        "BulanPositif": float((monthly > 0).mean()), "Tahun": yrs,
        "Mulai": r.index[0].strftime("%Y-%m-%d"),
    }
    if turnover is not None:
        out["TurnoverTahunan"] = float(turnover.sum() / yrs)
        if cost is not None:
            out["BiayaTahunan"] = float(turnover.sum() * cost / yrs)
    return out


def clean(d):
    def f(v):
        if isinstance(v, (float, np.floating)):
            return None if (np.isnan(v) or np.isinf(v)) else round(float(v), 5)
        return v
    return {k: f(v) for k, v in d.items()}


def yearly(r):
    return (1 + r).groupby(r.index.year).prod() - 1


def monthly_table(r):
    m = (1 + r).resample("ME").prod() - 1
    return [{"y": int(d.year), "m": int(d.month), "r": round(float(v), 5)} for d, v in m.items()]


__all__ = ["stats", "clean", "yearly", "monthly_table", "pd"]
