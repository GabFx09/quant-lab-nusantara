"""Backtest vektor + metrik."""
import numpy as np
import pandas as pd


def run(df, signal, cost, periods):
    ret = df["Close"].pct_change().fillna(0)
    pos = signal.shift(1).fillna(0)             # eksekusi di bar berikutnya
    turnover = pos.diff().abs().fillna(pos.abs())
    strat = pos * ret - turnover * cost
    return strat, pos


def trades(strat, pos):
    """Pecah equity menjadi trade: blok berturut-turut dengan posisi sama (≠0)."""
    block = (pos != pos.shift()).cumsum()
    out = []
    for _, g in strat[pos != 0].groupby(block[pos != 0]):
        out.append((1 + g).prod() - 1)
    return np.array(out)


def metrics(strat, pos, periods):
    eq = (1 + strat).cumprod()
    yrs = len(strat) / periods
    t = trades(strat, pos)
    wins, losses = t[t > 0], t[t <= 0]
    sd = strat.std()
    return {
        "CAGR": eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else -1.0,
        "Sharpe": strat.mean() / sd * np.sqrt(periods) if sd > 0 else 0.0,
        "MaxDD": (eq / eq.cummax() - 1).min(),
        "Trades": len(t),
        "Winrate": len(wins) / len(t) if len(t) else np.nan,
        "AvgWin": wins.mean() if len(wins) else 0.0,
        "AvgLoss": losses.mean() if len(losses) else 0.0,
        "ProfitFactor": wins.sum() / -losses.sum() if losses.sum() < 0 else np.inf,
        "Expectancy": t.mean() if len(t) else np.nan,
        "Exposure": (pos != 0).mean(),
    }
