"""Performance metrics on daily return series. All annualisation uses 252 trading days."""

from __future__ import annotations

import pandas as pd

from . import TRADING_DAYS


def annualized_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric: (prod(1 + r)) ** (periods / len(r)) - 1."""
    return (1 + r).prod() ** (periods / len(r)) - 1


def annualized_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    """std(ddof=1) * sqrt(periods)."""
    return r.std(ddof=1) * (periods ** 0.5)


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised mean excess return over annualised vol of excess returns. rf is a daily series
    aligned to r, or a constant daily rate."""
    excess = r - rf
    mean_excess = excess.mean()
    std_excess = excess.std(ddof=1)
    if std_excess == 0:
        return float("nan")
    return mean_excess * periods / (std_excess * (periods ** 0.5))


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Like sharpe but the denominator is the annualised downside deviation (negative excess returns only)."""
    excess = r - rf
    mean_excess = excess.mean()
    downside_var = (pd.Series(excess).clip(upper=0) ** 2).mean()
    downside_vol = (downside_var ** 0.5) * (periods ** 0.5)
    if downside_vol == 0:
        return float("nan")
    return mean_excess * periods / downside_vol


def drawdown(r: pd.Series) -> pd.Series:
    """Drawdown series of the cumulative wealth curve (0 at highs, negative otherwise)."""
    wealth = (1 + r).cumprod()
    return wealth / wealth.cummax() - 1


def max_drawdown(r: pd.Series) -> float:
    """Most negative drawdown, as a negative number."""
    return drawdown(r).min()


def turnover(weights: pd.DataFrame, periods_per_year: int = 12) -> float:
    """Mean one-way turnover per rebalance, annualised: mean over rebalance dates of
    0.5 * sum |w_t - w_{t-1}|, times periods_per_year. The first row counts against zero weights."""
    prev_weights = weights.shift(1).fillna(0)
    turnover_per_rebalance = 0.5 * (weights - prev_weights).abs().sum(axis=1)
    return turnover_per_rebalance.mean() * periods_per_year


def summary(r: pd.Series, rf: pd.Series | float = 0.0, weights: pd.DataFrame | None = None) -> dict:
    """{"ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "turnover" (None without weights),
    "n_days", "start", "end"} with floats rounded to 6 decimals and dates as ISO strings."""
    return {
        "ann_return": round(annualized_return(r), 6),
        "ann_vol": round(annualized_vol(r), 6),
        "sharpe": round(sharpe(r, rf), 6),
        "sortino": round(sortino(r, rf), 6),
        "max_drawdown": round(max_drawdown(r), 6),
        "turnover": round(turnover(weights), 6) if weights is not None else None,
        "n_days": len(r),
        "start": r.index[0].strftime("%Y-%m-%d"),
        "end": r.index[-1].strftime("%Y-%m-%d"),
    }
