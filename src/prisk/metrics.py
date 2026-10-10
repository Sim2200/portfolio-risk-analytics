"""Performance metrics on daily return series. All annualisation uses 252 trading days."""

from __future__ import annotations

import pandas as pd

from . import TRADING_DAYS


def annualized_return(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    """Geometric: (prod(1 + r)) ** (periods / len(r)) - 1."""
    raise NotImplementedError


def annualized_vol(r: pd.Series, periods: int = TRADING_DAYS) -> float:
    raise NotImplementedError


def sharpe(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Annualised mean excess return over annualised vol of excess returns. rf is a daily series
    aligned to r, or a constant daily rate."""
    raise NotImplementedError


def sortino(r: pd.Series, rf: pd.Series | float = 0.0, periods: int = TRADING_DAYS) -> float:
    """Like sharpe but the denominator is the annualised downside deviation (negative excess returns only)."""
    raise NotImplementedError


def drawdown(r: pd.Series) -> pd.Series:
    """Drawdown series of the cumulative wealth curve (0 at highs, negative otherwise)."""
    raise NotImplementedError


def max_drawdown(r: pd.Series) -> float:
    """Most negative drawdown, as a negative number."""
    raise NotImplementedError


def turnover(weights: pd.DataFrame, periods_per_year: int = 12) -> float:
    """Mean one-way turnover per rebalance, annualised: mean over rebalance dates of
    0.5 * sum |w_t - w_{t-1}|, times periods_per_year. The first row counts against zero weights."""
    raise NotImplementedError


def summary(r: pd.Series, rf: pd.Series | float = 0.0, weights: pd.DataFrame | None = None) -> dict:
    """{"ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "turnover" (None without weights),
    "n_days", "start", "end"} with floats rounded to 6 decimals and dates as ISO strings."""
    raise NotImplementedError
