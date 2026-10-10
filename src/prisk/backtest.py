"""Walk-forward backtest with monthly rebalancing, transaction costs and no look-ahead.

For each month end t (the rebalance date), a strategy sees only the trailing `fit_months` of daily
returns ending at t, produces target weights, and those weights are held, drifting with prices,
through the next calendar month. The cost of moving from the drifted pre-rebalance weights to the
targets (one-way turnover times `cost_bps`) is charged on the first trading day of the new month.
Every reported number is therefore out of sample: no return in a test month is ever part of the
window that produced that month's weights.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from . import TRADING_DAYS
from .covariance import ledoit_wolf_cov, sample_cov
from .optimizers import equal_weight, max_sharpe, min_variance, risk_parity

# A strategy maps (trailing daily asset returns, daily risk-free rate) to target weights.
Strategy = Callable[[pd.DataFrame, float], np.ndarray]


def make_strategies(cap: float = 0.10) -> dict[str, Strategy]:
    """The seven portfolios compared in the report. Expected returns are trailing sample means;
    the two covariance estimators are the comparison of interest."""
    def mu(r: pd.DataFrame) -> np.ndarray:
        return r.mean().to_numpy()

    return {
        "equal_weight": lambda r, rf: equal_weight(r.shape[1]),
        "min_variance_sample": lambda r, rf: min_variance(sample_cov(r), cap),
        "min_variance_ledoit_wolf": lambda r, rf: min_variance(ledoit_wolf_cov(r)[0], cap),
        "max_sharpe_sample": lambda r, rf: max_sharpe(mu(r), sample_cov(r), rf, cap),
        "max_sharpe_ledoit_wolf": lambda r, rf: max_sharpe(mu(r), ledoit_wolf_cov(r)[0], rf, cap),
        "risk_parity_sample": lambda r, rf: risk_parity(sample_cov(r), cap),
        "risk_parity_ledoit_wolf": lambda r, rf: risk_parity(ledoit_wolf_cov(r)[0], cap),
    }


def month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each calendar month present in the index."""
    s = pd.Series(index, index=index)
    return pd.DatetimeIndex(s.groupby([index.year, index.month]).last().to_numpy())


def drift(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Weights after one day of returns r, without trading: w (1 + r) renormalised."""
    v = w * (1 + r)
    s = v.sum()
    return v / s if s > 0 else w


def run_strategy(returns: pd.DataFrame, strategy: Strategy, rf_daily: pd.Series, fit_months: int = 36,
                 cost_bps: float = 10.0, name: str = "") -> dict:
    """Backtest one strategy. Returns {"returns": daily net Series, "gross_returns", "weights" (targets
    at rebalance dates), "turnover" (one-way, per rebalance), "costs" (per rebalance, decimal),
    "fit_windows": [(fit_start, fit_end, test_start, test_end)]} for the look-ahead test."""
    ends = month_ends(returns.index)
    cost_rate = cost_bps / 1e4
    daily_gross, daily_net, targets, turnover, costs, windows = [], [], {}, {}, {}, []
    w_pre = np.zeros(returns.shape[1])
    for i in range(fit_months, len(ends) - 1):
        rebalance_at, next_end = ends[i], ends[i + 1]
        fit = returns.loc[ends[i - fit_months] : rebalance_at]
        fit = fit.iloc[1:]  # the day after the window's opening month end, so exactly fit_months months
        rf = float(rf_daily.reindex(fit.index).fillna(0.0).mean())
        w = np.asarray(strategy(fit, rf), dtype=float)
        test = returns.loc[rebalance_at:next_end].iloc[1:]  # strictly after the rebalance date
        if test.empty:
            continue
        assert fit.index.max() <= rebalance_at < test.index.min(), "look-ahead: test overlaps fit"
        windows.append((fit.index.min(), fit.index.max(), test.index.min(), test.index.max()))
        one_way = 0.5 * float(np.abs(w - w_pre).sum())
        cost = cost_rate * 2 * one_way  # cost_bps applies to every dollar traded (buys plus sells)
        targets[rebalance_at], turnover[rebalance_at], costs[rebalance_at] = w, one_way, cost
        cur = w
        for day, (_, row) in enumerate(test.iterrows()):
            r = row.to_numpy()
            gross = float(cur @ r)
            daily_gross.append((row.name, gross))
            daily_net.append((row.name, gross - (cost if day == 0 else 0.0)))
            cur = drift(cur, r)
        w_pre = cur
    idx = [d for d, _ in daily_net]
    return {
        "name": name,
        "returns": pd.Series([v for _, v in daily_net], index=pd.DatetimeIndex(idx), name=name),
        "gross_returns": pd.Series([v for _, v in daily_gross], index=pd.DatetimeIndex(idx), name=name),
        "weights": pd.DataFrame.from_dict(targets, orient="index", columns=returns.columns),
        "turnover": pd.Series(turnover),
        "costs": pd.Series(costs),
        "fit_windows": windows,
    }


def benchmark_spy(returns: pd.DataFrame, index_like: pd.DatetimeIndex) -> pd.Series:
    """Buy-and-hold SPY over the out-of-sample period (no rebalancing, no cost)."""
    return returns["SPY"].reindex(index_like).rename("spy")


def benchmark_60_40(returns: pd.DataFrame, index_like: pd.DatetimeIndex, cost_bps: float = 10.0) -> dict:
    """60% SPY / 40% AGG rebalanced at month ends with the same cost model, out-of-sample period only."""
    two = returns[["SPY", "AGG"]].reindex(index_like)
    fixed = lambda r, rf: np.array([0.6, 0.4])  # noqa: E731
    # run_strategy needs fit_months of history before the first test month; give it a 1-month fit on
    # the same data (the weights are constant, so the fit window is irrelevant)
    return run_strategy(two, fixed, pd.Series(0.0, index=two.index), fit_months=1, cost_bps=cost_bps, name="60_40")


def annual_turnover(result: dict) -> float:
    """Mean one-way turnover per rebalance times 12."""
    t = result["turnover"]
    return float(t.mean() * 12) if len(t) else float("nan")


def yearly_returns(r: pd.Series) -> dict[str, float]:
    return {str(y): round(float((1 + g).prod() - 1), 6) for y, g in r.groupby(r.index.year)}


def periods_per_year(r: pd.Series) -> float:
    return len(r) / max(1e-9, (r.index.max() - r.index.min()).days / 365.25)


__all__ = ["make_strategies", "run_strategy", "benchmark_spy", "benchmark_60_40", "annual_turnover",
           "yearly_returns", "month_ends", "drift", "TRADING_DAYS"]
