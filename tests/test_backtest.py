"""Walk-forward backtest: no look-ahead, correct cost accounting, weights drift between rebalances."""

import numpy as np
import pandas as pd
import pytest

from prisk.backtest import benchmark_60_40, drift, month_ends, run_strategy


def synthetic_returns(n_assets: int = 4, years: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=252 * years)
    r = rng.normal(0.0003, 0.01, size=(len(idx), n_assets))
    return pd.DataFrame(r, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_month_ends_are_last_trading_days():
    idx = pd.bdate_range("2020-01-01", "2020-03-31")
    ends = month_ends(idx)
    assert list(ends.strftime("%Y-%m-%d")) == ["2020-01-31", "2020-02-28", "2020-03-31"]


def test_no_look_ahead_every_window():
    r = synthetic_returns()
    seen = []

    def spy_strategy(fit, rf):
        seen.append((fit.index.min(), fit.index.max(), len(fit)))
        return np.full(fit.shape[1], 1 / fit.shape[1])

    out = run_strategy(r, spy_strategy, pd.Series(0.0, index=r.index), fit_months=12)
    for (fit_start, fit_end, test_start, test_end), (s, e, n) in zip(out["fit_windows"], seen):
        assert fit_end < test_start  # the strategy never saw a test-month return
        assert (s, e) == (fit_start, fit_end)
    # fit windows are about 12 months long (within a few trading days)
    assert all(230 <= n <= 270 for _, _, n in seen)
    # the out-of-sample series starts after fit_months of history
    assert out["returns"].index.min() > r.index.min() + pd.Timedelta(days=300)


def test_cost_is_charged_once_per_rebalance_on_the_first_day():
    r = synthetic_returns(n_assets=2)
    alternating = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    calls = {"n": 0}

    def flip(fit, rf):
        w = alternating[calls["n"] % 2]
        calls["n"] += 1
        return w

    out = run_strategy(r, flip, pd.Series(0.0, index=r.index), fit_months=3, cost_bps=10)
    diff = (out["gross_returns"] - out["returns"]).round(12)
    charged = diff[diff > 0]
    # first rebalance moves from zero to fully invested (one-way 0.5), then full flips (one-way 1.0)
    assert len(charged) == len(out["costs"])
    assert out["costs"].iloc[0] == pytest.approx(10 / 1e4 * 2 * 0.5)
    assert out["costs"].iloc[1] == pytest.approx(10 / 1e4 * 2 * 1.0)
    assert set(charged.index) == set(out["returns"].groupby([out["returns"].index.year, out["returns"].index.month]).apply(lambda s: s.index.min()))


def test_weights_drift_with_returns_between_rebalances():
    w = np.array([0.5, 0.5])
    moved = drift(w, np.array([0.10, -0.10]))
    assert moved[0] > 0.5 > moved[1] and moved.sum() == pytest.approx(1.0)
    r = synthetic_returns(n_assets=2)
    out = run_strategy(r, lambda fit, rf: np.array([0.5, 0.5]), pd.Series(0.0, index=r.index), fit_months=3)
    # with drift, a rebalance back to 50/50 trades a little each month: turnover > 0 after the first
    assert (out["turnover"].iloc[1:] > 0).all()
    assert out["turnover"].iloc[1:].max() < 0.05


def test_60_40_benchmark_holds_fixed_weights():
    r = synthetic_returns(n_assets=2).rename(columns={"A0": "SPY", "A1": "AGG"})
    out = benchmark_60_40(r, r.index)
    assert (out["weights"]["SPY"] == 0.6).all() and (out["weights"]["AGG"] == 0.4).all()
    assert len(out["returns"]) > 0
