"""Tests for performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk import metrics, TRADING_DAYS


class TestAnnualizedReturn:
    def test_zero_return_series(self):
        r = pd.Series([0.0] * 252)
        assert metrics.annualized_return(r) == 0.0

    def test_constant_positive_return(self):
        r = pd.Series([0.01] * 252)
        expected = 1.01 ** 252 - 1
        assert np.isclose(metrics.annualized_return(r), expected, rtol=1e-6)


class TestAnnualizedVol:
    def test_zero_return_series(self):
        r = pd.Series([0.0] * 252)
        assert metrics.annualized_vol(r) == 0.0

    def test_vol_matches_formula(self):
        r = pd.Series(np.random.randn(252) * 0.01)
        expected = r.std(ddof=1) * np.sqrt(252)
        assert np.isclose(metrics.annualized_vol(r), expected, rtol=1e-6)


class TestSharpe:
    def test_zero_returns_zero_vol(self):
        r = pd.Series([0.0] * 252)
        result = metrics.sharpe(r, rf=0.0)
        assert np.isnan(result)

    def test_sharpe_with_zero_rf(self):
        np.random.seed(42)
        r = pd.Series([0.01 + 0.001 * x for x in np.random.randn(252)])
        mean_r = r.mean()
        std_r = r.std(ddof=1)
        expected = mean_r * 252 / (std_r * np.sqrt(252))
        assert np.isclose(metrics.sharpe(r, rf=0.0), expected, rtol=1e-6)

    def test_sharpe_with_series_rf(self):
        r = pd.Series([0.01] * 100, index=range(100))
        rf = pd.Series([0.002] * 100, index=range(100))
        result = metrics.sharpe(r, rf=rf)
        assert isinstance(result, float)


class TestSortino:
    def test_sortino_greater_than_sharpe(self):
        # Series with small negatives and large positives
        returns = np.array([0.02] * 200 + [-0.01] * 52)
        r = pd.Series(returns)
        sharpe_val = metrics.sharpe(r, rf=0.0)
        sortino_val = metrics.sortino(r, rf=0.0)
        assert sortino_val > sharpe_val

    def test_zero_downside_returns_nan(self):
        r = pd.Series([0.01] * 252)  # All positive
        result = metrics.sortino(r, rf=0.0)
        assert np.isnan(result)


class TestDrawdown:
    def test_simple_drawdown(self):
        r = pd.Series([0.1, -0.5, 0.2])
        dd = metrics.drawdown(r)
        assert dd.iloc[0] == 0.0
        # After -0.5: wealth = 1.1 * 0.5 = 0.55, cummax = 1.1, dd = 0.55/1.1 - 1 = -0.5
        assert np.isclose(dd.iloc[1], -0.5)

    def test_monotone_increasing_zero_drawdown(self):
        r = pd.Series([0.01] * 100)
        dd = metrics.drawdown(r)
        assert dd.min() == 0.0


class TestMaxDrawdown:
    def test_max_drawdown_simple(self):
        r = pd.Series([0.1, -0.5, 0.2])
        assert metrics.max_drawdown(r) == -0.5

    def test_monotone_increasing_zero(self):
        r = pd.Series([0.01] * 100)
        assert metrics.max_drawdown(r) == 0.0


class TestTurnover:
    def test_turnover_simple(self):
        # First row [0.5, 0.5] vs [0, 0]: 0.5 * (|0.5| + |0.5|) = 0.5
        # Second row [1.0, 0.0] vs [0.5, 0.5]: 0.5 * (|0.5| + |-0.5|) = 0.5
        weights = pd.DataFrame([[0.5, 0.5], [1.0, 0.0]])
        result = metrics.turnover(weights, periods_per_year=12)
        # Mean turnover per rebalance: (0.5 + 0.5) / 2 = 0.5
        # Annualized: 0.5 * 12 = 6.0
        assert np.isclose(result, 6.0, rtol=1e-6)


class TestSummary:
    def test_summary_keys(self):
        r = pd.Series(
            [0.01] * 252,
            index=pd.date_range("2020-01-01", periods=252)
        )
        result = metrics.summary(r)
        expected_keys = {
            "ann_return", "ann_vol", "sharpe", "sortino",
            "max_drawdown", "turnover", "n_days", "start", "end"
        }
        assert set(result.keys()) == expected_keys

    def test_summary_rounding(self):
        r = pd.Series(
            [0.01] * 252,
            index=pd.date_range("2020-01-01", periods=252)
        )
        result = metrics.summary(r)
        for key in ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"]:
            # Check it's rounded to 6 decimals
            val = result[key]
            if not np.isnan(val):
                # Round to 6 decimals and compare
                assert len(str(val).split(".")[-1]) <= 6

    def test_summary_turnover_none_without_weights(self):
        r = pd.Series(
            [0.01] * 252,
            index=pd.date_range("2020-01-01", periods=252)
        )
        result = metrics.summary(r, weights=None)
        assert result["turnover"] is None

    def test_summary_dates_iso_format(self):
        r = pd.Series(
            [0.01] * 252,
            index=pd.date_range("2020-01-01", periods=252)
        )
        result = metrics.summary(r)
        assert result["start"] == "2020-01-01"
        assert result["end"] == "2020-09-08"
