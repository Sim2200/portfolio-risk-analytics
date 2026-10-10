"""Tests for risk measures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from prisk import risk


class TestVarHistorical:
    def test_known_quantile(self):
        r = pd.Series(np.arange(-100, 901) / 100)  # -1 to 9
        var = risk.var_historical(r, alpha=0.99)
        expected = -np.quantile(r, 0.01)
        assert np.isclose(var, expected)

    def test_normal_samples_within_tolerance(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(1000) * 0.01)
        var_hist = risk.var_historical(r, alpha=0.99)
        var_param = risk.var_parametric(r, alpha=0.99)
        assert abs(var_hist - var_param) < 0.01  # Within 1% loss


class TestCvarHistorical:
    def test_cvar_greater_equal_var(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(1000) * 0.01)
        var = risk.var_historical(r, alpha=0.99)
        cvar = risk.cvar_historical(r, alpha=0.99)
        assert cvar >= var


class TestVarParametric:
    def test_parametric_var_formula(self):
        r = pd.Series([0.01, -0.02, 0.015, -0.005] * 250)
        var = risk.var_parametric(r, alpha=0.99)
        mu = r.mean()
        sigma = r.std(ddof=1)
        z = scipy_stats.norm.ppf(1 - 0.99)
        expected = -(mu + z * sigma)
        assert np.isclose(var, expected, rtol=1e-6)


class TestCvarParametric:
    def test_cvar_greater_equal_var_parametric(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(1000) * 0.01)
        var = risk.var_parametric(r, alpha=0.99)
        cvar = risk.cvar_parametric(r, alpha=0.99)
        assert cvar >= var


class TestVarCvarMonteCarlo:
    def test_single_asset_matches_parametric(self):
        np.random.seed(42)
        asset_returns = pd.DataFrame(np.random.randn(1000) * 0.01, columns=["A"])
        weights = np.array([1.0])
        var_mc, cvar_mc = risk.var_cvar_monte_carlo(asset_returns, weights, alpha=0.99)
        var_param = risk.var_parametric(asset_returns["A"], alpha=0.99)
        assert abs(var_mc - var_param) < 0.01

    def test_two_assets_weights(self):
        np.random.seed(42)
        asset_returns = pd.DataFrame(np.random.randn(1000, 2) * 0.01, columns=["A", "B"])
        weights = np.array([1.0, 0.0])
        var_mc, cvar_mc = risk.var_cvar_monte_carlo(asset_returns, weights, alpha=0.99, n_sims=10000)
        var_param = risk.var_parametric(asset_returns["A"], alpha=0.99)
        assert abs(var_mc - var_param) < 0.01


class TestRollingVarForecasts:
    def test_first_window_rows_nan(self):
        r = pd.Series(np.random.randn(500) * 0.01)
        window = 250
        forecasts = risk.rolling_var_forecasts(r, window=window, alpha=0.99)
        assert forecasts.iloc[0:window].isna().all()

    def test_forecast_matches_historical(self):
        r = pd.Series(np.random.randn(500) * 0.01)
        window = 250
        forecasts = risk.rolling_var_forecasts(r, window=window, alpha=0.99, method="historical")
        # Check that forecast at index t matches var_historical of r[t-window:t]
        t = 300
        expected = risk.var_historical(r.iloc[t - window:t], alpha=0.99)
        assert np.isclose(forecasts.iloc[t], expected, rtol=1e-6)

    def test_parametric_method(self):
        r = pd.Series(np.random.randn(500) * 0.01)
        window = 250
        forecasts = risk.rolling_var_forecasts(r, window=window, alpha=0.99, method="parametric")
        assert forecasts.iloc[window:].notna().all()


class TestKupiecPof:
    def test_no_breaches_no_error(self):
        result = risk.kupiec_pof(n_breaches=0, n_obs=1000, alpha=0.99)
        assert result["n_breaches"] == 0
        assert np.isclose(result["expected_breaches"], 10)

    def test_expected_breaches_no_rejection(self):
        result = risk.kupiec_pof(n_breaches=10, n_obs=1000, alpha=0.99)
        assert result["n_obs"] == 1000
        assert result["n_breaches"] == 10
        assert np.isclose(result["expected_breaches"], 10)
        assert result["breach_rate"] == 0.01
        assert not result["reject_at_5pct"]
        assert result["lr_statistic"] < 1  # LR should be near 0

    def test_many_breaches_rejection(self):
        result = risk.kupiec_pof(n_breaches=30, n_obs=1000, alpha=0.99)
        assert result["reject_at_5pct"]


class TestVarBacktest:
    def test_backtest_structure(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(500) * 0.01)
        result = risk.var_backtest(r, window=250, alpha=0.99)
        expected_keys = {
            "n_obs", "n_breaches", "expected_breaches", "breach_rate",
            "lr_statistic", "p_value", "reject_at_5pct", "method",
            "window", "alpha", "mean_var", "breach_dates"
        }
        assert set(result.keys()) == expected_keys

    def test_breach_rate_reasonable(self):
        np.random.seed(42)
        r = pd.Series(np.random.randn(4000) * 0.01)
        result = risk.var_backtest(r, window=250, alpha=0.99, method="historical")
        breach_rate = result["breach_rate"]
        # Should be roughly between 0.3% and 3% for iid normal
        assert 0.003 <= breach_rate <= 0.03

    def test_breach_dates_format(self):
        r = pd.Series(
            np.random.randn(500) * 0.01,
            index=pd.date_range("2020-01-01", periods=500)
        )
        result = risk.var_backtest(r, window=250, alpha=0.99)
        if result["breach_dates"]:
            # Check ISO format
            assert len(result["breach_dates"][0]) == 10
            assert result["breach_dates"][0].count("-") == 2


class TestStressTest:
    def test_stress_test_empty_window(self):
        r = pd.Series(
            [0.01] * 100,
            index=pd.date_range("2020-01-01", periods=100)
        )
        result = risk.stress_test(r, windows={"test": ("2025-01-01", "2025-12-31")})
        assert result["test"]["n_days"] == 0
        assert result["test"]["cumulative_return"] is None

    def test_stress_test_full_window(self):
        r = pd.Series(
            [0.01] * 100,
            index=pd.date_range("2020-01-01", periods=100)
        )
        result = risk.stress_test(r, windows={"test": ("2020-01-01", "2020-04-09")})
        assert result["test"]["n_days"] == 100
        cumulative_return = 1.01 ** 100 - 1
        assert np.isclose(result["test"]["cumulative_return"], cumulative_return, rtol=1e-6)

    def test_stress_test_partial_window(self):
        r = pd.Series(
            [0.01] * 100,
            index=pd.date_range("2020-01-01", periods=100)
        )
        result = risk.stress_test(r, windows={"test": ("2020-01-01", "2020-01-31")})
        assert result["test"]["n_days"] == 31

    def test_stress_test_worst_day(self):
        returns = [0.01] * 50 + [-0.05] + [0.01] * 50
        r = pd.Series(returns, index=pd.date_range("2020-01-01", periods=101))
        result = risk.stress_test(r, windows={"test": ("2020-01-01", "2020-04-10")})
        assert result["test"]["worst_day"] == -0.05

    def test_stress_test_max_drawdown(self):
        returns = [0.01] * 50 + [-0.02] + [0.01] * 50
        r = pd.Series(returns, index=pd.date_range("2020-01-01", periods=101))
        result = risk.stress_test(r, windows={"test": ("2020-01-01", "2020-04-10")})
        assert result["test"]["max_drawdown"] < 0
