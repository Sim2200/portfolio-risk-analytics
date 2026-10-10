"""Tests for portfolio optimizers and covariance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prisk.covariance import ledoit_wolf_cov, sample_cov
from prisk.optimizers import (equal_weight, max_sharpe, min_variance,
                               risk_contributions, risk_parity)


@pytest.fixture
def synthetic_data():
    """5-asset synthetic data with deterministic covariance and returns."""
    rng = np.random.default_rng(0)

    n_assets = 5
    n_obs = 252

    corr = rng.uniform(-0.3, 0.9, (n_assets, n_assets))
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1.0)

    vols = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) / np.sqrt(252)
    D = np.diag(vols)
    cov = D @ corr @ D

    mus = np.array([0.05, 0.08, 0.1, 0.12, 0.15]) / 252

    returns = pd.DataFrame(
        rng.multivariate_normal(np.zeros(n_assets), cov, n_obs),
        columns=[f"asset_{i}" for i in range(n_assets)]
    )

    return cov, mus, returns


class TestWeights:
    """Test that all optimizers return valid weights."""

    def test_equal_weight_sums_to_one(self):
        """Equal weight returns weights summing to 1."""
        for n in [1, 5, 10]:
            w = equal_weight(n)
            assert np.isclose(np.sum(w), 1.0)
            assert np.allclose(w, 1.0 / n)

    def test_min_variance_valid_weights(self, synthetic_data):
        """min_variance returns valid weights."""
        cov, _, _ = synthetic_data
        for cap in [0.25, 0.5]:
            w = min_variance(cov, cap=cap)
            assert np.isclose(np.sum(w), 1.0, atol=1e-8)
            assert np.all(w >= -1e-9)
            assert np.all(w <= cap + 1e-9)

    def test_max_sharpe_valid_weights(self, synthetic_data):
        """max_sharpe returns valid weights."""
        cov, mus, _ = synthetic_data
        for cap in [0.25, 0.5]:
            w = max_sharpe(mus, cov, rf=0.0, cap=cap)
            assert np.isclose(np.sum(w), 1.0, atol=1e-8)
            assert np.all(w >= -1e-9)
            assert np.all(w <= cap + 1e-9)

    def test_risk_parity_valid_weights(self, synthetic_data):
        """risk_parity returns valid weights."""
        cov, _, _ = synthetic_data
        for cap in [0.25, 1.0]:
            w = risk_parity(cov, cap=cap)
            assert np.isclose(np.sum(w), 1.0, atol=1e-8)
            assert np.all(w >= -1e-9)
            assert np.all(w <= cap + 1e-9)


class TestMinVariance:
    """Test min_variance optimizer."""

    def test_min_variance_better_than_equal(self, synthetic_data):
        """min_variance variance <= equal weight variance."""
        cov, _, _ = synthetic_data
        w_mv = min_variance(cov, cap=0.25)
        w_eq = equal_weight(5)

        var_mv = w_mv @ cov @ w_mv
        var_eq = w_eq @ cov @ w_eq

        assert var_mv <= var_eq + 1e-8

    def test_min_variance_better_than_random(self, synthetic_data):
        """min_variance variance <= 200 random feasible portfolios."""
        from prisk.optimizers import _project

        cov, _, _ = synthetic_data
        w_mv = min_variance(cov, cap=0.25)
        var_mv = w_mv @ cov @ w_mv

        rng = np.random.default_rng(42)
        n = cov.shape[0]
        cap = 0.25

        worse_count = 0
        for _ in range(200):
            d = rng.dirichlet(np.ones(n))
            d = _project(d, cap)
            var_rand = d @ cov @ d
            if var_rand < var_mv - 1e-8:
                worse_count += 1

        assert worse_count < 3

    def test_min_variance_infeasible_cap(self, synthetic_data):
        """min_variance raises ValueError for infeasible cap."""
        cov, _, _ = synthetic_data
        with pytest.raises(ValueError):
            min_variance(cov, cap=0.10)


class TestMaxSharpe:
    """Test max_sharpe optimizer."""

    def test_max_sharpe_better_than_equal(self, synthetic_data):
        """max_sharpe Sharpe >= equal weight Sharpe."""
        cov, mus, _ = synthetic_data
        w_ms = max_sharpe(mus, cov, rf=0.0, cap=0.5)
        w_eq = equal_weight(5)

        sharpe_ms = (w_ms @ mus) / np.sqrt(w_ms @ cov @ w_ms)
        sharpe_eq = (w_eq @ mus) / np.sqrt(w_eq @ cov @ w_eq)

        assert sharpe_ms >= sharpe_eq - 1e-6

    def test_max_sharpe_better_than_min_var(self, synthetic_data):
        """max_sharpe Sharpe >= min_variance Sharpe."""
        cov, mus, _ = synthetic_data
        w_ms = max_sharpe(mus, cov, rf=0.0, cap=0.5)
        w_mv = min_variance(cov, cap=0.5)

        sharpe_ms = (w_ms @ mus) / np.sqrt(w_ms @ cov @ w_ms)
        sharpe_mv = (w_mv @ mus) / np.sqrt(w_mv @ cov @ w_mv)

        assert sharpe_ms >= sharpe_mv - 1e-6

    def test_max_sharpe_better_than_random(self, synthetic_data):
        """max_sharpe Sharpe >= 200 random feasible portfolios."""
        from prisk.optimizers import _project

        cov, mus, _ = synthetic_data
        w_ms = max_sharpe(mus, cov, rf=0.0, cap=0.5)
        sharpe_ms = (w_ms @ mus) / np.sqrt(w_ms @ cov @ w_ms)

        rng = np.random.default_rng(42)
        n = cov.shape[0]
        cap = 0.5

        worse_count = 0
        for _ in range(200):
            d = rng.dirichlet(np.ones(n))
            d = _project(d, cap)
            sharpe_rand = (d @ mus) / np.sqrt(d @ cov @ d)
            if sharpe_rand > sharpe_ms + 1e-6:
                worse_count += 1

        assert worse_count < 3

    def test_max_sharpe_respects_cap(self, synthetic_data):
        """max_sharpe respects cap=0.25."""
        cov, mus, _ = synthetic_data
        w = max_sharpe(mus, cov, rf=0.0, cap=0.25)
        assert np.all(w <= 0.25 + 1e-9)


class TestRiskParity:
    """Test risk_parity optimizer."""

    def test_risk_parity_equal_contributions(self, synthetic_data):
        """risk_parity has equal risk contributions when cap is loose."""
        cov, _, _ = synthetic_data
        w = risk_parity(cov, cap=1.0)
        rc = risk_contributions(w, cov)
        target = 1.0 / 5

        assert np.allclose(rc, target, atol=1e-4)

    def test_risk_parity_respects_cap(self, synthetic_data):
        """risk_parity respects cap=0.25."""
        cov, _, _ = synthetic_data
        w = risk_parity(cov, cap=0.25)
        assert np.all(w <= 0.25 + 1e-9)


class TestRiskContributions:
    """Test risk_contributions function."""

    def test_risk_contributions_sum_to_one(self, synthetic_data):
        """risk_contributions sum to 1."""
        cov, _, _ = synthetic_data
        w = equal_weight(5)
        rc = risk_contributions(w, cov)
        assert np.isclose(np.sum(rc), 1.0)


class TestLedoitWolf:
    """Test Ledoit-Wolf covariance estimator."""

    def test_ledoit_wolf_valid_shrinkage(self, synthetic_data):
        """LedoitWolf shrinkage in [0, 1]."""
        _, _, returns = synthetic_data
        cov, shrink = ledoit_wolf_cov(returns)

        assert 0 <= shrink <= 1
        assert cov.shape == (5, 5)

    def test_ledoit_wolf_positive_definite(self, synthetic_data):
        """LedoitWolf matrix is symmetric positive definite."""
        _, _, returns = synthetic_data
        cov, _ = ledoit_wolf_cov(returns)

        assert np.allclose(cov, cov.T)

        eigvals = np.linalg.eigvalsh(cov)
        assert np.all(eigvals > 0)

    def test_ledoit_wolf_lower_condition_number(self):
        """LedoitWolf matrix has lower condition number with few observations."""
        rng = np.random.default_rng(0)
        n_assets = 5
        n_obs = 20

        corr = rng.uniform(-0.3, 0.9, (n_assets, n_assets))
        corr = (corr + corr.T) / 2
        np.fill_diagonal(corr, 1.0)

        vols = np.array([0.1, 0.15, 0.2, 0.25, 0.3]) / np.sqrt(252)
        D = np.diag(vols)
        true_cov = D @ corr @ D

        returns = pd.DataFrame(
            rng.multivariate_normal(np.zeros(n_assets), true_cov, n_obs),
            columns=[f"asset_{i}" for i in range(n_assets)]
        )

        sample = sample_cov(returns)
        lw_cov, _ = ledoit_wolf_cov(returns)

        cond_sample = np.linalg.cond(sample)
        cond_lw = np.linalg.cond(lw_cov)

        assert cond_lw < cond_sample


class TestAnalytic:
    """Test analytic solutions."""

    def test_min_variance_identity_cov(self):
        """min_variance with identity covariance returns [0.5, 0.5]."""
        cov = np.eye(2)
        w = min_variance(cov, cap=1.0)

        assert np.allclose(w, [0.5, 0.5])
