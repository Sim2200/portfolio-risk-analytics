"""Value at Risk and expected shortfall, their backtest, and historical stress windows.

Conventions: returns are daily decimals; VaR and CVaR are reported as positive loss numbers at
confidence `alpha` (0.99 = the 1% left tail); a breach is a day whose loss exceeds the VaR
forecast made with information up to the previous day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """-(the (1 - alpha) empirical quantile of r)."""
    raise NotImplementedError


def cvar_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """Mean loss on the days beyond the historical VaR."""
    raise NotImplementedError


def var_parametric(r: pd.Series, alpha: float = 0.99) -> float:
    """Normal: -(mean + z_{1-alpha} * std)."""
    raise NotImplementedError


def cvar_parametric(r: pd.Series, alpha: float = 0.99) -> float:
    """Normal expected shortfall: -(mean - std * pdf(z) / (1 - alpha))."""
    raise NotImplementedError


def var_cvar_monte_carlo(asset_returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.99,
                         n_sims: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """Simulate portfolio returns from a multivariate normal fitted to the asset returns (sample
    mean and covariance) and read VaR and CVaR off the simulated distribution."""
    raise NotImplementedError


def rolling_var_forecasts(r: pd.Series, window: int = 250, alpha: float = 0.99, method: str = "historical") -> pd.Series:
    """One-day-ahead VaR for each day t using returns [t - window, t - 1] only (no look-ahead);
    NaN for the first `window` days. method in {"historical", "parametric"}."""
    raise NotImplementedError


def kupiec_pof(n_breaches: int, n_obs: int, alpha: float = 0.99) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test. Returns {"n_obs", "n_breaches",
    "expected_breaches", "breach_rate", "lr_statistic", "p_value", "reject_at_5pct"}."""
    raise NotImplementedError


def var_backtest(r: pd.Series, window: int = 250, alpha: float = 0.99, method: str = "historical") -> dict:
    """Rolling forecasts, breach count, Kupiec test; returns kupiec_pof's dict plus "method",
    "window", "alpha", "mean_var"."""
    raise NotImplementedError


STRESS_WINDOWS = {
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "rate_shock_2022": ("2022-01-03", "2022-10-12"),
}


def stress_test(r: pd.Series, windows: dict[str, tuple[str, str]] = STRESS_WINDOWS) -> dict:
    """For each window: {"start", "end", "cumulative_return", "max_drawdown", "worst_day", "n_days"}."""
    raise NotImplementedError
