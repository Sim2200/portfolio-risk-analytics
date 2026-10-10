"""Value at Risk and expected shortfall, their backtest, and historical stress windows.

Conventions: returns are daily decimals; VaR and CVaR are reported as positive loss numbers at
confidence `alpha` (0.99 = the 1% left tail); a breach is a day whose loss exceeds the VaR
forecast made with information up to the previous day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """-(the (1 - alpha) empirical quantile of r)."""
    return -np.quantile(r, 1 - alpha)


def cvar_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """Mean loss on the days beyond the historical VaR."""
    quantile = np.quantile(r, 1 - alpha)
    return -r[r <= quantile].mean()


def var_parametric(r: pd.Series, alpha: float = 0.99) -> float:
    """Normal: -(mean + z_{1-alpha} * std)."""
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu + z * sigma)


def cvar_parametric(r: pd.Series, alpha: float = 0.99) -> float:
    """Normal expected shortfall: -(mean - std * pdf(z) / (1 - alpha))."""
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = stats.norm.ppf(1 - alpha)
    return -(mu - sigma * stats.norm.pdf(z) / (1 - alpha))


def var_cvar_monte_carlo(asset_returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.99,
                         n_sims: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """Simulate portfolio returns from a multivariate normal fitted to the asset returns (sample
    mean and covariance) and read VaR and CVaR off the simulated distribution."""
    mean = asset_returns.mean().values
    cov = asset_returns.cov(ddof=1).values
    rng = np.random.default_rng(seed)
    sims = rng.multivariate_normal(mean, cov, n_sims) @ weights
    var = -np.quantile(sims, 1 - alpha)
    quantile = np.quantile(sims, 1 - alpha)
    cvar = -sims[sims <= quantile].mean()
    return var, cvar


def rolling_var_forecasts(r: pd.Series, window: int = 250, alpha: float = 0.99, method: str = "historical") -> pd.Series:
    """One-day-ahead VaR for each day t using returns [t - window, t - 1] only (no look-ahead);
    NaN for the first `window` days. method in {"historical", "parametric"}."""
    forecasts = pd.Series(index=r.index, dtype=float)
    for t in range(window, len(r)):
        window_returns = r.iloc[t - window:t]
        if method == "historical":
            forecasts.iloc[t] = var_historical(window_returns, alpha)
        elif method == "parametric":
            forecasts.iloc[t] = var_parametric(window_returns, alpha)
    return forecasts


def kupiec_pof(n_breaches: int, n_obs: int, alpha: float = 0.99) -> dict:
    """Kupiec proportion-of-failures likelihood-ratio test. Returns {"n_obs", "n_breaches",
    "expected_breaches", "breach_rate", "lr_statistic", "p_value", "reject_at_5pct"}."""
    p = 1 - alpha
    x = n_breaches
    n = n_obs
    expected_breaches = p * n
    breach_rate = x / n if n > 0 else 0

    # LR = -2 * ln[(1-p)^(n-x) p^x] + 2 * ln[(1-x/n)^(n-x) (x/n)^x]
    # Using 0 * ln(0) = 0 convention
    term1_numerator = 0
    if n - x > 0 and p > 0:
        term1_numerator += (n - x) * np.log(1 - p)
    if x > 0 and p > 0:
        term1_numerator += x * np.log(p)

    term1 = -2 * term1_numerator

    term2_numerator = 0
    if n - x > 0 and x < n:
        term2_numerator += (n - x) * np.log(1 - x / n)
    if x > 0:
        term2_numerator += x * np.log(x / n)

    term2 = 2 * term2_numerator
    lr_statistic = term1 + term2

    p_value = 1 - stats.chi2.cdf(lr_statistic, 1)
    reject_at_5pct = p_value < 0.05

    return {  # plain Python types so the dict serialises to JSON as-is
        "n_obs": int(n_obs),
        "n_breaches": int(n_breaches),
        "expected_breaches": round(float(expected_breaches), 4),
        "breach_rate": round(float(breach_rate), 6),
        "lr_statistic": round(float(lr_statistic), 6),
        "p_value": round(float(p_value), 6),
        "reject_at_5pct": bool(reject_at_5pct),
    }


def var_backtest(r: pd.Series, window: int = 250, alpha: float = 0.99, method: str = "historical") -> dict:
    """Rolling forecasts, breach count, Kupiec test; returns kupiec_pof's dict plus "method",
    "window", "alpha", "mean_var"."""
    forecasts = rolling_var_forecasts(r, window, alpha, method)
    realized_losses = -r

    # Find breaches where forecast exists and loss exceeds forecast
    breaches = (realized_losses > forecasts) & forecasts.notna()
    n_breaches = breaches.sum()
    n_obs = forecasts.notna().sum()

    kupiec_result = kupiec_pof(n_breaches, n_obs, alpha)
    mean_var = forecasts.mean()

    # Get breach dates (ISO strings, up to 50)
    breach_indices = breaches[breaches].index
    breach_dates = []
    for idx in breach_indices[:50]:
        if isinstance(idx, pd.Timestamp):
            breach_dates.append(idx.strftime("%Y-%m-%d"))
        else:
            breach_dates.append(str(idx))

    result = kupiec_result.copy()
    result.update({
        "method": method,
        "window": window,
        "alpha": alpha,
        "mean_var": mean_var,
        "breach_dates": breach_dates,
    })
    return result


STRESS_WINDOWS = {
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "rate_shock_2022": ("2022-01-03", "2022-10-12"),
}


def stress_test(r: pd.Series, windows: dict[str, tuple[str, str]] = STRESS_WINDOWS) -> dict:
    """For each window: {"start", "end", "cumulative_return", "max_drawdown", "worst_day", "n_days"}."""
    from . import metrics

    result = {}
    for window_name, (start_str, end_str) in windows.items():
        start_date = pd.Timestamp(start_str)
        end_date = pd.Timestamp(end_str)

        # Slice the series inclusive
        mask = (r.index >= start_date) & (r.index <= end_date)
        slice_r = r[mask]

        if len(slice_r) == 0:
            result[window_name] = {
                "start": start_str,
                "end": end_str,
                "cumulative_return": None,
                "max_drawdown": None,
                "worst_day": None,
                "n_days": 0,
            }
        else:
            cumulative_return = (1 + slice_r).prod() - 1
            max_dd = metrics.max_drawdown(slice_r)
            worst_day = slice_r.min()

            result[window_name] = {
                "start": start_str,
                "end": end_str,
                "cumulative_return": cumulative_return,
                "max_drawdown": max_dd,
                "worst_day": worst_day,
                "n_days": len(slice_r),
            }

    return result
