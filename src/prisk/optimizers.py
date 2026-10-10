"""Long-only portfolio weights with a per-asset cap. Every function returns a numpy array that
sums to 1 with 0 <= w_i <= cap. Inputs are at the daily scale; the solutions are scale-free."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _project(w: np.ndarray, cap: float) -> np.ndarray:
    """Clip to [0, cap] and renormalize until sum is 1 within 1e-12."""
    w = np.asarray(w, dtype=float)
    for _ in range(100):
        w = np.clip(w, 0, cap)
        s = np.sum(w)
        if abs(s - 1.0) < 1e-12:
            break
        w = w / s
    return w


def equal_weight(n: int) -> np.ndarray:
    """Equal weight portfolio."""
    return np.full(n, 1.0 / n)


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fraction of portfolio variance attributable to each asset; sums to 1."""
    cov_w = cov @ w
    portfolio_var = w @ cov_w
    return w * cov_w / portfolio_var


def min_variance(cov: np.ndarray, cap: float = 0.10) -> np.ndarray:
    """argmin w' cov w subject to sum w = 1, 0 <= w <= cap (SLSQP)."""
    n = cov.shape[0]
    if cap * n < 1.0:
        raise ValueError(f"cap {cap} * n {n} < 1: infeasible")

    def objective(w):
        return w @ cov @ w

    def gradient(w):
        return 2 * cov @ w

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, cap) for _ in range(n)]

    x0 = equal_weight(n)
    result = minimize(objective, x0, method="SLSQP", jac=gradient,
                      constraints=constraints, bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-12})

    w = result.x.copy()
    w = np.clip(w, 0, cap)
    w = w / np.sum(w)
    return w


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, cap: float = 0.10) -> np.ndarray:
    """argmax (w'mu - rf) / sqrt(w' cov w) subject to sum w = 1, 0 <= w <= cap."""
    n = cov.shape[0]

    def objective(w):
        ret = w @ mu - rf
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 1e10
        return -ret / vol

    def gradient(w):
        cov_w = cov @ w
        ret = w @ mu - rf
        vol_sq = w @ cov_w
        vol = np.sqrt(vol_sq)
        if vol < 1e-12:
            return np.zeros(n)
        grad = (-mu * vol + ret * cov_w / vol) / vol_sq
        return grad

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, cap) for _ in range(n)]

    starts = [equal_weight(n)]
    if cap * n >= 1.0:
        try:
            starts.append(min_variance(cov, cap))
        except:
            pass

    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.dirichlet(np.ones(n))
        d = _project(d, cap)
        starts.append(d)

    best_w = None
    best_sharpe = -np.inf

    for x0 in starts:
        try:
            result = minimize(objective, x0, method="SLSQP", jac=gradient,
                             constraints=constraints, bounds=bounds,
                             options={"maxiter": 500, "ftol": 1e-12})
            if result.success or result.fun < 1e9:
                w = result.x.copy()
                w = np.clip(w, 0, cap)
                w = w / np.sum(w)
                vol = np.sqrt(w @ cov @ w)
                if vol > 1e-12:
                    sharpe = (w @ mu - rf) / vol
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_w = w
        except:
            pass

    if best_w is None:
        if cap * n >= 1.0:
            return min_variance(cov, cap)
        else:
            return equal_weight(n)
    return best_w


def risk_parity(cov: np.ndarray, cap: float = 0.10) -> np.ndarray:
    """Equal risk contribution: minimise sum_i (RC_i - 1/n)^2."""
    n = cov.shape[0]

    def objective(w):
        rc = risk_contributions(w, cov)
        target = 1.0 / n
        return np.sum((rc - target) ** 2)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, cap) for _ in range(n)]

    x0 = equal_weight(n)
    result = minimize(objective, x0, method="SLSQP",
                      constraints=constraints, bounds=bounds,
                      options={"maxiter": 500, "ftol": 1e-14})

    w = _project(result.x, cap)
    w = w / np.sum(w)
    return np.clip(w, 0, cap)
