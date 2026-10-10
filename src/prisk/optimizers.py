"""Long-only portfolio weights with a per-asset cap. Every function returns a numpy array that
sums to 1 with 0 <= w_i <= cap. Inputs are at the daily scale; the solutions are scale-free."""

from __future__ import annotations

import numpy as np


def equal_weight(n: int) -> np.ndarray:
    raise NotImplementedError


def min_variance(cov: np.ndarray, cap: float = 0.10) -> np.ndarray:
    """argmin w' cov w subject to sum w = 1, 0 <= w <= cap (SLSQP)."""
    raise NotImplementedError


def max_sharpe(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0, cap: float = 0.10) -> np.ndarray:
    """argmax (w'mu - rf) / sqrt(w' cov w) subject to sum w = 1, 0 <= w <= cap. Non-convex under the
    cap, so SLSQP from several starting points (equal weight, min variance, a few random simplex
    points) and the best feasible result is returned."""
    raise NotImplementedError


def risk_parity(cov: np.ndarray, cap: float = 0.10) -> np.ndarray:
    """Equal risk contribution: minimise sum_i (RC_i - 1/n)^2 where RC_i = w_i (cov w)_i / (w' cov w),
    subject to sum w = 1, 0 <= w <= cap."""
    raise NotImplementedError


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Fraction of portfolio variance attributable to each asset; sums to 1."""
    raise NotImplementedError
