"""Covariance estimators on daily returns (returned at the daily scale)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_cov(returns: pd.DataFrame) -> np.ndarray:
    """Unbiased sample covariance, (n, n)."""
    raise NotImplementedError


def ledoit_wolf_cov(returns: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage towards the scaled identity (scikit-learn's LedoitWolf). Returns the
    (n, n) matrix and the shrinkage intensity in [0, 1]."""
    raise NotImplementedError
