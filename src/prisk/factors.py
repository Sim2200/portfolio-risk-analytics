"""Factor attribution: Fama-French 5 factors plus momentum, by OLS on daily excess returns.

    r_t - RF_t = alpha + b_mkt (Mkt-RF)_t + b_smb SMB_t + b_hml HML_t + b_rmw RMW_t + b_cma CMA_t + b_mom Mom_t + e_t

Standard errors are Newey-West (HAC) with 5 lags, since daily returns are autocorrelated under
volatility clustering. Alpha is reported per day and annualised (x 252).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import TRADING_DAYS

FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]


def regress(r: pd.Series, factors: pd.DataFrame, hac_lags: int = 5) -> dict:
    """Returns {"n", "alpha_daily", "alpha_annualized", "alpha_t", "alpha_p", "betas": {factor: b},
    "t_stats": {factor: t}, "r2", "r2_adj", "residual_vol_annualized"}."""
    df = pd.concat([r.rename("r"), factors[FACTORS + ["RF"]]], axis=1, join="inner").dropna()
    y = df["r"] - df["RF"]
    x = sm.add_constant(df[FACTORS])
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    return {
        "n": int(fit.nobs),
        "alpha_daily": round(float(fit.params["const"]), 8),
        "alpha_annualized": round(float(fit.params["const"] * TRADING_DAYS), 6),
        "alpha_t": round(float(fit.tvalues["const"]), 4),
        "alpha_p": round(float(fit.pvalues["const"]), 4),
        "betas": {f: round(float(fit.params[f]), 4) for f in FACTORS},
        "t_stats": {f: round(float(fit.tvalues[f]), 4) for f in FACTORS},
        "r2": round(float(fit.rsquared), 4),
        "r2_adj": round(float(fit.rsquared_adj), 4),
        "residual_vol_annualized": round(float(np.std(fit.resid, ddof=x.shape[1]) * np.sqrt(TRADING_DAYS)), 6),
        "hac_lags": hac_lags,
    }
