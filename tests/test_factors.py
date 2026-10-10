"""Factor regression recovers known betas and alpha on synthetic data."""

import numpy as np
import pandas as pd

from prisk.factors import FACTORS, regress


def test_regression_recovers_betas_and_alpha():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2015-01-01", periods=2000)
    f = pd.DataFrame(rng.normal(0, 0.01, size=(len(idx), len(FACTORS))), index=idx, columns=FACTORS)
    f["RF"] = 0.0001
    true = {"Mkt-RF": 0.9, "SMB": 0.2, "HML": -0.3, "RMW": 0.1, "CMA": 0.0, "Mom": 0.15}
    alpha_daily = 0.0002
    r = f["RF"] + alpha_daily + sum(true[k] * f[k] for k in FACTORS) + rng.normal(0, 0.002, len(idx))
    out = regress(r, f)
    assert out["n"] == len(idx)
    for k, b in true.items():
        assert abs(out["betas"][k] - b) < 0.03
    assert abs(out["alpha_daily"] - alpha_daily) < 0.0001
    assert out["alpha_annualized"] == round(out["alpha_daily"] * 252, 6)
    assert out["r2"] > 0.9 and abs(out["t_stats"]["Mkt-RF"]) > 50
