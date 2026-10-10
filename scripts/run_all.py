"""Run every experiment and write results/*.json. Everything the README quotes comes from here.

    python scripts/run_all.py [--start 2010-01-01] [--fit-months 36] [--cost-bps 10] [--cap 0.10]

Outputs
  results/data.json        what was used: tickers, date range, days, sources, factor coverage
  results/backtest.json    out-of-sample metrics per strategy and benchmark, yearly returns, turnover,
                           costs, the covariance-estimator comparison by sub-period
  results/risk.json        1-day 99% VaR / CVaR (historical, parametric, Monte Carlo) on the out-of-sample
                           daily returns, the rolling VaR backtest with Kupiec's test, stress windows
  results/factors.json     Fama-French 5 + momentum regressions per strategy
  results/figures/*.png    wealth curves, drawdowns, weights over time
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prisk import TRADING_DAYS  # noqa: E402
from prisk.backtest import annual_turnover, benchmark_60_40, benchmark_spy, make_strategies, run_strategy, yearly_returns  # noqa: E402
from prisk.data import BENCHMARKS, TICKERS, align, download_factors, download_prices, to_returns  # noqa: E402
from prisk.factors import regress  # noqa: E402
from prisk.metrics import drawdown, max_drawdown, summary  # noqa: E402
from prisk.risk import (STRESS_WINDOWS, cvar_historical, cvar_parametric, stress_test, var_backtest,  # noqa: E402
                        var_cvar_monte_carlo, var_historical, var_parametric)

RES = ROOT / "results"
FIG = RES / "figures"


def dump(name: str, obj: dict) -> None:
    (RES / name).write_text(json.dumps(obj, indent=2, default=float) + "\n")
    print("wrote", RES / name, flush=True)


def sub_periods(r: pd.Series) -> dict[str, pd.Series]:
    """Calendar blocks for the estimator comparison, so a winner over the whole sample can be shown
    to lose in some periods when that is the case."""
    years = sorted(set(r.index.year))
    blocks = {}
    for y0 in range(years[0], years[-1] + 1, 3):
        block = r[(r.index.year >= y0) & (r.index.year < y0 + 3)]
        if len(block) > 200:
            blocks[f"{y0}-{min(y0 + 2, years[-1])}"] = block
    return blocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--fit-months", type=int, default=36)
    ap.add_argument("--cost-bps", type=float, default=10.0)
    ap.add_argument("--cap", type=float, default=0.10)
    ap.add_argument("--var-window", type=int, default=250)
    ap.add_argument("--mc-sims", type=int, default=100_000)
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    t0 = time.time()

    # ---- data ------------------------------------------------------------------------------
    prices = download_prices(TICKERS + BENCHMARKS, a.start)
    factors = download_factors(a.start)
    returns, factors = align(to_returns(prices), factors)
    assets = returns[TICKERS]
    rf_daily = factors["RF"]
    meta = json.loads((ROOT / "data" / "prices.meta.json").read_text()) if (ROOT / "data" / "prices.meta.json").exists() else {}
    dump("data.json", {
        "tickers": TICKERS, "benchmarks": BENCHMARKS, "n_assets": len(TICKERS),
        "first_date": str(returns.index.min().date()), "last_date": str(returns.index.max().date()),
        "trading_days": int(len(returns)), "price_sources": meta.get("sources", {}), "downloaded_at": meta.get("downloaded_at"),
        "factors": {"columns": list(factors.columns), "last_date": str(factors.index.max().date()),
                    "source": "Ken French data library (daily 5 factors 2x3, momentum), via pandas-datareader"},
        "price_source": "Yahoo Finance adjusted closes via yfinance (Stooq CSV fallback), cached in data/prices.csv",
        "note": "Fixed list of 30 large US stocks chosen today: survivorship bias (see README).",
    })

    # ---- backtests -------------------------------------------------------------------------
    strategies = make_strategies(a.cap)
    results = {}
    for name, strat in strategies.items():
        t1 = time.time()
        results[name] = run_strategy(assets, strat, rf_daily, a.fit_months, a.cost_bps, name)
        print(f"{name:28s} {len(results[name]['returns'])} days in {time.time() - t1:.0f} s", flush=True)
    oos_index = results["equal_weight"]["returns"].index
    spy = benchmark_spy(returns, oos_index)
    b6040 = benchmark_60_40(returns, oos_index, a.cost_bps)
    series = {name: r["returns"] for name, r in results.items()}
    series["spy"] = spy
    series["60_40"] = b6040["returns"].reindex(oos_index)
    rf_oos = rf_daily.reindex(oos_index).fillna(0.0)

    bt = {"config": {"fit_months": a.fit_months, "cost_bps": a.cost_bps, "cap": a.cap, "rebalance": "monthly (last trading day)",
                     "expected_returns": "trailing sample mean", "risk_free": "Ken French daily RF",
                     "oos_start": str(oos_index.min().date()), "oos_end": str(oos_index.max().date()), "oos_days": int(len(oos_index))},
          "strategies": {}, "covariance_comparison": {}}
    for name, r in series.items():
        s = summary(r, rf_oos)
        if name in results:
            s["turnover"] = round(annual_turnover(results[name]), 6)
            s["total_cost_drag_annualized"] = round(float((results[name]["gross_returns"] - results[name]["returns"]).mean() * TRADING_DAYS), 6)
            w = results[name]["weights"]
            s["weights"] = {"mean_max_weight": round(float(w.max(axis=1).mean()), 6),
                            "mean_effective_n": round(float((1 / (w ** 2).sum(axis=1)).mean()), 3),
                            "rebalances": int(len(w))}
        elif name == "60_40":
            s["turnover"] = round(annual_turnover(b6040), 6)
        s["yearly_returns"] = yearly_returns(r)
        s["worst_year"] = min(s["yearly_returns"], key=s["yearly_returns"].get)
        bt["strategies"][name] = s

    # Ledoit-Wolf vs sample, whole sample and by 3-year block, for each optimiser family
    for family in ("min_variance", "max_sharpe", "risk_parity"):
        a_s, b_s = series[f"{family}_sample"], series[f"{family}_ledoit_wolf"]
        comp = {"full_sample": {"sample": summary(a_s, rf_oos), "ledoit_wolf": summary(b_s, rf_oos)}, "by_period": {}}
        for label, block in sub_periods(a_s).items():
            sb = b_s.reindex(block.index)
            comp["by_period"][label] = {"sample_sharpe": round(summary(block, rf_oos.reindex(block.index))["sharpe"], 4),
                                        "ledoit_wolf_sharpe": round(summary(sb, rf_oos.reindex(sb.index))["sharpe"], 4),
                                        "sample_vol": round(summary(block, 0.0)["ann_vol"], 4),
                                        "ledoit_wolf_vol": round(summary(sb, 0.0)["ann_vol"], 4)}
        wins = sum(v["ledoit_wolf_sharpe"] > v["sample_sharpe"] for v in comp["by_period"].values())
        comp["ledoit_wolf_wins_sharpe_in_periods"] = f"{wins} of {len(comp['by_period'])}"
        bt["covariance_comparison"][family] = comp
    dump("backtest.json", bt)

    # ---- risk ------------------------------------------------------------------------------
    risk = {"alpha": 0.99, "horizon_days": 1, "var_window": a.var_window, "mc_sims": a.mc_sims, "strategies": {}}
    for name, r in series.items():
        entry = {"historical": {"var": round(var_historical(r), 6), "cvar": round(cvar_historical(r), 6)},
                 "parametric_normal": {"var": round(var_parametric(r), 6), "cvar": round(cvar_parametric(r), 6)}}
        if name in results:
            # Monte Carlo on the asset-level covariance with the strategy's mean weights
            w_mean = results[name]["weights"].mean().to_numpy()
            w_mean = w_mean / w_mean.sum()
            mc_var, mc_cvar = var_cvar_monte_carlo(assets.reindex(oos_index), w_mean, 0.99, a.mc_sims, 0)
            entry["monte_carlo_normal"] = {"var": round(mc_var, 6), "cvar": round(mc_cvar, 6), "weights": "mean target weights over the OOS period"}
        entry["backtest"] = {m: var_backtest(r, a.var_window, 0.99, m) for m in ("historical", "parametric")}
        entry["stress"] = stress_test(r, STRESS_WINDOWS)
        risk["strategies"][name] = entry
        print(f"risk {name:24s} hist VaR {entry['historical']['var']:.4f}  breaches hist {entry['backtest']['historical']['n_breaches']}"
              f"/{entry['backtest']['historical']['expected_breaches']:.1f}", flush=True)
    risk["stress_windows"] = {k: {"start": v[0], "end": v[1]} for k, v in STRESS_WINDOWS.items()}
    dump("risk.json", risk)

    # ---- factors ---------------------------------------------------------------------------
    fac = {"model": "Fama-French 5 factors + momentum, daily, OLS with Newey-West (5 lags) standard errors",
           "sample": {"start": str(oos_index.min().date()), "end": str(min(oos_index.max(), factors.index.max()).date())},
           "strategies": {name: regress(r, factors) for name, r in series.items()}}
    dump("factors.json", fac)

    # ---- figures ---------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for name, r in series.items():
        ax[0].plot((1 + r).cumprod(), label=name, lw=1.2 if name in ("spy", "60_40") else 0.9)
        ax[1].plot(drawdown(r), lw=0.8)
    ax[0].set_yscale("log"); ax[0].set_title("Out-of-sample wealth (log scale), net of 10 bps costs"); ax[0].legend(fontsize=7, ncol=3)
    ax[1].set_title("Drawdown"); ax[1].set_ylim(-0.6, 0.02)
    fig.tight_layout(); fig.savefig(FIG / "wealth_drawdown.png", dpi=130); plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax_, name in zip(axes, ("min_variance_ledoit_wolf", "max_sharpe_ledoit_wolf", "risk_parity_ledoit_wolf")):
        results[name]["weights"].plot.area(ax=ax_, legend=False, lw=0); ax_.set_title(f"{name}: target weights at each rebalance"); ax_.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIG / "weights.png", dpi=130); plt.close(fig)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
