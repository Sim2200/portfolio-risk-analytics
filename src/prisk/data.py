"""Daily prices and Fama-French factors, downloaded once and cached as CSV under data/.

Prices: adjusted closes (splits and dividends) from Yahoo Finance via yfinance, with Stooq's CSV
endpoint as the fallback. Factors: Ken French's daily 5 factors and momentum (pandas-datareader).

    python -m prisk.data --start 2010-01-01            # refresh data/prices.csv and data/factors.csv

Fixed ticker list: 30 large US stocks that were liquid over the whole period plus SPY and AGG.
Choosing today's large names introduces survivorship bias (see README).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "UNH", "HD",
    "KO", "PEP", "CVX", "MRK", "PFE", "WMT", "DIS", "CSCO", "INTC", "IBM",
    "MCD", "BA", "CAT", "MMM", "GS", "WFC", "BAC", "T", "VZ", "ORCL",
]
BENCHMARKS = ["SPY", "AGG"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def download_prices(tickers: list[str], start: str, end: str | None = None, cache: Path = DATA_DIR / "prices.csv",
                    refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, DatetimeIndex ascending. Loads the cache when
    present unless refresh=True; otherwise downloads, writes the cache and returns it."""
    raise NotImplementedError


def download_factors(start: str, cache: Path = DATA_DIR / "factors.csv", refresh: bool = False) -> pd.DataFrame:
    """Daily Fama-French 5 factors + momentum + RF as decimals (not percent): columns
    Mkt-RF, SMB, HML, RMW, CMA, RF, Mom. DatetimeIndex ascending."""
    raise NotImplementedError


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, first row dropped."""
    raise NotImplementedError


def align(returns: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join on dates so both frames cover exactly the same trading days."""
    raise NotImplementedError
