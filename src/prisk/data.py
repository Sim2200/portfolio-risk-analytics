"""Daily prices and Fama-French factors, downloaded once and cached as CSV under data/.

Prices: adjusted closes (splits and dividends) from Yahoo Finance via yfinance, with Stooq's CSV
endpoint as the fallback. Factors: Ken French's daily 5 factors and momentum (pandas-datareader).

    python -m prisk.data --start 2010-01-01            # refresh data/prices.csv and data/factors.csv

Fixed ticker list: 30 large US stocks that were liquid over the whole period plus SPY and AGG.
Choosing today's large names introduces survivorship bias (see README).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import pandas_datareader
import requests
import yfinance

TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "JNJ", "XOM", "PG", "UNH", "HD",
    "KO", "PEP", "CVX", "MRK", "PFE", "WMT", "DIS", "CSCO", "INTC", "IBM",
    "MCD", "BA", "CAT", "MMM", "GS", "WFC", "BAC", "T", "VZ", "ORCL",
]
BENCHMARKS = ["SPY", "AGG"]
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

SOURCES = {}


def download_prices(tickers: list[str], start: str, end: str | None = None, cache: Path = DATA_DIR / "prices.csv",
                    refresh: bool = False) -> pd.DataFrame:
    """Adjusted daily closes, one column per ticker, DatetimeIndex ascending. Loads the cache when
    present unless refresh=True; otherwise downloads, writes the cache and returns it."""
    global SOURCES
    SOURCES.clear()

    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    # Download from yfinance
    data = yfinance.download(
        tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
    )

    # Handle single vs. multiple tickers
    if len(tickers) == 1:
        df = pd.DataFrame(data)
        df.columns = tickers
    else:
        df = data.get("Close", data)

    # Record yfinance as source
    for ticker in df.columns:
        SOURCES[ticker] = "yfinance"

    # Handle missing tickers with Stooq fallback
    for ticker in tickers:
        if ticker not in df.columns or df[ticker].isna().all():
            try:
                url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=d"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                stooq_df = pd.read_csv(
                    pd.io.common.StringIO(response.text),
                    parse_dates=["Date"], usecols=["Date", "Close"]
                )
                stooq_df = stooq_df.set_index("Date").sort_index()
                df[ticker] = stooq_df["Close"]
                SOURCES[ticker] = "stooq"
            except Exception:
                pass

    # Keep rows with date >= start
    start_dt = pd.to_datetime(start)
    df = df[df.index >= start_dt]

    # Sort ascending
    df = df.sort_index()

    # Drop columns entirely NaN
    df = df.dropna(axis=1, how="all")

    # Forward-fill at most 3 consecutive NaNs
    df = df.ffill(limit=3)

    # Drop remaining NaN rows
    df = df.dropna(how="any")

    # Write cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.index.name = "Date"
    df.to_csv(cache)

    # Write metadata
    meta_file = cache.with_suffix(".meta.json")
    meta = {
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "start": start,
        "end": end,
        "tickers": tickers,
        "sources": SOURCES,
        "rows": len(df),
        "first_date": df.index[0].isoformat(),
        "last_date": df.index[-1].isoformat(),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return df


def download_factors(start: str, cache: Path = DATA_DIR / "factors.csv", refresh: bool = False) -> pd.DataFrame:
    """Daily Fama-French 5 factors + momentum + RF as decimals (not percent): columns
    Mkt-RF, SMB, HML, RMW, CMA, RF, Mom. DatetimeIndex ascending."""

    if cache.exists() and not refresh:
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    # Download 5 factors
    factors_5 = pandas_datareader.data.DataReader(
        "F-F_Research_Data_5_Factors_2x3_daily", "famafrench", start
    )[0]

    # Download momentum
    momentum = pandas_datareader.data.DataReader(
        "F-F_Momentum_Factor_daily", "famafrench", start
    )[0]

    # Strip whitespace from column names
    factors_5.columns = factors_5.columns.str.strip()
    momentum.columns = momentum.columns.str.strip()

    # Rename momentum column
    momentum = momentum.rename(columns={"Mom ": "Mom"})

    # Join on date (inner)
    df = factors_5.join(momentum, how="inner")

    # Divide by 100 to convert from percent to decimal
    df = df / 100.0

    # Set index name
    df.index.name = "Date"

    # Sort ascending
    df = df.sort_index()

    # Convert Period index to DatetimeIndex
    if hasattr(df.index, 'to_timestamp'):
        df.index = df.index.to_timestamp()

    # Write cache
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache)

    # Write metadata
    meta_file = cache.with_suffix(".meta.json")
    meta = {
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "rows": len(df),
        "first_date": df.index[0].isoformat(),
        "last_date": df.index[-1].isoformat(),
    }
    with open(meta_file, "w") as f:
        json.dump(meta, f, indent=2)

    return df


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, first row dropped."""
    returns = prices.pct_change().dropna(how="all")
    return returns.dropna(how="any")


def align(returns: pd.DataFrame, factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inner-join on dates so both frames cover exactly the same trading days."""
    common_dates = returns.index.intersection(factors.index)
    return returns.loc[common_dates], factors.loc[common_dates]


def main():
    parser = argparse.ArgumentParser(description="Download prices and factors.")
    parser.add_argument("--start", default="2010-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh")

    args = parser.parse_args()

    # Download prices
    prices = download_prices(TICKERS + BENCHMARKS, args.start, args.end, refresh=args.refresh)
    print(f"Prices shape: {prices.shape}")
    print(f"Prices date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Sources: {SOURCES}")

    # Download factors
    factors = download_factors(args.start, refresh=args.refresh)
    print(f"Factors shape: {factors.shape}")
    print(f"Factors date range: {factors.index[0].date()} to {factors.index[-1].date()}")


if __name__ == "__main__":
    main()
