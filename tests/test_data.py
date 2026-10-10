from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from prisk.data import TICKERS, BENCHMARKS, DATA_DIR, align, download_factors, download_prices, to_returns


class TestToReturns:
    def test_simple_returns(self):
        # Hand-made price frame
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [50.0, 51.0, 49.0]},
            index=pd.date_range("2020-01-01", periods=3),
        )

        returns = to_returns(prices)

        # First row dropped, pct_change gives (101-100)/100=0.01, (102-101)/101≈0.0099
        assert len(returns) == 2
        assert returns.index[0] == pd.Timestamp("2020-01-02")
        assert abs(returns.loc[pd.Timestamp("2020-01-02"), "A"] - 0.01) < 1e-6
        assert abs(returns.loc[pd.Timestamp("2020-01-02"), "B"] - (51 - 50) / 50) < 1e-6

    def test_drops_nan_rows(self):
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [50.0, 51.0, float("nan")]},
            index=pd.date_range("2020-01-01", periods=3),
        )

        returns = to_returns(prices)

        # Drops rows with any NaN
        assert len(returns) == 1
        assert returns.index[0] == pd.Timestamp("2020-01-02")


class TestAlign:
    def test_inner_join_on_dates(self):
        returns = pd.DataFrame(
            {"A": [0.01, 0.02, 0.03]},
            index=pd.date_range("2020-01-02", periods=3),
        )
        factors = pd.DataFrame(
            {"Mkt-RF": [0.001, 0.002, 0.003, 0.004]},
            index=pd.date_range("2020-01-01", periods=4),
        )

        aligned_returns, aligned_factors = align(returns, factors)

        assert len(aligned_returns) == len(aligned_factors) == 3
        assert aligned_returns.index[0] == pd.Timestamp("2020-01-02")
        assert aligned_factors.index[0] == pd.Timestamp("2020-01-02")
        assert aligned_returns.index.equals(aligned_factors.index)


class TestDownloadPricesCached:
    def test_cached_file_not_refreshed(self, tmp_path):
        # Create a fake cached file
        cache_file = tmp_path / "prices.csv"
        cached_df = pd.DataFrame(
            {"A": [100.0, 101.0]},
            index=pd.date_range("2020-01-01", periods=2),
        )
        cached_df.index.name = "Date"
        cached_df.to_csv(cache_file)

        # Monkeypatch yfinance.download to raise if called
        import prisk.data
        original_download = prisk.data.yfinance.download

        def mock_download(*args, **kwargs):
            raise RuntimeError("yfinance.download should not be called for cached file")

        prisk.data.yfinance.download = mock_download

        try:
            result = download_prices(
                ["A"],
                start="2020-01-01",
                cache=cache_file,
                refresh=False,
            )
            assert result.equals(cached_df)
        finally:
            prisk.data.yfinance.download = original_download


class TestDownloadFactorsCached:
    def test_cached_file_not_refreshed(self, tmp_path):
        # Create a fake cached file
        cache_file = tmp_path / "factors.csv"
        cached_df = pd.DataFrame(
            {
                "Mkt-RF": [0.001, 0.002],
                "SMB": [0.0005, 0.0006],
                "HML": [-0.0001, 0.0002],
                "RMW": [0.00001, 0.00002],
                "CMA": [0.00003, 0.00004],
                "RF": [0.00001, 0.00001],
                "Mom": [0.0, 0.0],
            },
            index=pd.date_range("2020-01-01", periods=2),
        )
        cached_df.index.name = "Date"
        cached_df.to_csv(cache_file)

        import prisk.data
        original_datareader = prisk.data.pandas_datareader.data.DataReader

        def mock_datareader(*args, **kwargs):
            raise RuntimeError("DataReader should not be called for cached file")

        prisk.data.pandas_datareader.data.DataReader = mock_datareader

        try:
            result = download_factors(
                start="2020-01-01",
                cache=cache_file,
                refresh=False,
            )
            assert result.equals(cached_df)
        finally:
            prisk.data.pandas_datareader.data.DataReader = original_datareader


class TestDataIntegrity:
    def test_committed_prices_and_factors(self):
        prices_file = DATA_DIR / "prices.csv"
        factors_file = DATA_DIR / "factors.csv"

        if not prices_file.exists() or not factors_file.exists():
            pytest.skip("Data files not present")

        # Read prices
        prices = pd.read_csv(prices_file, index_col=0, parse_dates=True)

        # 32 tickers: 30 + SPY + AGG
        assert prices.shape[1] == 32, f"Expected 32 columns, got {prices.shape[1]}"

        # No NaNs
        assert prices.isna().sum().sum() == 0, "Prices should have no NaNs"

        # Read factors
        factors = pd.read_csv(factors_file, index_col=0, parse_dates=True)

        # 7 columns: Mkt-RF, SMB, HML, RMW, CMA, RF, Mom
        assert factors.shape[1] == 7, f"Expected 7 columns, got {factors.shape[1]}"

        # Factor values are decimals
        assert abs(factors["Mkt-RF"].max()) < 0.2, "Mkt-RF should be decimal (< 0.2)"
        assert abs(factors["Mkt-RF"].min()) < 0.2, "Mkt-RF should be decimal (< 0.2)"
        assert abs(factors["RF"].max()) < 0.01, "RF should be very small decimal"
