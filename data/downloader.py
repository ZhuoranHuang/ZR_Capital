"""Multi-source market data download abstraction.

Strategies and notebooks must read through this layer; no module outside
`data/` may import a vendor SDK directly.

Concrete sources today: `YFinanceSource`. Longbridge and Polygon adapters
follow the same `DataSource` contract.
"""
import os
import time
from abc import ABC, abstractmethod

import pandas as pd
import yfinance as yf

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')


class DataSource(ABC):
    source_name: str = 'base'

    @abstractmethod
    def download(self, ticker, period, interval='1d') -> pd.DataFrame:
        """Fetch OHLCV from the vendor. Implemented per source."""

    def get(self, ticker, period, interval='1d', use_cache=True, ttl_hours=24) -> pd.DataFrame:
        cache_path = self._cache_path(ticker, period, interval)

        if use_cache and os.path.exists(cache_path):
            age_hours = (time.time() - os.path.getmtime(cache_path)) / 3600
            if age_hours < ttl_hours:
                try:
                    return pd.read_parquet(cache_path)
                except Exception as e:
                    print(f"[downloader] 缓存读取失败 ({e})，重新下载")

        data = self.download(ticker, period, interval)

        try:
            os.makedirs(CACHE_DIR, exist_ok=True)
            data.to_parquet(cache_path)
        except Exception as e:
            print(f"[downloader] 缓存写入失败 ({e})，跳过缓存")

        return data

    def _cache_path(self, ticker, period, interval):
        filename = f"{self.source_name}_{ticker}_{period}_{interval}.parquet"
        return os.path.join(CACHE_DIR, filename)


class YFinanceSource(DataSource):
    source_name = 'yfinance'

    def download(self, ticker, period, interval='1d'):
        return yf.download(ticker, period=period, interval=interval)


_SOURCES = {'yfinance': YFinanceSource}


def get_source(name: str) -> DataSource:
    if name not in _SOURCES:
        raise ValueError(f"Unknown data source: {name}. Available: {list(_SOURCES)}")
    return _SOURCES[name]()
