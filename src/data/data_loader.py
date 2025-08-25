from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import yfinance as yf
import os
from pathlib import Path
from config.settings import DataConfig

class BaseDataLoader(ABC):
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    @abstractmethod
    def load_data(self, symbol: str, start_date: datetime, 
                 end_date: datetime, timeframe: str) -> pd.DataFrame:
        """Load historical price data for given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            start_date: Start date for data
            end_date: End date for data  
            timeframe: Timeframe ('1h', '4h', '1d')
            
        Returns:
            DataFrame with OHLCV data
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, symbol: str, timeframe: str, 
                       periods: int = 100) -> pd.DataFrame:
        """Get latest N periods of data for symbol.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            periods: Number of periods to retrieve
            
        Returns:
            DataFrame with recent OHLCV data
        """
        pass
    
    @abstractmethod
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality and completeness.
        
        Args:
            data: Price data DataFrame
            
        Returns:
            True if data is valid
        """
        pass


class YahooDataLoader(BaseDataLoader):
    
    def __init__(self, config: DataConfig):
        super().__init__(config)
        self._interval_map = {
            '1m': '1m',
            '5m': '5m', 
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        
        if self.config.cache_enabled:
            Path(self.config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self, symbol: str, start_date: datetime, 
                 end_date: datetime, timeframe: str) -> pd.DataFrame:
        
        cache_key = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_file = None
        
        if self.config.cache_enabled:
            cache_file = Path(self.config.cache_dir) / f"{cache_key}.parquet"
            if cache_file.exists():
                return pd.read_parquet(cache_file)
        
        interval = self._interval_map.get(timeframe, '1d')
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date, 
            interval=interval,
            auto_adjust=True,
            prepost=False
        )
        
        if data.empty:
            raise ValueError(f"No data found for {symbol} from {start_date} to {end_date}")
        
        data.columns = data.columns.str.lower()
        data = data.rename(columns={
            'adj close': 'adj_close'
        })
        
        if not self.validate_data(data):
            raise ValueError(f"Data validation failed for {symbol}")
        
        if self.config.cache_enabled and cache_file:
            data.to_parquet(cache_file)
        
        return data
    
    def get_latest_data(self, symbol: str, timeframe: str, 
                       periods: int = 100) -> pd.DataFrame:
        
        interval = self._interval_map.get(timeframe, '1d')
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            period=f"{periods}d" if interval == '1d' else "60d",
            interval=interval,
            auto_adjust=True,
            prepost=False
        )
        
        if data.empty:
            raise ValueError(f"No recent data available for {symbol}")
        
        data.columns = data.columns.str.lower()
        data = data.rename(columns={
            'adj close': 'adj_close'
        })
        
        data = data.tail(periods)
        
        if not self.validate_data(data):
            raise ValueError(f"Data validation failed for {symbol}")
        
        return data
    
    def validate_data(self, data: pd.DataFrame) -> bool:        
        if data.empty:
            return False
        
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if not all(col in data.columns for col in required_columns):
            return False
        
        ohlc_cols = ['open', 'high', 'low', 'close']
        if data[ohlc_cols].isnull().any().any():
            return False
        
        invalid_rows = (
            (data['high'] < data['low']) |
            (data['high'] < data['open']) |
            (data['high'] < data['close']) |
            (data['low'] > data['open']) |
            (data['low'] > data['close'])
        )
        
        if invalid_rows.any():
            return False
        
        if (data[ohlc_cols] <= 0).any().any():
            return False
        
        return True