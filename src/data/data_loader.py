from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from datetime import datetime

class BaseDataLoader(ABC):
    
    def __init__(self, config: 'DataConfig'):
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