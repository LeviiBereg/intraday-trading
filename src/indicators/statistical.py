from abc import ABC, abstractmethod
import pandas as pd

class BaseStatisticalIndicator(ABC):
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values.
        
        Args:
            data: time series data
            
        Returns:
            DataFrame with indicator columns added
        """
        pass
    
    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from indicator.
        
        Args:
            data: DataFrame with indicator values
            
        Returns:
            Series with signal values
        """
        pass