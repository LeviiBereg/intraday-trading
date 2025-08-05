from abc import ABC, abstractmethod
import pandas as pd

class BaseTrendDetector(ABC):
    
    @abstractmethod
    def detect_trend_direction(self, data: pd.DataFrame) -> pd.Series:
        """Detect trend direction (up, down, sideways).
        
        Args:
            data: Price data
            
        Returns:
            Series with trend classifications
        """
        pass
    
    @abstractmethod
    def calculate_trend_strength(self, data: pd.DataFrame) -> pd.Series:
        """Calculate trend strength metric.
        
        Args:
            data: Price data
            
        Returns:
            Series with trend strength values (0-1)
        """
        pass
    
    @abstractmethod
    def is_ranging_market(self, data: pd.DataFrame) -> bool:
        """Determine if market is currently ranging.
        
        Args:
            data: Recent price data
            
        Returns:
            True if market is in ranging regime
        """
        pass