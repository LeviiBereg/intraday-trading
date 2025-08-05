from abc import ABC, abstractmethod
import pandas as pd
from typing import Tuple, List

class BaseRangeDetector(ABC):
    
    @abstractmethod
    def detect_pivot_points(self, data: pd.DataFrame, 
                          window: int = 20) -> pd.DataFrame:
        """Detect pivot high and low points.
        
        Args:
            data: OHLCV data
            window: Window for pivot detection
            
        Returns:
            DataFrame with pivot columns
        """
        pass
    
    @abstractmethod
    def identify_support_resistance(self, data: pd.DataFrame) -> Tuple[float, float]:
        """Identify current support and resistance levels.
        
        Args:
            data: Price data with pivot points
            
        Returns:
            Tuple of (support_level, resistance_level)
        """
        pass
    
    @abstractmethod
    def validate_range_quality(self, data: pd.DataFrame, 
                             support: float, resistance: float) -> bool:
        """Validate if detected range is tradeable.
        
        Args:
            data: Price data
            support: Support level
            resistance: Resistance level
            
        Returns:
            True if range is valid for trading
        """
        pass