from abc import ABC, abstractmethod
import pandas as pd

class BaseVolatilityRegimeDetector(ABC):
    
    @abstractmethod
    def calculate_volatility_regimes(self, data: pd.DataFrame) -> pd.Series:
        """Classify volatility regimes (low, medium, high).
        
        Args:
            data: Price data
            
        Returns:
            Series with volatility regime classifications
        """
        pass
    
    @abstractmethod
    def detect_volatility_spikes(self, data: pd.DataFrame, 
                               threshold: float = 2.0) -> pd.Series:
        """Detect sudden volatility spikes.
        
        Args:
            data: Price data
            threshold: Spike detection threshold (std devs)
            
        Returns:
            Boolean series indicating volatility spikes
        """
        pass