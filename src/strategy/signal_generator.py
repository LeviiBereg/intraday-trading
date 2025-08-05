from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, List

class BaseSignalGenerator(ABC):
    
    @abstractmethod
    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals.
        
        Args:
            data: Market data with indicators
            
        Returns:
            Series with entry signals
        """
        pass
    
    @abstractmethod
    def generate_exit_signals(self, data: pd.DataFrame, 
                            positions: List[Any]) -> Dict[str, bool]:
        """Generate exit signals for active positions.
        
        Args:
            data: Current market data
            positions: List of active positions
            
        Returns:
            Dict mapping position IDs to exit decisions
        """
        pass
    
    @abstractmethod
    def filter_signals(self, signals: pd.Series, 
                      filters: Dict[str, Any]) -> pd.Series:
        """Apply filters to trading signals.
        
        Args:
            signals: Raw trading signals
            filters: Filter criteria
            
        Returns:
            Filtered signals
        """
        pass