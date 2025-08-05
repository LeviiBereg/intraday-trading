from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple, Any

class BaseWalkForwardOptimizer(ABC):
    
    @abstractmethod
    def setup_optimization_windows(self, data: pd.DataFrame, 
                                  train_window: int = 252, 
                                  test_window: int = 63) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Setup training and testing windows for walk-forward analysis.
        
        Args:
            data: Historical data
            train_window: Training window size in periods
            test_window: Testing window size in periods
            
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        pass
    
    @abstractmethod
    def optimize_parameters(self, train_data: pd.DataFrame, 
                          parameter_space: Dict[str, List]) -> Dict[str, Any]:
        """Optimize strategy parameters on training data.
        
        Args:
            train_data: Training dataset
            parameter_space: Dictionary defining parameter ranges
            
        Returns:
            Optimized parameters
        """
        pass
    
    @abstractmethod
    def validate_on_test_data(self, test_data: pd.DataFrame, 
                            optimized_params: Dict[str, Any]) -> Dict[str, float]:
        """Validate optimized parameters on out-of-sample data.
        
        Args:
            test_data: Test dataset
            optimized_params: Parameters from optimization
            
        Returns:
            Performance metrics on test data
        """
        pass