from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple

class BaseRiskMetrics(ABC):
    
    @abstractmethod
    def calculate_var(self, returns: pd.Series, 
                     confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk.
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            VaR value
        """
        pass
    
    @abstractmethod
    def calculate_expected_shortfall(self, returns: pd.Series, 
                                   confidence_level: float = 0.05) -> float:
        """Calculate Expected Shortfall (Conditional VaR).
        
        Args:
            returns: Historical returns
            confidence_level: Confidence level
            
        Returns:
            Expected Shortfall value
        """
        pass
    
    @abstractmethod
    def calculate_maximum_drawdown(self, equity_curve: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration.
        
        Args:
            equity_curve: Portfolio equity curve
            
        Returns:
            Tuple of (max_drawdown, duration_days)
        """
        pass
    
    @abstractmethod
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio.
        
        Args:
            returns: Portfolio returns
            risk_free_rate: Risk-free rate
            
        Returns:
            Sharpe ratio
        """
        pass