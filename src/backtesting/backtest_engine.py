from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any
from src.strategy.base_strategy import BaseStrategy

class BaseBacktestEngine(ABC):
    
    def __init__(self, strategy: BaseStrategy, 
                 initial_capital: float = 100000):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results: Dict[str, Any] = {}
    
    @abstractmethod
    def load_test_data(self, symbol: str, start_date: str, 
                      end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Historical price data
        """
        pass
    
    @abstractmethod
    def run_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Execute backtest on historical data.
        
        Args:
            data: Historical market data
            
        Returns:
            Dictionary with backtest results
        """
        pass
    
    @abstractmethod
    def calculate_performance_metrics(self, trades: List['Position']) -> Dict[str, float]:
        """Calculate performance metrics from completed trades.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Dictionary with performance metrics
        """
        pass
    
    @abstractmethod
    def generate_equity_curve(self, trades: List['Position']) -> pd.Series:
        """Generate portfolio equity curve.
        
        Args:
            trades: List of completed trades
            
        Returns:
            Series with equity values over time
        """
        pass