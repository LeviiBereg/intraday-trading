from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum

class SignalType(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0
    CLOSE = 2

class Position:
    
    def __init__(self, symbol: str, side: str, size: float, 
                 entry_price: float, entry_time: pd.Timestamp):
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entry_price = entry_price
        self.entry_time = entry_time
        self.exit_price: Optional[float] = None
        self.exit_time: Optional[pd.Timestamp] = None
        self.pnl: Optional[float] = None

class BaseStrategy(ABC):
    
    def __init__(self, config: 'StrategyConfig'):
        self.config = config
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on market data.
        
        Args:
            data: Market data with indicators
            
        Returns:
            Series with signal values
        """
        pass
    
    @abstractmethod
    def should_enter_trade(self, data: pd.DataFrame, 
                          signal: SignalType) -> bool:
        """Determine if conditions are met to enter trade.
        
        Args:
            data: Current market data
            signal: Generated signal
            
        Returns:
            True if should enter trade
        """
        pass
    
    @abstractmethod
    def should_exit_trade(self, position: Position, 
                         data: pd.DataFrame) -> bool:
        """Determine if position should be closed.
        
        Args:
            position: Current position
            data: Current market data
            
        Returns:
            True if should exit position
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: SignalType, 
                              data: pd.DataFrame) -> float:
        """Calculate optimal position size for trade.
        
        Args:
            signal: Trading signal
            data: Current market data
            
        Returns:
            Position size
        """
        pass