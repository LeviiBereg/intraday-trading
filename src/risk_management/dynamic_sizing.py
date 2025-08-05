from abc import ABC, abstractmethod
import pandas as pd

class BaseDynamicSizer(ABC):
    
    @abstractmethod
    def adjust_size_for_regime(self, base_size: float, 
                              regime: str, volatility: float) -> float:
        """Adjust position size based on market regime.
        
        Args:
            base_size: Base position size
            regime: Current market regime
            volatility: Current volatility level
            
        Returns:
            Adjusted position size
        """
        pass
    
    @abstractmethod
    def calculate_volatility_adjustment(self, current_vol: float, 
                                      target_vol: float = 0.15) -> float:
        """Calculate volatility-based size adjustment.
        
        Args:
            current_vol: Current market volatility
            target_vol: Target volatility level
            
        Returns:
            Size adjustment multiplier
        """
        pass
