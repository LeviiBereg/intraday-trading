from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple

class BaseHMMRegimeDetector(ABC):
    
    def __init__(self, n_states: int = 3, **kwargs):
        self.n_states = n_states
        self.model = None
        self.params = kwargs
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for HMM model.
        
        Args:
            data: OHLCV price data
            
        Returns:
            Feature array for model training
        """
        pass
    
    @abstractmethod
    def fit(self, features: np.ndarray) -> None:
        """Train HMM model on historical data.
        
        Args:
            features: Prepared feature array
        """
        pass
    
    @abstractmethod
    def predict_regime(self, features: np.ndarray) -> np.ndarray:
        """Predict current market regime.
        
        Args:
            features: Current feature values
            
        Returns:
            Array of regime predictions
        """
        pass
    
    @abstractmethod
    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get probabilities for each regime.
        
        Args:
            features: Current feature values
            
        Returns:
            Array of regime probabilities
        """
        pass