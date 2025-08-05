from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple

class BaseMLFilter(ABC):
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for ML model.
        
        Args:
            data: Market data
            
        Returns:
            Feature array
        """
        pass
    
    @abstractmethod
    def create_labels(self, data: pd.DataFrame, 
                     target_horizon: int = 5) -> np.ndarray:
        """Create target labels for training.
        
        Args:
            data: Market data
            target_horizon: Forward-looking periods for labels
            
        Returns:
            Label array
        """
        pass
    
    @abstractmethod
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train ML model.
        
        Args:
            features: Training features
            labels: Training labels
        """
        pass
    
    @abstractmethod
    def predict_probability(self, features: np.ndarray) -> np.ndarray:
        """Predict signal probabilities.
        
        Args:
            features: Current features
            
        Returns:
            Probability predictions
        """
        pass
    
    @abstractmethod
    def filter_signals(self, signals: pd.Series, 
                      probabilities: np.ndarray, 
                      threshold: float = 0.6) -> pd.Series:
        """Filter signals based on ML probabilities.
        
        Args:
            signals: Raw trading signals
            probabilities: ML probability predictions
            threshold: Probability threshold for filtering
            
        Returns:
            Filtered signals
        """
        pass