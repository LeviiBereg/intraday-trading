from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List

class BasePreprocessor(ABC):
    
    @abstractmethod
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean raw price data (handle missing values, outliers, etc.).
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Cleaned DataFrame
        """
        pass
    
    @abstractmethod
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical and statistical features.
        
        Args:
            data: Cleaned OHLCV data
            
        Returns:
            DataFrame with additional feature columns
        """
        pass
    
    @abstractmethod
    def normalize_features(self, data: pd.DataFrame, 
                         features: List[str]) -> pd.DataFrame:
        """Normalize feature values for ML models.
        
        Args:
            data: DataFrame with features
            features: List of feature column names to normalize
            
        Returns:
            DataFrame with normalized features
        """
        pass