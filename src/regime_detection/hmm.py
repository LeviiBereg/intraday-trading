from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Any
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

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


class GaussianMixtureRegimeDetector(BaseHMMRegimeDetector):
    
    def __init__(self, n_states: int = 3, **kwargs):
        super().__init__(n_states, **kwargs)
        self.scaler = StandardScaler()
        self.model = GaussianMixture(
            n_components=n_states,
            covariance_type='full',
            max_iter=100,
            random_state=kwargs.get('random_state', 13)
        )
        self.fitted = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        # Price-based features
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        data['volatility'] = data['returns'].rolling(window=20).std()
        data['realized_vol'] = np.sqrt(data['returns'].rolling(window=20).var() * 252)
        
        data['volume_ma'] = data['volume'].rolling(window=20).mean()
        data['volume_ratio'] = data['volume'] / data['volume_ma']
        
        data['hl_ratio'] = (data['high'] - data['low']) / data['close']
        data['oc_ratio'] = (data['open'] - data['close']) / data['close']
        
        data['rsi'] = self._calculate_rsi(data['close'])
        data['momentum'] = data['close'] / data['close'].shift(10) - 1
        
        feature_cols = [
            'returns', 'log_returns', 'volatility', 'realized_vol',
            'volume_ratio', 'hl_ratio', 'oc_ratio', 'rsi', 'momentum'
        ]
        
        features = data[feature_cols].dropna().values
        return features
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def fit(self, features: np.ndarray) -> None:
        features_scaled = self.scaler.fit_transform(features)
        
        self.model.fit(features_scaled)
        self.fitted = True
        
    def predict_regime(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def get_regime_characteristics(self) -> dict:
        if not self.fitted:
            raise ValueError("Model must be fitted before analyzing regimes")
            
        characteristics = {}
        means = self.scaler.inverse_transform(self.model.means_)
        
        feature_names = [
            'returns', 'log_returns', 'volatility', 'realized_vol',
            'volume_ratio', 'hl_ratio', 'oc_ratio', 'rsi', 'momentum'
        ]
        
        for i in range(self.n_states):
            regime_mean = means[i]
            characteristics[f'regime_{i}'] = {
                'weight': self.model.weights_[i],
                'features': dict(zip(feature_names, regime_mean))
            }
            
        return characteristics
    
    def classify_regimes(self) -> dict:
        if not self.fitted:
            raise ValueError("Model must be fitted before classification")
            
        characteristics = self.get_regime_characteristics()
        regime_types = {}
        
        for regime_name, props in characteristics.items():
            features = props['features']
            
            volatility = features['volatility']
            returns = features['returns']
            
            if volatility > 0.02 and abs(returns) > 0.01:
                regime_type = 'high_volatility'
            elif volatility < 0.01 and abs(returns) < 0.005:
                regime_type = 'low_volatility'  
            elif returns > 0.005:
                regime_type = 'trending_up'
            elif returns < -0.005:
                regime_type = 'trending_down'
            else:
                regime_type = 'ranging'
                
            regime_types[regime_name] = regime_type
            
        return regime_types


class SimpleHMMRegimeDetector(BaseHMMRegimeDetector):
    
    def __init__(self, n_states: int = 3, **kwargs):
        super().__init__(n_states, **kwargs)
        self.transition_matrix = None
        self.emission_means: np.ndarray[Any] = None
        self.emission_stds: np.ndarray[Any] = None
        self.initial_probs: np.ndarray[Any] = None
        self.scaler = StandardScaler()
        self.fitted = False
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        data = data.copy()
        data['returns'] = data['close'].pct_change()
        data['volatility'] = data['returns'].rolling(window=10).std()
        
        features = data[['returns', 'volatility']].dropna().values
        return features
    
    def fit(self, features: np.ndarray) -> None:
        features_scaled = self.scaler.fit_transform(features)
        n_obs, n_features = features_scaled.shape
        
        np.random.seed(self.params.get('random_state', 13))
        self.transition_matrix = np.random.dirichlet([1] * self.n_states, self.n_states)
        self.initial_probs = np.random.dirichlet([1] * self.n_states)
        
        self.emission_means = np.random.randn(self.n_states, n_features)
        self.emission_stds = np.ones((self.n_states, n_features))
        
        for iteration in range(50):
            responsibilities = self._compute_responsibilities(features_scaled)
            self._update_parameters(features_scaled, responsibilities)
            
        self.fitted = True
    
    def _compute_responsibilities(self, features: np.ndarray) -> np.ndarray:
        n_obs = len(features)
        responsibilities = np.zeros((n_obs, self.n_states))
        
        for t in range(n_obs):
            for s in range(self.n_states):
                diff = features[t] - self.emission_means[s]
                likelihood = np.exp(-0.5 * np.sum((diff / (self.emission_stds[s] + 1e-10)) ** 2))
                responsibilities[t, s] = likelihood
        
        # Normalize
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        return responsibilities
    
    def _update_parameters(self, features: np.ndarray, responsibilities: np.ndarray) -> None:
        for s in range(self.n_states):
            weights = responsibilities[:, s] + 1e-10
            self.emission_means[s] = np.average(features, axis=0, weights=weights)
            
            diff = features - self.emission_means[s]
            self.emission_stds[s] = np.sqrt(np.average(diff**2, axis=0, weights=weights) + 1e-10)
    
    def predict_regime(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        features_scaled = self.scaler.transform(features)
        responsibilities = self._compute_responsibilities(features_scaled)
        return np.argmax(responsibilities, axis=1)
    
    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
            
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        features_scaled = self.scaler.transform(features)
        return self._compute_responsibilities(features_scaled)