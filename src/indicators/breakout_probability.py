from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, Any
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class BaseBreakoutPredictor(ABC):

    @abstractmethod
    def prepare_features(self, data: pd.DataFrame,
                        support_resistance_data: pd.DataFrame = None) -> pd.DataFrame:
        """Prepare features for breakout prediction."""
        pass

    @abstractmethod
    def create_breakout_labels(self, data: pd.DataFrame,
                              lookforward_periods: int = 24) -> pd.DataFrame:
        """Create breakout labels for training."""
        pass

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        """Train the breakout prediction model."""
        pass

    @abstractmethod
    def predict_breakout_probability(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Predict breakout probabilities."""
        pass


class CatBoostBreakoutPredictor(BaseBreakoutPredictor):

    def __init__(self,
                 upward_model_params: Dict[str, Any] = None,
                 downward_model_params: Dict[str, Any] = None,
                 breakout_threshold: float = 0.02,
                 lookforward_periods: int = 24):

        self.breakout_threshold = breakout_threshold
        self.lookforward_periods = lookforward_periods

        default_params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': 42,
            'od_type': 'Iter',
            'od_wait': 50,
            'verbose': False
        }

        self.upward_model_params = upward_model_params or default_params
        self.downward_model_params = downward_model_params or default_params

        self.upward_model = CatBoostClassifier(**self.upward_model_params)
        self.downward_model = CatBoostClassifier(**self.downward_model_params)

        self.feature_columns = None
        self.is_fitted = False

    def prepare_features(self, data: pd.DataFrame,
                        support_resistance_data: pd.DataFrame = None) -> pd.DataFrame:

        df = data.copy()

        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Volatility features
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
            df[f'realized_vol_{window}'] = np.sqrt(df['returns'].rolling(window).var() * 252)

        # Volume features
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_momentum'] = df['volume'].pct_change(5)

        # Volume accumulation patterns
        df['money_flow'] = df['close'] * df['volume']
        df['money_flow_ratio'] = (df['money_flow'] / df['money_flow'].rolling(20).mean())

        # Price range features
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_14'] = df['true_range'].rolling(14).mean()
        df['range_efficiency'] = abs(df['close'] - df['open']) / df['true_range']

        # Momentum indicators
        for period in [5, 10, 14, 20]:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)
            df[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1

        # Support/Resistance features if provided
        if support_resistance_data is not None:
            sr_df = support_resistance_data.copy()

            # Distance to support/resistance levels
            df['distance_to_support'] = (df['close'] - sr_df['range_bottom']) / df['close']
            df['distance_to_resistance'] = (sr_df['range_top'] - df['close']) / df['close']

            # Position within range
            df['range_position'] = sr_df.get('range_position', np.nan)
            df['range_width'] = sr_df.get('range_width', np.nan)

            # Range quality metrics
            df['range_stability'] = 1 - (sr_df['range_top'].rolling(10).std() / df['close'])
            df['range_age'] = sr_df.get('range_age', 0)

        # Technical pattern features
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_shadow'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['lower_shadow'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']

        # Price action patterns
        df['bullish_engulfing'] = (
            (df['close'] > df['open']) &
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1))
        ).astype(int)

        df['bearish_engulfing'] = (
            (df['close'] < df['open']) &
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1))
        ).astype(int)

        # Volume divergence features
        df['price_trend_5'] = np.where(df['close'] > df['close'].shift(5), 1, -1)
        df['volume_trend_5'] = np.where(df['volume'] > df['volume'].shift(5), 1, -1)
        df['volume_price_divergence'] = df['price_trend_5'] * df['volume_trend_5'] * -1

        # Market microstructure
        df['bid_ask_proxy'] = (df['high'] - df['low']) / df['close']
        df['closing_strength'] = (df['close'] - df['low']) / (df['high'] - df['low'])

        return df

    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def create_breakout_labels(self, data: pd.DataFrame,
                              lookforward_periods: int = None) -> pd.DataFrame:

        if lookforward_periods is None:
            lookforward_periods = self.lookforward_periods

        df = data.copy()

        # Calculate future price movements
        future_high = df['high'].rolling(window=lookforward_periods, min_periods=1).max().shift(-lookforward_periods)
        future_low = df['low'].rolling(window=lookforward_periods, min_periods=1).min().shift(-lookforward_periods)

        # Calculate breakout thresholds
        current_price = df['close']
        upward_threshold = current_price * (1 + self.breakout_threshold)
        downward_threshold = current_price * (1 - self.breakout_threshold)

        # Create binary labels
        df['upward_breakout'] = (future_high > upward_threshold).astype(int)
        df['downward_breakout'] = (future_low < downward_threshold).astype(int)

        # Additional validation - ensure significant price movement
        future_returns = (future_high - current_price) / current_price
        future_declines = (current_price - future_low) / current_price

        # Only consider as breakout if movement is sustained (not just a spike)
        sustained_upward = future_returns > self.breakout_threshold
        sustained_downward = future_declines > self.breakout_threshold

        df['upward_breakout'] = df['upward_breakout'] & sustained_upward
        df['downward_breakout'] = df['downward_breakout'] & sustained_downward

        return df

    def fit(self, data: pd.DataFrame,
            support_resistance_data: pd.DataFrame = None,
            test_size: float = 0.2,
            validate_model: bool = True) -> Dict[str, Any]:

        # Prepare features
        feature_df = self.prepare_features(data, support_resistance_data)

        # Create labels
        labeled_df = self.create_breakout_labels(feature_df)

        # Define feature columns (excluding target variables and non-predictive columns)
        exclude_cols = ['upward_breakout', 'downward_breakout', 'open', 'high', 'low', 'close', 'volume']
        if 'date' in labeled_df.columns:
            exclude_cols.append('date')

        self.feature_columns = [col for col in labeled_df.columns
                               if col not in exclude_cols and
                               labeled_df[col].dtype in ['float64', 'int64', 'float32', 'int32']]

        # Prepare training data
        X = labeled_df[self.feature_columns].fillna(0)
        y_up = labeled_df['upward_breakout']
        y_down = labeled_df['downward_breakout']

        # Remove rows with NaN targets
        valid_idx = ~(y_up.isna() | y_down.isna())
        X = X[valid_idx]
        y_up = y_up[valid_idx]
        y_down = y_down[valid_idx]

        results = {}

        # Split data for validation
        if validate_model and test_size > 0:
            X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
                X, y_up, y_down, test_size=test_size, random_state=42, stratify=y_up
            )
        else:
            X_train, X_test = X, None
            y_up_train, y_up_test = y_up, None
            y_down_train, y_down_test = y_down, None

        # Train upward breakout model
        self.upward_model.fit(X_train, y_up_train)

        # Train downward breakout model
        self.downward_model.fit(X_train, y_down_train)

        self.is_fitted = True

        # Model validation
        if validate_model and X_test is not None:
            # Upward breakout predictions
            y_up_pred = self.upward_model.predict(X_test)
            y_up_prob = self.upward_model.predict_proba(X_test)[:, 1]

            # Downward breakout predictions
            y_down_pred = self.downward_model.predict(X_test)
            y_down_prob = self.downward_model.predict_proba(X_test)[:, 1]

            results['upward_model'] = {
                'classification_report': classification_report(y_up_test, y_up_pred, output_dict=True),
                'roc_auc': roc_auc_score(y_up_test, y_up_prob),
                'confusion_matrix': confusion_matrix(y_up_test, y_up_pred).tolist(),
                'feature_importance': dict(zip(self.feature_columns,
                                             self.upward_model.feature_importances_))
            }

            results['downward_model'] = {
                'classification_report': classification_report(y_down_test, y_down_pred, output_dict=True),
                'roc_auc': roc_auc_score(y_down_test, y_down_prob),
                'confusion_matrix': confusion_matrix(y_down_test, y_down_pred).tolist(),
                'feature_importance': dict(zip(self.feature_columns,
                                             self.downward_model.feature_importances_))
            }

            # Training statistics
            results['training_stats'] = {
                'total_samples': len(X),
                'training_samples': len(X_train),
                'test_samples': len(X_test) if X_test is not None else 0,
                'upward_breakout_rate': float(y_up.mean()),
                'downward_breakout_rate': float(y_down.mean()),
                'feature_count': len(self.feature_columns)
            }

        return results

    def predict_breakout_probability(self, data: pd.DataFrame,
                                   support_resistance_data: pd.DataFrame = None) -> Dict[str, np.ndarray]:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Prepare features
        feature_df = self.prepare_features(data, support_resistance_data)

        # Select feature columns and handle missing values
        X = feature_df[self.feature_columns].fillna(0)

        # Get probability predictions
        upward_probs = self.upward_model.predict_proba(X)[:, 1]
        downward_probs = self.downward_model.predict_proba(X)[:, 1]

        # Calculate combined breakout probability
        no_breakout_prob = (1 - upward_probs) * (1 - downward_probs)
        total_breakout_prob = 1 - no_breakout_prob

        # Directional bias
        directional_bias = np.where(
            total_breakout_prob > 0,
            (upward_probs - downward_probs) / total_breakout_prob,
            0
        )

        return {
            'upward_probability': upward_probs,
            'downward_probability': downward_probs,
            'total_breakout_probability': total_breakout_prob,
            'directional_bias': directional_bias,
            'no_breakout_probability': no_breakout_prob
        }

    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:

        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importance")

        return {
            'upward_model': dict(zip(self.feature_columns, self.upward_model.feature_importances_)),
            'downward_model': dict(zip(self.feature_columns, self.downward_model.feature_importances_))
        }

    def get_model_info(self) -> Dict[str, Any]:

        if not self.is_fitted:
            return {"status": "Model not fitted"}

        return {
            "status": "Model fitted",
            "feature_count": len(self.feature_columns),
            "breakout_threshold": self.breakout_threshold,
            "lookforward_periods": self.lookforward_periods,
            "upward_model_params": self.upward_model_params,
            "downward_model_params": self.downward_model_params,
            "feature_columns": self.feature_columns
        }