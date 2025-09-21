import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_strategy import BaseStrategy, SignalType, Position
from ..regime_detection.hmm import GaussianMixtureRegimeDetector
from ..indicators.technical import PivotPointIndicator


class RangeTradingStrategy(BaseStrategy):
    """
    Range trading strategy that combines regime detection, support/resistance analysis,
    and breakout probability prediction to trade only within identified ranges.
    """

    def __init__(self, config: 'StrategyConfig'):
        super().__init__(config)

        # Initialize components
        self.regime_detector = GaussianMixtureRegimeDetector(
            n_states=3,
            random_state=13
        )
        self.pivot_detector = PivotPointIndicator(
            window=20,
            min_strength=2
        )

        # Strategy parameters
        self.min_range_width = getattr(config, 'min_range_width', 0.02)
        self.min_range_quality = getattr(config, 'min_range_quality', 0.5)
        # Distance-based trading parameters
        self.support_distance_threshold = getattr(config, 'support_distance_threshold', 0.15)
        self.resistance_distance_threshold = getattr(config, 'resistance_distance_threshold', 0.15)
        self.max_trading_distance = getattr(config, 'max_trading_distance', 0.4)
        # Score threshold parameters for entry signals
        self.buy_score_threshold = getattr(config, 'buy_score_threshold', 0.3)
        self.sell_score_threshold = getattr(config, 'sell_score_threshold', 0.3)

        # Internal state
        self.is_fitted = False
        self.current_regime_data = None
        self.current_sr_data = None

    def fit_models(self, price_data: pd.DataFrame,
                   support_resistance_data: pd.DataFrame) -> Dict:
        """Fit the regime detection model with historical data."""

        # Fit regime detection model
        hmm_features = self.regime_detector.prepare_features(price_data)
        self.regime_detector.fit(hmm_features)

        self.is_fitted = True
        return {'regime_model_fitted': True}

    def update_model_predictions(self, price_data: pd.DataFrame,
                               support_resistance_data: pd.DataFrame) -> None:
        """Update regime predictions and support/resistance data."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before generating predictions")

        # Update regime predictions
        hmm_features = self.regime_detector.prepare_features(price_data)
        regime_predictions = self.regime_detector.predict_regime(hmm_features)
        regime_probabilities = self.regime_detector.get_regime_probabilities(hmm_features)

        # Create regime dataframe aligned with price data
        regime_start_idx = len(price_data) - len(regime_predictions)
        self.current_regime_data = pd.DataFrame(index=price_data.index[regime_start_idx:])
        self.current_regime_data['regime_state'] = regime_predictions
        self.current_regime_data['regime_prob_0'] = regime_probabilities[:, 0]
        self.current_regime_data['regime_prob_1'] = regime_probabilities[:, 1]
        self.current_regime_data['regime_prob_2'] = regime_probabilities[:, 2]

        # Map regimes to labels
        regime_characteristics = self.regime_detector.get_regime_characteristics()
        regime_labels = self._map_regimes_to_labels(regime_characteristics)
        state_to_label = {i: regime_labels[f'regime_{i}'] for i in range(3)}
        self.current_regime_data['regime_label'] = self.current_regime_data['regime_state'].map(state_to_label)

        # Update support/resistance data
        self.current_sr_data = support_resistance_data.copy()

    def _map_regimes_to_labels(self, characteristics: Dict) -> Dict[str, str]:
        """Map regime characteristics to interpretable labels."""
        regime_labels = {}

        for regime_id, props in characteristics.items():
            features = props['features']
            returns = features['returns']
            volatility = features['volatility']
            momentum = features['momentum']

            if volatility > 0.015 and returns > 0.003 and momentum > 0.01:
                label = "bullish_breakout"
            elif volatility > 0.015 and returns < -0.003 and momentum < -0.01:
                label = "bearish_breakout"
            elif volatility < 0.012 and abs(returns) < 0.002:
                label = "trading_range"
            elif returns > 0:
                label = "bullish_trend"
            else:
                label = "bearish_trend"

            regime_labels[regime_id] = label

        return regime_labels

    def calculate_strategy_score(self, current_time: pd.Timestamp) -> float:
        """
        Calculate bounded numerical strategy score (-1 to +1) based on distance to support/resistance.

        Returns:
            -1: Strong sell signal (near resistance)
             0: Neutral/hold
            +1: Strong buy signal (near support)
        """
        if (self.current_regime_data is None or self.current_sr_data is None):
            return 0.0

        try:
            # Get current regime
            if current_time not in self.current_regime_data.index:
                return 0.0

            current_regime = self.current_regime_data.loc[current_time, 'regime_label']

            # Only trade in trading ranges
            if current_regime != 'trading_range':
                return 0.0

            # Get current support/resistance data
            if current_time not in self.current_sr_data.index:
                return 0.0

            range_width = self.current_sr_data.loc[current_time, 'range_width']
            dynamic_support = self.current_sr_data.loc[current_time, 'dynamic_support']
            dynamic_resistance = self.current_sr_data.loc[current_time, 'dynamic_resistance']

            # Check minimum range width
            if pd.isna(range_width) or range_width < self.min_range_width:
                return 0.0

            # Check if we have valid support/resistance levels
            if pd.isna(dynamic_support) or pd.isna(dynamic_resistance):
                return 0.0

            # Get current price from range position (reverse calculation)
            range_position = self.current_sr_data.loc[current_time, 'range_position']
            if pd.isna(range_position):
                return 0.0

            # Calculate current price from range position
            current_price = dynamic_support + (range_position * (dynamic_resistance - dynamic_support))

            # Calculate distances to support and resistance as percentage of range width
            range_span = dynamic_resistance - dynamic_support
            if range_span <= 0:
                return 0.0

            distance_to_support = (current_price - dynamic_support) / range_span
            distance_to_resistance = (dynamic_resistance - current_price) / range_span

            # Calculate signal strength based on distance thresholds
            base_score = 0.0

            # Buy signal: gradual increase as we approach support
            if distance_to_support <= self.support_distance_threshold:
                # Normalize distance to [0, 1] where 0 = at support, 1 = at threshold
                normalized_distance = distance_to_support / self.support_distance_threshold
                # Invert so closer to support gives stronger signal
                base_score = 1.0 - normalized_distance

            # Sell signal: gradual increase as we approach resistance
            elif distance_to_resistance <= self.resistance_distance_threshold:
                # Normalize distance to [0, 1] where 0 = at resistance, 1 = at threshold
                normalized_distance = distance_to_resistance / self.resistance_distance_threshold
                # Invert so closer to resistance gives stronger signal (negative)
                base_score = -(1.0 - normalized_distance)

            # Neutral in middle of range (beyond both thresholds)
            else:
                base_score = 0.0
            # Ensure bounded between -1 and 1
            return np.clip(base_score, -1.0, 1.0)

        except (KeyError, IndexError, ValueError):
            return 0.0

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals based on strategy score."""
        signals = pd.Series(0, index=data.index, name='strategy_signals')

        for timestamp in data.index:
            score = self.calculate_strategy_score(timestamp)
            signals.loc[timestamp] = score

        return signals

    def should_enter_trade(self, data: pd.DataFrame, signal: SignalType) -> bool:
        """Determine if conditions are met to enter trade."""
        current_time = data.index[-1]

        # Only trade in trading range regime
        if (self.current_regime_data is not None and
            current_time in self.current_regime_data.index):
            current_regime = self.current_regime_data.loc[current_time, 'regime_label']
            if current_regime != 'trading_range':
                return False

        # Get strategy score
        score = self.calculate_strategy_score(current_time)

        # Use tunable signal thresholds
        if signal == SignalType.BUY and score > self.buy_score_threshold:
            return True
        elif signal == SignalType.SELL and score < -self.sell_score_threshold:
            return True

        return False

    def should_exit_trade(self, position: Position, data: pd.DataFrame) -> bool:
        """Determine if position should be closed."""
        current_time = data.index[-1]

        # Always exit if not in trading range regime
        if (self.current_regime_data is not None and
            current_time in self.current_regime_data.index):
            current_regime = self.current_regime_data.loc[current_time, 'regime_label']
            if current_regime != 'trading_range':
                return True

        # Exit based on distance to target levels
        if (self.current_sr_data is not None and
            current_time in self.current_sr_data.index):

            dynamic_support = self.current_sr_data.loc[current_time, 'dynamic_support']
            dynamic_resistance = self.current_sr_data.loc[current_time, 'dynamic_resistance']

            if not (pd.isna(dynamic_support) or pd.isna(dynamic_resistance)):
                range_span = dynamic_resistance - dynamic_support
                if range_span > 0:
                    # Get current price from range position
                    range_position = self.current_sr_data.loc[current_time, 'range_position']
                    if pd.isna(range_position):
                        return False
                    current_price = dynamic_support + (range_position * range_span)

                    distance_to_support = (current_price - dynamic_support) / range_span
                    distance_to_resistance = (dynamic_resistance - current_price) / range_span

                    # Exit long positions when close to resistance
                    if position.side == 'long' and distance_to_resistance <= self.resistance_distance_threshold:
                        return True

                    # Exit short positions when close to support
                    if position.side == 'short' and distance_to_support <= self.support_distance_threshold:
                        return True

        return False

    def calculate_position_size(self, signal: SignalType, data: pd.DataFrame) -> float:
        """Calculate optimal position size for trade based on distance to support/resistance."""
        current_time = data.index[-1]

        # Get strategy score for signal strength
        score = abs(self.calculate_strategy_score(current_time))

        # Base position size from config
        base_size = getattr(self.config, 'base_position_size', 1.0)

        # Adjust for signal strength (closer to support/resistance = larger position)
        signal_adjustment = score

        # Calculate final position size
        position_size = base_size * signal_adjustment

        min_position_size = 100 * base_size
        if position_size > 0 and position_size < min_position_size:
            position_size = min_position_size

        return max(0.0, position_size)