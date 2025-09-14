import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base_strategy import BaseStrategy, SignalType, Position
from ..regime_detection.hmm import GaussianMixtureRegimeDetector
from ..indicators.technical import PivotPointIndicator
from ..indicators.breakout_probability import CatBoostBreakoutPredictor


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
        self.breakout_predictor = CatBoostBreakoutPredictor(
            breakout_threshold=0.02,
            lookforward_periods=24
        )

        # Strategy parameters
        self.max_breakout_probability = getattr(config, 'max_breakout_probability', 0.6)
        self.min_range_width = getattr(config, 'min_range_width', 0.02)
        self.range_entry_buffer = getattr(config, 'range_entry_buffer', 0.2)
        self.range_exit_buffer = getattr(config, 'range_exit_buffer', 0.8)
        self.min_range_quality = getattr(config, 'min_range_quality', 0.5)

        # Internal state
        self.is_fitted = False
        self.current_regime_data = None
        self.current_sr_data = None
        self.current_breakout_predictions = None

    def fit_models(self, price_data: pd.DataFrame,
                   support_resistance_data: pd.DataFrame) -> Dict:
        """Fit all the underlying models with historical data."""

        # Fit regime detection model
        hmm_features = self.regime_detector.prepare_features(price_data)
        self.regime_detector.fit(hmm_features)

        # Fit breakout prediction model
        training_results = self.breakout_predictor.fit(
            data=price_data,
            support_resistance_data=support_resistance_data,
            test_size=0.2,
            validate_model=True
        )

        self.is_fitted = True
        return training_results

    def update_model_predictions(self, price_data: pd.DataFrame,
                               support_resistance_data: pd.DataFrame) -> None:
        """Update all model predictions with latest data."""
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

        # Update breakout predictions
        breakout_predictions = self.breakout_predictor.predict_breakout_probability(
            data=price_data,
            support_resistance_data=support_resistance_data
        )

        self.current_breakout_predictions = pd.DataFrame(
            breakout_predictions,
            index=price_data.index
        )

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
        Calculate bounded numerical strategy score (-1 to +1).

        Returns:
            -1: Strong sell signal
             0: Neutral/hold
            +1: Strong buy signal
        """
        if (self.current_regime_data is None or
            self.current_sr_data is None or
            self.current_breakout_predictions is None):
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

            range_position = self.current_sr_data.loc[current_time, 'range_position']
            range_width = self.current_sr_data.loc[current_time, 'range_width']

            # Check minimum range width
            if pd.isna(range_width) or range_width < self.min_range_width:
                return 0.0

            # Get range quality - use a window around current time
            try:
                # Get a small window of data around current time for range quality calculation
                current_idx = self.current_sr_data.index.get_loc(current_time)
                start_idx = max(0, current_idx - 10)
                end_idx = min(len(self.current_sr_data), current_idx + 1)
                quality_data = self.current_sr_data.iloc[start_idx:end_idx]

                if len(quality_data) > 0:
                    range_quality = self.pivot_detector.get_range_quality(quality_data)
                    if len(range_quality) == 0 or range_quality.iloc[-1] < self.min_range_quality:
                        return 0.0
                else:
                    return 0.0
            except Exception:
                # If range quality calculation fails, continue with reduced confidence
                pass

            # Get breakout probability
            if current_time not in self.current_breakout_predictions.index:
                return 0.0

            breakout_prob = self.current_breakout_predictions.loc[current_time, 'total_breakout_probability']
            directional_bias = self.current_breakout_predictions.loc[current_time, 'directional_bias']

            # Avoid high breakout probability periods
            if breakout_prob > self.max_breakout_probability:
                return 0.0

            # Calculate base score based on range position
            if pd.isna(range_position):
                return 0.0

            # Strong buy signal near support (low range position)
            if range_position <= self.range_entry_buffer:
                base_score = 1.0 - (range_position / self.range_entry_buffer)

            # Strong sell signal near resistance (high range position)
            elif range_position >= self.range_exit_buffer:
                base_score = -1.0 + ((1.0 - range_position) / (1.0 - self.range_exit_buffer))

            # Neutral in middle of range
            else:
                base_score = 0.0

            # Adjust for breakout probability (reduce signal strength)
            breakout_adjustment = 1.0 - breakout_prob
            adjusted_score = base_score * breakout_adjustment

            # Adjust for range quality
            try:
                if 'range_quality' in locals() and len(range_quality) > 0:
                    quality_adjustment = min(range_quality.iloc[-1], 1.0)
                else:
                    quality_adjustment = 0.5  # neutral adjustment if no quality data
            except Exception:
                quality_adjustment = 0.5

            final_score = adjusted_score * quality_adjustment

            # Apply directional bias for fine-tuning
            if abs(directional_bias) > 0.3:
                bias_adjustment = directional_bias * 0.1  # Small adjustment
                final_score += bias_adjustment

            # Ensure bounded between -1 and 1
            return np.clip(final_score, -1.0, 1.0)

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

        # Get strategy score
        score = self.calculate_strategy_score(current_time)

        # Strong signal threshold
        if signal == SignalType.BUY and score > 0.5:
            return True
        elif signal == SignalType.SELL and score < -0.5:
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

        # Exit on high breakout probability
        if (self.current_breakout_predictions is not None and
            current_time in self.current_breakout_predictions.index):
            breakout_prob = self.current_breakout_predictions.loc[current_time, 'total_breakout_probability']
            if breakout_prob > self.max_breakout_probability:
                return True

        # Exit based on range position
        if (self.current_sr_data is not None and
            current_time in self.current_sr_data.index):
            range_position = self.current_sr_data.loc[current_time, 'range_position']

            if not pd.isna(range_position):
                # Exit long positions near resistance
                if position.side == 'long' and range_position >= 0.8:
                    return True

                # Exit short positions near support
                if position.side == 'short' and range_position <= 0.2:
                    return True

        return False

    def calculate_position_size(self, signal: SignalType, data: pd.DataFrame) -> float:
        """Calculate optimal position size for trade."""
        current_time = data.index[-1]

        # Get strategy score for signal strength
        score = abs(self.calculate_strategy_score(current_time))

        # Get breakout probability for risk adjustment
        breakout_risk = 0.5  # default
        if (self.current_breakout_predictions is not None and
            current_time in self.current_breakout_predictions.index):
            breakout_risk = self.current_breakout_predictions.loc[current_time, 'total_breakout_probability']

        # Get range width for volatility adjustment
        range_volatility = 0.02  # default
        if (self.current_sr_data is not None and
            current_time in self.current_sr_data.index):
            range_width = self.current_sr_data.loc[current_time, 'range_width']
            if not pd.isna(range_width):
                range_volatility = range_width

        # Base position size from config
        base_size = getattr(self.config, 'base_position_size', 1.0)

        # Adjust for signal strength
        signal_adjustment = score

        # Adjust for breakout risk (reduce size for higher risk)
        risk_adjustment = 1.0 - breakout_risk

        # Adjust for volatility (reduce size for higher volatility)
        volatility_adjustment = min(1.0, 0.02 / max(range_volatility, 0.01))

        # Calculate final position size
        position_size = base_size * signal_adjustment * risk_adjustment * volatility_adjustment

        return max(0.0, position_size)