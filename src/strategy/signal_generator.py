from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from .base_strategy import SignalType, Position
from .range_trading_strategy import RangeTradingStrategy


class BaseSignalGenerator(ABC):

    @abstractmethod
    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals.

        Args:
            data: Market data with indicators

        Returns:
            Series with entry signals
        """
        pass

    @abstractmethod
    def generate_exit_signals(self, data: pd.DataFrame,
                            positions: List[Any]) -> Dict[str, bool]:
        """Generate exit signals for active positions.

        Args:
            data: Current market data
            positions: List of active positions

        Returns:
            Dict mapping position IDs to exit decisions
        """
        pass

    @abstractmethod
    def filter_signals(self, signals: pd.Series,
                      filters: Dict[str, Any]) -> pd.Series:
        """Apply filters to trading signals.

        Args:
            signals: Raw trading signals
            filters: Filter criteria

        Returns:
            Filtered signals
        """
        pass


class RangeTradingSignalGenerator(BaseSignalGenerator):
    """
    Signal generator specifically designed for range trading strategy.
    Converts numerical strategy scores into discrete buy/sell/neutral signals.
    """

    def __init__(self, strategy: RangeTradingStrategy,
                 buy_threshold: float = 0.5,
                 sell_threshold: float = -0.5,
                 neutral_buffer: float = 0.1):
        """
        Initialize signal generator.

        Args:
            strategy: Range trading strategy instance
            buy_threshold: Score threshold for buy signals
            sell_threshold: Score threshold for sell signals
            neutral_buffer: Buffer around zero for neutral signals
        """
        self.strategy = strategy
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.neutral_buffer = neutral_buffer

        # Signal smoothing parameters
        self.signal_lookback = 2
        self.min_signal_strength = 0.2

    def generate_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate entry signals based on strategy scores."""

        # Get raw strategy scores
        scores = self.strategy.generate_signals(data)

        # Initialize signal series
        signals = pd.Series('neutral', index=data.index, name='signals')

        for i, (timestamp, score) in enumerate(scores.items()):

            # Skip if insufficient data for smoothing
            if i < self.signal_lookback:
                continue

            # Get recent scores for smoothing
            recent_scores = scores.iloc[max(0, i-self.signal_lookback):i+1]
            avg_score = recent_scores.mean()

            # Generate signals based on thresholds
            if avg_score >= self.buy_threshold and score >= self.min_signal_strength:
                signals.loc[timestamp] = 'buy'
            elif avg_score <= self.sell_threshold and abs(score) >= self.min_signal_strength:
                signals.loc[timestamp] = 'sell'
            else:
                signals.loc[timestamp] = 'neutral'

        return signals

    def generate_exit_signals(self, data: pd.DataFrame,
                            positions: List[Position]) -> Dict[str, bool]:
        """Generate exit signals for active positions."""

        exit_decisions = {}
        current_time = data.index[-1]

        for position in positions:
            position_id = f"{position.symbol}_{position.entry_time}"

            # Use strategy's exit logic
            should_exit = self.strategy.should_exit_trade(position, data)

            # Additional exit conditions based on signal reversal
            if not should_exit:
                current_score = self.strategy.calculate_strategy_score(current_time)

                # Exit long position if strong negative signal
                if position.side == 'long' and current_score < -0.3:
                    should_exit = True

                # Exit short position if strong positive signal
                elif position.side == 'short' and current_score > 0.3:
                    should_exit = True

            exit_decisions[position_id] = should_exit

        return exit_decisions

    def filter_signals(self, signals: pd.Series,
                      filters: Dict[str, Any]) -> pd.Series:
        """Apply additional filters to trading signals."""

        filtered_signals = signals.copy()

        # Time-based filters
        if 'allowed_hours' in filters:
            allowed_hours = filters['allowed_hours']
            hour_mask = filtered_signals.index.hour.isin(allowed_hours)
            filtered_signals[~hour_mask] = 'neutral'

        # Minimum signal gap filter
        if 'min_signal_gap' in filters:
            min_gap = filters['min_signal_gap']
            filtered_signals = self._apply_signal_gap_filter(filtered_signals, min_gap)

        # Maximum daily signals filter
        if 'max_daily_signals' in filters:
            max_daily = filters['max_daily_signals']
            filtered_signals = self._apply_daily_limit_filter(filtered_signals, max_daily)

        # Volatility filter
        if 'max_volatility' in filters and hasattr(self.strategy, 'current_sr_data'):
            max_vol = filters['max_volatility']
            filtered_signals = self._apply_volatility_filter(filtered_signals, max_vol)

        return filtered_signals

    def _apply_signal_gap_filter(self, signals: pd.Series, min_gap_hours: int) -> pd.Series:
        """Ensure minimum time gap between signals."""

        filtered = signals.copy()
        last_signal_time = None

        for timestamp, signal in signals.items():
            if signal != 'neutral':
                if last_signal_time is not None:
                    time_diff = (timestamp - last_signal_time).total_seconds() / 3600
                    if time_diff < min_gap_hours:
                        filtered.loc[timestamp] = 'neutral'
                        continue

                last_signal_time = timestamp

        return filtered

    def _apply_daily_limit_filter(self, signals: pd.Series, max_daily: int) -> pd.Series:
        """Limit number of signals per day."""

        filtered = signals.copy()
        daily_counts = {}

        for timestamp, signal in signals.items():
            if signal != 'neutral':
                date = timestamp.date()

                if date not in daily_counts:
                    daily_counts[date] = 0

                if daily_counts[date] >= max_daily:
                    filtered.loc[timestamp] = 'neutral'
                else:
                    daily_counts[date] += 1

        return filtered

    def _apply_volatility_filter(self, signals: pd.Series, max_volatility: float) -> pd.Series:
        """Filter out signals during high volatility periods."""

        filtered = signals.copy()

        if self.strategy.current_sr_data is None:
            return filtered

        for timestamp, signal in signals.items():
            if (signal != 'neutral' and
                timestamp in self.strategy.current_sr_data.index):

                range_width = self.strategy.current_sr_data.loc[timestamp, 'range_width']

                if not pd.isna(range_width) and range_width > max_volatility:
                    filtered.loc[timestamp] = 'neutral'

        return filtered

    def get_signal_strength(self, timestamp: pd.Timestamp) -> float:
        """Get signal strength at specific timestamp."""
        return abs(self.strategy.calculate_strategy_score(timestamp))

    def get_signal_statistics(self, signals: pd.Series) -> Dict[str, Any]:
        """Calculate signal statistics."""

        stats = {
            'total_signals': len(signals[signals != 'neutral']),
            'buy_signals': len(signals[signals == 'buy']),
            'sell_signals': len(signals[signals == 'sell']),
            'neutral_periods': len(signals[signals == 'neutral']),
            'signal_frequency': len(signals[signals != 'neutral']) / len(signals) if len(signals) > 0 else 0
        }

        # Daily signal distribution
        non_neutral_signals = signals[signals != 'neutral']
        if len(non_neutral_signals) > 0:
            try:
                # Extract dates from the non-neutral signals' index
                signal_dates = pd.to_datetime(non_neutral_signals.index).date
                daily_signals = non_neutral_signals.groupby(signal_dates).count()

                stats['avg_daily_signals'] = daily_signals.mean() if len(daily_signals) > 0 else 0
                stats['max_daily_signals'] = daily_signals.max() if len(daily_signals) > 0 else 0
                stats['signal_days'] = len(daily_signals[daily_signals > 0]) if len(daily_signals) > 0 else 0
            except Exception:
                # Fallback if groupby fails
                stats['avg_daily_signals'] = 0
                stats['max_daily_signals'] = 0
                stats['signal_days'] = 0
        else:
            stats['avg_daily_signals'] = 0
            stats['max_daily_signals'] = 0
            stats['signal_days'] = 0

        return stats