from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

class BaseTechnicalIndicator(ABC):
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    @abstractmethod
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate indicator values.
        
        Args:
            data: time series data
            
        Returns:
            DataFrame with indicator columns added
        """
        pass
    
    @abstractmethod
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """Generate trading signals from indicator.
        
        Args:
            data: DataFrame with indicator values
            
        Returns:
            Series with signal values
        """
        pass


class PivotPointIndicator(BaseTechnicalIndicator):
    
    def __init__(self, window: int = 20, min_strength: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.min_strength = min_strength
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        df['pivot_high'] = self._find_pivot_highs(df['high'])
        df['pivot_low'] = self._find_pivot_lows(df['low'])
        
        df['support_1'], df['resistance_1'] = self._calculate_sr_levels(df)
        
        df['dynamic_support'] = self._calculate_dynamic_levels(df, 'pivot_low')
        df['dynamic_resistance'] = self._calculate_dynamic_levels(df, 'pivot_high')
        
        df['range_top'] = df['dynamic_resistance']
        df['range_bottom'] = df['dynamic_support']
        df['range_middle'] = (df['range_top'] + df['range_bottom']) / 2
        
        df['range_width'] = (df['range_top'] - df['range_bottom']) / df['close']
        
        df['range_position'] = ((df['close'] - df['range_bottom']) / 
                               (df['range_top'] - df['range_bottom'])).clip(0, 1)
        
        return df
    
    def _find_pivot_highs(self, highs: pd.Series) -> pd.Series:
        pivot_highs = pd.Series(np.nan, index=highs.index)
        
        for i in range(self.min_strength, len(highs) - self.min_strength):
            window_slice = highs.iloc[i-self.min_strength:i+self.min_strength+1]
            if highs.iloc[i] == window_slice.max():
                if self._is_significant_pivot(highs, i, 'high'):
                    pivot_highs.iloc[i] = highs.iloc[i]
                    
        return pivot_highs
    
    def _find_pivot_lows(self, lows: pd.Series) -> pd.Series:
        pivot_lows = pd.Series(np.nan, index=lows.index)
        
        for i in range(self.min_strength, len(lows) - self.min_strength):
            window_slice = lows.iloc[i-self.min_strength:i+self.min_strength+1]
            if lows.iloc[i] == window_slice.min():
                if self._is_significant_pivot(lows, i, 'low'):
                    pivot_lows.iloc[i] = lows.iloc[i]
                    
        return pivot_lows
    
    def _is_significant_pivot(self, series: pd.Series, idx: int, pivot_type: str) -> bool:
        center_price = series.iloc[idx]
        window_data = series.iloc[max(0, idx-self.window//2):idx+self.window//2+1]
        
        if pivot_type == 'high':
            percentile_90 = window_data.quantile(0.9)
            return center_price >= percentile_90
        else:  # low
            percentile_10 = window_data.quantile(0.1)
            return center_price <= percentile_10
    
    def _calculate_sr_levels(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        
        support_1 = 2 * pivot - df['high'].shift(1)
        resistance_1 = 2 * pivot - df['low'].shift(1)
        
        return support_1, resistance_1
    
    def _calculate_dynamic_levels(self, df: pd.DataFrame, pivot_col: str) -> pd.Series:
        levels = pd.Series(np.nan, index=df.index)
        
        for i in range(self.window, len(df)):
            recent_pivots = df[pivot_col].iloc[i-self.window:i].dropna()
            
            if len(recent_pivots) >= 2:
                if pivot_col == 'pivot_low':
                    levels.iloc[i] = recent_pivots.max()
                else:
                    levels.iloc[i] = recent_pivots.min()
                    
        levels = levels.fillna(method='ffill')
        
        return levels
    
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        
        df = self.calculate(data)
        
        buy_condition = (
            (df['range_position'] <= 0.2) &
            (df['close'] > df['range_bottom'] * 1.001) &
            (df['range_width'] > 0.02) &
            (df['close'] > df['close'].shift(1))
        )
        
        sell_condition = (
            (df['range_position'] >= 0.8) &
            (df['close'] < df['range_top'] * 0.999) &
            (df['range_width'] > 0.02) &
            (df['close'] < df['close'].shift(1))
        )
        
        exit_long_condition = (
            (df['range_position'] >= 0.7) |
            (df['close'] < df['range_bottom'] * 0.995)
        )
        
        exit_short_condition = (
            (df['range_position'] <= 0.3) |
            (df['close'] > df['range_top'] * 1.005)
        )
        
        signals[buy_condition] = 1
        signals[sell_condition] = -1
        signals[exit_long_condition] = 2
        signals[exit_short_condition] = -2
        
        return signals
    
    def get_range_quality(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate(data)
        
        width_score = np.clip(df['range_width'] * 50, 0, 1)
        
        position_variance = df['range_position'].rolling(window=self.window).var()
        distribution_score = 1 - np.abs(df['range_position'] - 0.5) * 2
        
        range_stability = 1 - (df['range_top'].rolling(window=10).std() / df['close'])
        range_stability = np.clip(range_stability, 0, 1)
        
        quality_score = (width_score + distribution_score + range_stability) / 3
        
        return quality_score
    
    def identify_breakouts(self, data: pd.DataFrame, 
                          breakout_threshold: float = 0.01) -> Dict[str, pd.Series]:
        df = self.calculate(data)
        
        upside_breakout = (
            (df['close'] > df['range_top'] * (1 + breakout_threshold)) &
            (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5)
        )
        
        downside_breakout = (
            (df['close'] < df['range_bottom'] * (1 - breakout_threshold)) &
            (df['volume'] > df['volume'].rolling(window=20).mean() * 1.5)
        )
        
        return {
            'upside_breakout': upside_breakout,
            'downside_breakout': downside_breakout
        }


class BollingerBandsIndicator(BaseTechnicalIndicator):
    
    def __init__(self, window: int = 20, num_std: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        self.num_std = num_std
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        df['bb_middle'] = df['close'].rolling(window=self.window).mean()
        rolling_std = df['close'].rolling(window=self.window).std()
        
        df['bb_upper'] = df['bb_middle'] + (rolling_std * self.num_std)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * self.num_std)
        
        df['bb_position'] = ((df['close'] - df['bb_lower']) / 
                            (df['bb_upper'] - df['bb_lower'])).clip(0, 1)
        
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
        
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate(data)
        signals = pd.Series(0, index=data.index)
        
        buy_signal = (df['bb_position'] <= 0.1) & (df['close'] > df['close'].shift(1))
        sell_signal = (df['bb_position'] >= 0.9) & (df['close'] < df['close'].shift(1))
        
        signals[buy_signal] = 1
        signals[sell_signal] = -1
        
        return signals


class RSIIndicator(BaseTechnicalIndicator):
    
    def __init__(self, window: int = 14, **kwargs):
        super().__init__(**kwargs)
        self.window = window
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        delta = df['close'].diff()
        
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=self.window).mean()
        avg_losses = losses.rolling(window=self.window).mean()
        
        rs = avg_gains / avg_losses
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
        
    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        df = self.calculate(data)
        signals = pd.Series(0, index=data.index)
        
        oversold = df['rsi'] <= 30
        overbought = df['rsi'] >= 70
        
        signals[oversold] = 1
        signals[overbought] = -1
        
        return signals