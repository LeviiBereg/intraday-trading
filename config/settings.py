from dataclasses import dataclass
from typing import Dict, Any
import os

@dataclass
class DataConfig:
    timeframes: list | None = None
    symbols: list | None = None
    data_source: str = "yahoo"
    cache_enabled: bool = True
    cache_dir: str = "data/cache"
    
    def __post_init__(self):
        if self.timeframes is None:
            self.timeframes = ["1h", "4h"]
        if self.symbols is None:
            self.symbols = ["SPY"]

@dataclass
class RegimeConfig:
    """Regime detection configuration."""
    hmm_states: int = 3
    volatility_window: int = 20
    trend_threshold: float = 0.02
    regime_memory: int = 5

@dataclass
class StrategyConfig:
    """Range trading strategy configuration."""
    pivot_window: int = 20
    volume_threshold: float = 1.5
    breakout_threshold: float = 0.7
    max_holding_period: int = 24
    close_at_eod: bool = True
    min_range_width: float = 0.02
    min_range_quality: float = 0.5
    base_position_size: float = 1.0
    support_distance_threshold: float = 0.15
    resistance_distance_threshold: float = 0.15
    max_trading_distance: float = 0.4
    # Score threshold parameters for entry signals
    buy_score_threshold: float = 0.3
    sell_score_threshold: float = 0.3

@dataclass
class RiskConfig:
    """Risk management configuration."""
    base_risk_per_trade: float = 0.02
    max_portfolio_risk: float = 0.06
    kelly_fraction: float = 0.25
    drawdown_limit: float = 0.15

@dataclass
class OptimizationConfig:
    """Parameter optimization configuration."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    walk_forward_periods: int = 12