#!/usr/bin/env python3
"""
Test script to verify optimization fixes work properly.
This script tests that the optimization no longer returns -10 Sharpe ratios systematically.
"""

import sys
import os
import pandas as pd
import numpy as np
import logging

# Add the project root to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.settings import DataConfig, RegimeConfig, StrategyConfig, RiskConfig
from src.strategy.parameter_optimizer import RangeTradingOptimizer
from src.backtesting.backtest_engine import BacktestEngine

def create_test_data(n_periods=1000):
    """Create synthetic test data for optimization testing."""
    np.random.seed(42)

    # Generate synthetic price data
    dates = pd.date_range(start='2023-01-01', periods=n_periods, freq='1H')

    # Create realistic OHLCV data with some patterns
    base_price = 100.0
    returns = np.random.normal(0, 0.01, n_periods)

    # Add some regime patterns
    for i in range(100, n_periods, 200):
        # Add ranging periods
        returns[i:i+50] = np.random.normal(0, 0.005, 50)

    # Add trending periods
    for i in range(50, n_periods, 300):
        trend_strength = np.random.choice([-0.002, 0.002])
        returns[i:i+30] = np.random.normal(trend_strength, 0.008, 30)

    # Calculate cumulative prices
    cumulative_returns = np.cumsum(returns)
    close_prices = base_price * np.exp(cumulative_returns)

    # Generate OHLCV
    data = []
    for i in range(n_periods):
        close = close_prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.005)))
        low = close * (1 - abs(np.random.normal(0, 0.005)))
        open_price = close * (1 + np.random.normal(0, 0.003))
        volume = np.random.randint(100000, 1000000)

        data.append({
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data, index=dates)
    return df

def test_optimization():
    """Test the optimization process."""

    # Setup logging to see what's happening
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Creating test data...")
    test_data = create_test_data(1000)

    # Split into train/val
    split_idx = int(0.8 * len(test_data))
    train_data = test_data.iloc[:split_idx].copy()
    val_data = test_data.iloc[split_idx:].copy()

    logger.info(f"Train data: {len(train_data)} periods")
    logger.info(f"Validation data: {len(val_data)} periods")

    # Create configurations
    data_config = DataConfig()
    backtest_engine = BacktestEngine(initial_capital=100000, transaction_cost=0.001)

    # Create optimizer with small number of trials for testing
    optimizer = RangeTradingOptimizer(
        data_config=data_config,
        backtest_engine=backtest_engine,
        optimization_metric='sharpe_ratio',
        n_trials=5,  # Small number for quick testing
        random_seed=42
    )

    logger.info("Starting optimization test...")

    try:
        # Run optimization
        results = optimizer.optimize(train_data=train_data, validation_data=val_data)

        logger.info(f"✅ Optimization completed successfully!")
        logger.info(f"Best Sharpe ratio: {results.best_value:.4f}")
        logger.info(f"Best parameters: {results.best_params}")

        # Check if we got reasonable results (not -10)
        if results.best_value > -5.0:
            logger.info("✅ SUCCESS: Optimization no longer returns -10 penalty scores")
            return True
        else:
            logger.warning(f"❌ ISSUE: Best value still very low: {results.best_value}")
            return False

    except Exception as e:
        logger.error(f"❌ FAILED: Optimization crashed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_optimization()
    if success:
        print("\n🎉 OPTIMIZATION FIXES VALIDATED SUCCESSFULLY!")
    else:
        print("\n❌ OPTIMIZATION FIXES NEED FURTHER WORK")
        sys.exit(1)