# Intraday Range Trading Strategy

An automated intraday trading strategy that combines six key components into a single methodology for systematic range-bound trading. The strategy operates exclusively within detected price ranges while avoiding trend and breakout scenarios.

## 📊 Strategy Overview

This project implements an intelligent range trading system with strict breakout avoidance:

### Core Strategy Logic
- **Range-Only Trading**: Trades exclusively during identified trading range regimes
- **Immediate Breakout Exits**: Automatically closes positions when prices break out of ranges
- **No Trend Following**: Strategy deactivates during bullish/bearish trending periods
- **Support/Resistance Based**: Buys near dynamic support, sells near dynamic resistance

### Six-Component Architecture

1. **Regime Detection**: Hidden Markov Models identify trading ranges vs trending periods
2. **Support/Resistance Detection**: Pivot point analysis with volume validation
3. **Breakout Probability Forecasting**: CatBoost models predict breakout likelihood
4. **Dynamic Risk Management**: Modified Kelly Criterion with breakout probability weighting
5. **Parameter Optimization**: Genetic algorithms with walk-forward validation
6. **Performance Monitoring**: Real-time analytics with trade-level reporting

## 🎯 Key Features

- **Regime-Aware Trading**: Only operates in trading range conditions (3-state HMM)
- **Breakout Avoidance**: Probabilistic models prevent trading during high breakout risk
- **Dynamic Position Sizing**: Risk-adjusted sizing based on range quality and breakout probability
- **Multi-Layer Filtering**: Time-based, volatility, and signal gap filters

## 📁 Repository Structure

```
intraday-trading/
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration (dataclasses)
├── src/
│   ├── data/
│   │   ├── data_loader.py       # Yahoo Finance data acquisition
│   │   └── preprocessing.py     # Data cleaning and feature engineering
│   ├── regime_detection/
│   │   ├── hmm.py               # Gaussian Mixture HMM implementation
│   │   ├── volatility_regime.py # Volatility-based regime classification
│   │   └── trend_detector.py    # Trend vs range identification
│   ├── indicators/
│   │   ├── technical.py         # Pivot points, support/resistance detection
│   │   ├── statistical.py       # Statistical measures and volatility
│   │   ├── breakout_probability.py # CatBoost breakout prediction models
│   │   └── custom.py            # Custom range detection indicators
│   ├── strategy/
│   │   ├── base_strategy.py     # Abstract strategy interface (SignalType, Position)
│   │   ├── range_trading_strategy.py # Main range trading implementation
│   │   ├── signal_generator.py  # Entry/exit signal generation with filters
│   │   ├── parameter_optimizer.py # Bayesian optimization with Optuna
│   │   ├── optimization_metrics.py # Strategy evaluation metrics
│   │   └── component_optimizer.py # Component-level optimization
│   ├── risk_management/
│   │   ├── risk_metrics.py      # VaR, drawdown calculations
│   │   └── dynamic_sizing.py    # Regime-aware position sizing
│   └── backtesting/
│       ├── backtest_engine.py   # Comprehensive backtesting framework
│       └── walk_forward.py      # Walk-forward optimization
├── examples/
│   ├── workflow.ipynb           # Complete strategy workflow (Jupyter)
│   └── workflow.py              # Mirror script for workflow execution
├── tests/
│   └── __init__.py              # Test structure (placeholder)
└── utils/
    └── logger.py                # Logging configuration
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LeviiBereg/intraday-trading.git
cd intraday-trading

# Install dependencies (Poetry recommended)
poetry install
# OR with pip
pip install -r requirements.txt
```

### Running the Strategy

Execute the complete workflow:

```bash
# Run the full strategy workflow
python examples/workflow.py

# Or use the Jupyter notebook for interactive analysis
jupyter lab examples/workflow.ipynb
```

## 🔧 Strategy Configuration

Key parameters in `config/settings.py`:

```python
@dataclass
class StrategyConfig:
    max_breakout_probability: float = 0.6    # Max allowed breakout risk
    min_range_width: float = 0.02           # Minimum range width to trade
    range_entry_buffer: float = 0.2         # Entry zone (20% from support)
    range_exit_buffer: float = 0.8          # Exit zone (80% to resistance)
    min_range_quality: float = 0.5          # Minimum range quality score
    base_position_size: float = 1.0         # Base position sizing multiplier
```

## 📊 Strategy Workflow

1. **Data Acquisition**: Load 1H SPY data from Yahoo Finance
2. **Regime Detection**: Identify trading ranges using 3-state HMM
3. **S/R Detection**: Calculate dynamic support/resistance levels
4. **Breakout Prediction**: Train CatBoost models for breakout probability
5. **Signal Generation**: Generate filtered buy/sell signals
6. **Parameter Optimization**: Bayesian optimization with Optuna
7. **Backtesting**: Compare optimized vs baseline performance

## 🎯 Strategy Rules

### Entry Conditions
- ✅ Current regime = "trading_range"
- ✅ Price near support (range_position ≤ 0.2) → BUY
- ✅ Price near resistance (range_position ≥ 0.8) → SELL
- ✅ Breakout probability < max_breakout_probability
- ✅ Range width > min_range_width
- ✅ Range quality > min_range_quality

### Exit Conditions
- ❌ Regime changes to bullish/bearish breakout → IMMEDIATE EXIT
- ❌ Breakout probability exceeds threshold → EXIT
- ❌ Price breaks out of range boundaries → EXIT
- ✅ Target reached or stop loss hit

## 📈 Performance Tracking

The system tracks comprehensive metrics:
- Total return, Sharpe ratio, max drawdown
- Win rate, profit factor, trade frequency
- Regime alignment, breakout avoidance success
- Parameter sensitivity analysis

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.