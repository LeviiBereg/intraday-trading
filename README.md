# Intraday Range Trading Strategy

A systematic range trading strategy implementation that combines statistical learning with technical indicators for 1H and 4H timeframes. This repository contains a modular system for executing intraday range-based strategies with regime-aware risk management.

## 📊 Overview

This project implements a dynamic range trading strategy that:
- Identifies price oscillations between support and resistance zones
- Detects regime shifts and false breakouts
- Applies machine learning techniques to filter entry signals
- Manages overnight risk through adaptive position sizing
- Provides backtesting and walk-forward optimization

## 🎯 Key Features

- **Multi-Timeframe Analysis**: Works with 1H and 4H price data
- **Regime Detection**: Uses Hidden Markov Models and volatility-based classification
- **Dynamic Risk Management**: Adaptive position sizing based on market conditions

## 📁 Repository Structure

```
intraday-trading/
├── README.md
├── requirements.txt
├── config/
│   ├── __init__.py
│   └── settings.py              # Configuration parameters
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py       # Historical data acquisition
│   │   ├── preprocessing.py     # Data cleaning and feature engineering
│   │   └── market_data.py       # Real-time data interface
│   ├── indicators/
│   │   ├── __init__.py
│   │   ├── technical.py         # Technical indicators (Bollinger Bands, etc.)
│   │   ├── statistical.py       # Statistical measures and volatility
│   │   └── custom.py            # Custom range detection indicators
│   ├── regime_detection/
│   │   ├── __init__.py
│   │   ├── hmm_model.py         # Hidden Markov Model implementation
│   │   ├── volatility_regime.py # Volatility-based regime classification
│   │   └── trend_detector.py    # Trend vs range identification
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── base_strategy.py     # Abstract base strategy class
│   │   ├── range_strategy.py    # Main range trading strategy
│   │   ├── signal_generator.py  # Entry/exit signal generation
│   │   └── ml_filter.py         # ML-based signal filtering
│   ├── risk_management/
│   │   ├── __init__.py
│   │   ├── position_sizing.py   # Kelly criterion, ERC, Risk Parity
│   │   ├── risk_metrics.py      # VaR, drawdown calculations
│   │   └── dynamic_sizing.py    # Regime-aware position sizing
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging configuration
│       ├── performance.py       # Performance calculation utilities
│       └── visualization.py     # Plotting and visualization tools
├── backtesting/
│   ├── __init__.py
│   ├── backtest_engine.py       # Main backtesting framework
│   ├── walk_forward.py          # Walk-forward optimization
│   ├── monte_carlo.py           # Monte Carlo simulations
│   └── stress_testing.py        # Stress test scenarios
├── examples/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_regime_detection.ipynb
│   ├── 03_strategy_evaluation.ipynb
│   └── 04_backtesting.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py
│   ├── test_strategy.py
│   ├── test_regime_detection.py
│   └── test_risk_management.py
└── scripts/
    ├── run_backtest.py          # Execute backtesting
    ├── optimize_parameters.py   # Parameter optimization
    ├── generate_report.py       # Performance reporting
    └── live_trading.py          # Live trading execution
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/LeviiBereg/intraday-trading.git
cd intraday-trading

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Configuration

Key parameters can be configured in `config/settings.py`:
TBD

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.