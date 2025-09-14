from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from ..strategy.base_strategy import BaseStrategy, Position


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    trades: List[Position]
    portfolio_values: pd.Series
    portfolio_returns: pd.Series
    metrics: Dict[str, float]
    signals: pd.Series
    start_date: datetime
    end_date: datetime
    initial_capital: float


class BaseBacktestEngine(ABC):

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.results: Dict[str, Any] = {}

    @abstractmethod
    def load_test_data(self, symbol: str, start_date: str,
                      end_date: str) -> pd.DataFrame:
        """Load historical data for backtesting.

        Args:
            symbol: Trading symbol
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Historical price data
        """
        pass

    @abstractmethod
    def run_backtest(self, strategy: BaseStrategy, signal_generator: Any,
                    data: pd.DataFrame, initial_capital: float) -> BacktestResult:
        """Execute backtest on historical data.

        Args:
            strategy: Trading strategy instance
            signal_generator: Signal generator instance
            data: Historical market data
            initial_capital: Starting capital

        Returns:
            BacktestResult with comprehensive results
        """
        pass

    @abstractmethod
    def calculate_performance_metrics(self, trades: List[Position],
                                    portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics from completed trades.

        Args:
            trades: List of completed trades
            portfolio_returns: Portfolio return series

        Returns:
            Dictionary with performance metrics
        """
        pass

    @abstractmethod
    def generate_equity_curve(self, trades: List[Position]) -> pd.Series:
        """Generate portfolio equity curve.

        Args:
            trades: List of completed trades

        Returns:
            Series with equity values over time
        """
        pass


class BacktestEngine(BaseBacktestEngine):
    """Concrete implementation of backtesting engine."""

    def __init__(self, initial_capital: float = 100000,
                 transaction_cost: float = 0.001):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Transaction cost as fraction of trade value
        """
        super().__init__(initial_capital)
        self.transaction_cost = transaction_cost

    def load_test_data(self, symbol: str, start_date: str,
                      end_date: str) -> pd.DataFrame:
        """Load test data - placeholder implementation."""
        # This would typically load from a data source
        # For now, return empty DataFrame
        return pd.DataFrame()

    def run_backtest(self, strategy: BaseStrategy, signal_generator: Any,
                    data: pd.DataFrame, initial_capital: Optional[float] = None) -> BacktestResult:
        """Execute comprehensive backtest."""

        if initial_capital is None:
            initial_capital = self.initial_capital

        # Generate signals
        signals = signal_generator.generate_entry_signals(data)

        # Initialize tracking variables
        trades = []
        portfolio_value = initial_capital
        portfolio_values = pd.Series(portfolio_value, index=[data.index[0]])
        current_position = None
        cash = initial_capital

        # Simulate trading
        for i, (timestamp, signal) in enumerate(signals.items()):
            if i == 0:
                continue

            current_price = data.loc[timestamp, 'close']

            # Handle existing position
            if current_position is not None:
                # Check for exit conditions
                should_exit = strategy.should_exit_trade(current_position, data.loc[:timestamp])

                if should_exit:
                    # Exit position
                    exit_price = current_price
                    position_value = current_position.quantity * exit_price
                    transaction_cost_exit = position_value * self.transaction_cost

                    # Calculate PnL
                    if current_position.side == 'long':
                        pnl = position_value - current_position.entry_value - transaction_cost_exit - current_position.entry_cost
                    else:  # short
                        pnl = current_position.entry_value - position_value - transaction_cost_exit - current_position.entry_cost

                    # Update position
                    current_position.exit_time = timestamp
                    current_position.exit_price = exit_price
                    current_position.pnl = pnl

                    # Update cash
                    cash += position_value - transaction_cost_exit
                    if current_position.side == 'short':
                        cash += current_position.entry_value  # Return borrowed cash

                    trades.append(current_position)
                    current_position = None

            # Handle new position entry
            if current_position is None and signal in ['buy', 'sell']:
                # Convert signal to SignalType enum
                from ..strategy.base_strategy import SignalType
                signal_type = SignalType.BUY if signal == 'buy' else SignalType.SELL

                # Calculate position size using correct method signature
                position_size = strategy.calculate_position_size(signal_type, data.loc[:timestamp])

                if position_size > 0:
                    # Enter new position
                    side = 'long' if signal == 'buy' else 'short'
                    quantity = position_size / current_price
                    entry_value = quantity * current_price
                    entry_cost = entry_value * self.transaction_cost

                    if cash >= entry_value + entry_cost:
                        current_position = Position(
                            symbol=data.attrs.get('symbol', 'UNKNOWN'),
                            side=side,
                            quantity=quantity,
                            entry_time=timestamp,
                            entry_price=current_price,
                            entry_value=entry_value,
                            entry_cost=entry_cost
                        )

                        # Update cash
                        if side == 'long':
                            cash -= (entry_value + entry_cost)
                        else:  # short - simplified short handling
                            cash -= entry_cost  # Only pay transaction cost

            # Calculate portfolio value
            current_portfolio_value = cash
            if current_position is not None:
                position_market_value = current_position.quantity * current_price
                if current_position.side == 'long':
                    current_portfolio_value += position_market_value
                else:  # short
                    # For short positions, add the difference between entry and current value
                    current_portfolio_value += (current_position.entry_value - position_market_value)

            portfolio_values = pd.concat([portfolio_values, pd.Series([current_portfolio_value], index=[timestamp])])

        # Close any remaining position
        if current_position is not None:
            final_price = data.iloc[-1]['close']
            position_value = current_position.quantity * final_price
            transaction_cost_exit = position_value * self.transaction_cost

            if current_position.side == 'long':
                pnl = position_value - current_position.entry_value - transaction_cost_exit - current_position.entry_cost
            else:
                pnl = current_position.entry_value - position_value - transaction_cost_exit - current_position.entry_cost

            current_position.exit_time = data.index[-1]
            current_position.exit_price = final_price
            current_position.pnl = pnl
            trades.append(current_position)

        # Calculate returns
        portfolio_returns = portfolio_values.pct_change().fillna(0)

        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(trades, portfolio_returns)

        return BacktestResult(
            trades=trades,
            portfolio_values=portfolio_values,
            portfolio_returns=portfolio_returns,
            metrics=metrics,
            signals=signals,
            start_date=data.index[0],
            end_date=data.index[-1],
            initial_capital=initial_capital
        )

    def calculate_performance_metrics(self, trades: List[Position],
                                    portfolio_returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""

        if len(portfolio_returns) == 0:
            return self._empty_metrics()

        # Basic return metrics
        total_return = (portfolio_returns + 1).prod() - 1

        # Annualized metrics (assuming hourly data)
        periods_per_year = 252 * 24
        n_periods = len(portfolio_returns)

        if n_periods > 0:
            annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
            volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
        else:
            annualized_return = 0.0
            volatility = 0.0

        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(portfolio_returns)

        # Risk-adjusted returns
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0

        # Trade statistics
        if trades:
            trade_pnls = [trade.pnl for trade in trades if trade.pnl is not None]

            if trade_pnls:
                winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
                losing_trades = [pnl for pnl in trade_pnls if pnl < 0]

                win_rate = len(winning_trades) / len(trade_pnls)
                avg_win = np.mean(winning_trades) if winning_trades else 0.0
                avg_loss = np.mean(losing_trades) if losing_trades else 0.0

                gross_profit = sum(winning_trades) if winning_trades else 0.0
                gross_loss = abs(sum(losing_trades)) if losing_trades else 0.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            else:
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0
                profit_factor = 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }

    def generate_equity_curve(self, trades: List[Position]) -> pd.Series:
        """Generate portfolio equity curve from trades."""
        # This is a simplified implementation
        # In practice, this would need more sophisticated equity curve calculation

        if not trades:
            return pd.Series([self.initial_capital])

        equity_curve = pd.Series(dtype=float)
        running_equity = self.initial_capital

        for trade in trades:
            if trade.pnl is not None:
                running_equity += trade.pnl
                if trade.exit_time is not None:
                    equity_curve[trade.exit_time] = running_equity

        return equity_curve