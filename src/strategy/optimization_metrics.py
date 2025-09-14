import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ..backtesting.backtest_engine import BacktestResult


@dataclass
class OptimizationMetrics:
    """Comprehensive metrics for strategy optimization."""

    # Return metrics
    total_return: float
    annualized_return: float
    excess_return: float

    # Risk metrics
    volatility: float
    max_drawdown: float
    max_drawdown_duration: int
    downside_deviation: float

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float

    # Consistency metrics
    monthly_win_rate: float
    var_95: float
    cvar_95: float

    # Market exposure
    time_in_market: float
    avg_holding_period: float


class StrategyEvaluator:
    """Enhanced evaluation system for strategy optimization."""

    def __init__(self, benchmark_return: float = 0.02, risk_free_rate: float = 0.02):
        """
        Initialize evaluator.

        Args:
            benchmark_return: Annual benchmark return for excess calculations
            risk_free_rate: Annual risk-free rate
        """
        self.benchmark_return = benchmark_return
        self.risk_free_rate = risk_free_rate

    def calculate_comprehensive_metrics(self, backtest_result: BacktestResult) -> OptimizationMetrics:
        """
        Calculate comprehensive optimization metrics from backtest results.

        Args:
            backtest_result: Results from strategy backtest

        Returns:
            OptimizationMetrics with all calculated values
        """

        portfolio_returns = backtest_result.portfolio_returns
        trades = backtest_result.trades
        portfolio_values = backtest_result.portfolio_values

        # Basic return metrics
        total_return = self._calculate_total_return(portfolio_values)
        annualized_return = self._calculate_annualized_return(portfolio_returns)
        excess_return = annualized_return - self.benchmark_return

        # Risk metrics
        volatility = self._calculate_volatility(portfolio_returns)
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        max_dd_duration = self._calculate_max_drawdown_duration(portfolio_values)
        downside_deviation = self._calculate_downside_deviation(portfolio_returns)

        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(portfolio_returns, volatility)
        sortino_ratio = self._calculate_sortino_ratio(portfolio_returns, downside_deviation)
        calmar_ratio = self._calculate_calmar_ratio(annualized_return, max_drawdown)

        # Trade statistics
        trade_stats = self._calculate_trade_statistics(trades)

        # Consistency metrics
        monthly_win_rate = self._calculate_monthly_win_rate(portfolio_returns)
        var_95, cvar_95 = self._calculate_var_cvar(portfolio_returns)

        # Market exposure metrics
        time_in_market = self._calculate_time_in_market(trades, len(portfolio_returns))
        avg_holding_period = self._calculate_avg_holding_period(trades)

        return OptimizationMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            excess_return=excess_return,
            volatility=volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_dd_duration,
            downside_deviation=downside_deviation,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=trade_stats['total_trades'],
            win_rate=trade_stats['win_rate'],
            avg_win=trade_stats['avg_win'],
            avg_loss=trade_stats['avg_loss'],
            profit_factor=trade_stats['profit_factor'],
            expectancy=trade_stats['expectancy'],
            monthly_win_rate=monthly_win_rate,
            var_95=var_95,
            cvar_95=cvar_95,
            time_in_market=time_in_market,
            avg_holding_period=avg_holding_period
        )

    def _calculate_total_return(self, portfolio_values: pd.Series) -> float:
        """Calculate total portfolio return."""
        if len(portfolio_values) == 0 or portfolio_values.iloc[0] == 0:
            return 0.0
        return (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1.0

    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """Calculate annualized return."""
        if len(returns) == 0:
            return 0.0

        # Assuming hourly returns, convert to annual
        periods_per_year = 252 * 24  # Trading days * hours per day
        cumulative_return = (1 + returns).prod() - 1
        n_periods = len(returns)

        if n_periods == 0 or cumulative_return <= -1:
            return 0.0

        try:
            annual_return = (1 + cumulative_return) ** (periods_per_year / n_periods) - 1
            return annual_return
        except (ValueError, OverflowError):
            return 0.0

    def _calculate_volatility(self, returns: pd.Series) -> float:
        """Calculate annualized volatility."""
        if len(returns) == 0:
            return 0.0

        # Convert hourly to annual volatility
        periods_per_year = 252 * 24
        return returns.std() * np.sqrt(periods_per_year)

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(portfolio_values) == 0:
            return 0.0

        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        return drawdown.min()

    def _calculate_max_drawdown_duration(self, portfolio_values: pd.Series) -> int:
        """Calculate maximum drawdown duration in periods."""
        if len(portfolio_values) == 0:
            return 0

        running_max = portfolio_values.expanding().max()
        drawdown = portfolio_values < running_max

        # Find longest consecutive True sequence
        max_duration = 0
        current_duration = 0

        for is_drawdown in drawdown:
            if is_drawdown:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def _calculate_downside_deviation(self, returns: pd.Series, target: float = 0.0) -> float:
        """Calculate downside deviation."""
        if len(returns) == 0:
            return 0.0

        downside_returns = returns[returns < target]
        if len(downside_returns) == 0:
            return 0.0

        # Annualize downside deviation
        periods_per_year = 252 * 24
        return np.sqrt(((downside_returns - target) ** 2).mean()) * np.sqrt(periods_per_year)

    def _calculate_sharpe_ratio(self, returns: pd.Series, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        if volatility == 0 or len(returns) == 0:
            return 0.0

        excess_returns = returns.mean() * 252 * 24 - self.risk_free_rate
        return excess_returns / volatility

    def _calculate_sortino_ratio(self, returns: pd.Series, downside_deviation: float) -> float:
        """Calculate Sortino ratio."""
        if downside_deviation == 0 or len(returns) == 0:
            return 0.0

        excess_returns = returns.mean() * 252 * 24 - self.risk_free_rate
        return excess_returns / downside_deviation

    def _calculate_calmar_ratio(self, annualized_return: float, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return 0.0
        return annualized_return / abs(max_drawdown)

    def _calculate_trade_statistics(self, trades: List[Any]) -> Dict[str, float]:
        """Calculate trade-level statistics."""
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }

        # Extract PnL from trades
        pnls = []
        for trade in trades:
            if hasattr(trade, 'pnl') and trade.pnl is not None:
                pnls.append(trade.pnl)

        if not pnls:
            return {
                'total_trades': len(trades),
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'expectancy': 0.0
            }

        pnls = np.array(pnls)
        winning_trades = pnls[pnls > 0]
        losing_trades = pnls[pnls < 0]

        # Calculate statistics
        total_trades = len(pnls)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0.0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0.0

        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0.0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy
        }

    def _calculate_monthly_win_rate(self, returns: pd.Series) -> float:
        """Calculate monthly win rate."""
        if len(returns) == 0:
            return 0.0

        try:
            # Resample to monthly returns
            monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            winning_months = (monthly_returns > 0).sum()
            total_months = len(monthly_returns)

            return winning_months / total_months if total_months > 0 else 0.0

        except Exception:
            # Fallback for non-datetime index
            return 0.0

    def _calculate_var_cvar(self, returns: pd.Series, confidence: float = 0.05) -> tuple:
        """Calculate Value at Risk and Conditional Value at Risk."""
        if len(returns) == 0:
            return 0.0, 0.0

        # Calculate VaR (5th percentile)
        var_95 = returns.quantile(confidence)

        # Calculate CVaR (expected value of returns below VaR)
        cvar_95 = returns[returns <= var_95].mean() if any(returns <= var_95) else var_95

        return var_95, cvar_95

    def _calculate_time_in_market(self, trades: List[Any], total_periods: int) -> float:
        """Calculate percentage of time in market."""
        if not trades or total_periods == 0:
            return 0.0

        total_position_periods = 0

        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                try:
                    # Calculate holding period (assuming hourly data)
                    if trade.exit_time and trade.entry_time:
                        holding_period = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                        total_position_periods += holding_period
                except Exception:
                    continue

        return total_position_periods / total_periods

    def _calculate_avg_holding_period(self, trades: List[Any]) -> float:
        """Calculate average holding period in hours."""
        if not trades:
            return 0.0

        holding_periods = []

        for trade in trades:
            if hasattr(trade, 'entry_time') and hasattr(trade, 'exit_time'):
                try:
                    if trade.exit_time and trade.entry_time:
                        period = (trade.exit_time - trade.entry_time).total_seconds() / 3600
                        holding_periods.append(period)
                except Exception:
                    continue

        return np.mean(holding_periods) if holding_periods else 0.0

    def create_composite_score(self, metrics: OptimizationMetrics,
                             weights: Optional[Dict[str, float]] = None) -> float:
        """
        Create composite optimization score from multiple metrics.

        Args:
            metrics: Calculated optimization metrics
            weights: Weights for different metrics

        Returns:
            Composite score for optimization
        """

        if weights is None:
            weights = {
                'sharpe_ratio': 0.3,
                'calmar_ratio': 0.2,
                'profit_factor': 0.2,
                'win_rate': 0.1,
                'total_return': 0.1,
                'max_drawdown_penalty': 0.1
            }

        # Normalize metrics to 0-1 scale with reasonable bounds
        normalized_sharpe = max(0, min(1, (metrics.sharpe_ratio + 1) / 3))  # -1 to 2 range
        normalized_calmar = max(0, min(1, metrics.calmar_ratio / 5))  # 0 to 5 range
        normalized_profit_factor = max(0, min(1, (metrics.profit_factor - 1) / 2))  # 1 to 3 range
        normalized_win_rate = metrics.win_rate  # Already 0-1
        normalized_return = max(0, min(1, (metrics.total_return + 0.2) / 0.4))  # -20% to 20%

        # Penalty for large drawdowns
        drawdown_penalty = max(0, min(1, abs(metrics.max_drawdown) / 0.3))  # 0 to 30% drawdown

        # Calculate weighted composite score
        composite_score = (
            weights['sharpe_ratio'] * normalized_sharpe +
            weights['calmar_ratio'] * normalized_calmar +
            weights['profit_factor'] * normalized_profit_factor +
            weights['win_rate'] * normalized_win_rate +
            weights['total_return'] * normalized_return -
            weights['max_drawdown_penalty'] * drawdown_penalty
        )

        return composite_score