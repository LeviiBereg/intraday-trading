from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import optuna
from dataclasses import dataclass
import logging
from ..backtesting.backtest_engine import BacktestEngine
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.settings import DataConfig, RegimeConfig, StrategyConfig, RiskConfig
from .range_trading_strategy import RangeTradingStrategy
from .signal_generator import RangeTradingSignalGenerator
from ..indicators.technical import PivotPointIndicator


@dataclass
class OptimizationResult:
    """Results from parameter optimization."""
    best_params: Dict[str, Any]
    best_value: float
    study: optuna.Study
    optimization_history: pd.DataFrame


class BaseParameterOptimizer(ABC):
    """Base class for parameter optimization."""

    @abstractmethod
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define the parameter search space."""
        pass

    @abstractmethod
    def create_strategy(self, params: Dict[str, Any]) -> Any:
        """Create strategy instance with given parameters."""
        pass

    @abstractmethod
    def evaluate_objective(self, strategy: Any, data: pd.DataFrame) -> float:
        """Evaluate the objective function."""
        pass


class RangeTradingOptimizer(BaseParameterOptimizer):
    """Bayesian optimization for range trading strategy parameters."""

    def __init__(self,
                 data_config: DataConfig,
                 backtest_engine: BacktestEngine,
                 optimization_metric: str = 'sharpe_ratio',
                 n_trials: int = 100,
                 random_seed: int = 42):
        """
        Initialize optimizer.

        Args:
            data_config: Data configuration
            backtest_engine: Backtesting engine
            optimization_metric: Metric to optimize ('sharpe_ratio', 'calmar_ratio', 'total_return')
            n_trials: Number of optimization trials
            random_seed: Random seed for reproducibility
        """
        self.data_config = data_config
        self.backtest_engine = backtest_engine
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        self.random_seed = random_seed

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define search space for distance-based range trading parameters."""

        params = {}

        # Strategy Parameters
        params['min_range_width'] = trial.suggest_float('min_range_width', 0.005, 0.05)
        params['min_range_quality'] = trial.suggest_float('min_range_quality', 0.1, 0.6)
        params['base_position_size'] = trial.suggest_float('base_position_size', 0.5, 2.0)

        # Distance-based trading parameters
        params['support_distance_threshold'] = trial.suggest_float('support_distance_threshold', 0.05, 0.5)
        params['resistance_distance_threshold'] = trial.suggest_float('resistance_distance_threshold', 0.05, 0.5)
        params['max_trading_distance'] = trial.suggest_float('max_trading_distance', 0.2, 0.6)

        # Score threshold parameters for strategy entry signals
        params['buy_score_threshold'] = trial.suggest_float('buy_score_threshold', 0.1, 0.7)
        params['sell_score_threshold'] = trial.suggest_float('sell_score_threshold', 0.1, 0.7)

        # Pivot Point Parameters
        params['pivot_window'] = trial.suggest_int('pivot_window', 10, 50)

        # Signal Generator Parameters
        params['buy_threshold'] = trial.suggest_float('buy_threshold', 0.05, 1.0)
        params['sell_threshold'] = trial.suggest_float('sell_threshold', 0.05, 1.0)
        params['neutral_buffer'] = trial.suggest_float('neutral_buffer', 0.05, 0.2)

        return params

    def create_strategy(self, params: Dict[str, Any]) -> Tuple[RangeTradingStrategy, RangeTradingSignalGenerator]:
        """Create strategy instance with optimized parameters."""

        strategy_config = StrategyConfig(
            pivot_window=params['pivot_window'],
            min_range_width=params['min_range_width'],
            min_range_quality=params['min_range_quality'],
            base_position_size=params['base_position_size'],
            support_distance_threshold=params['support_distance_threshold'],
            resistance_distance_threshold=params['resistance_distance_threshold'],
            max_trading_distance=params['max_trading_distance'],
            buy_score_threshold=params['buy_score_threshold'],
            sell_score_threshold=params['sell_score_threshold']
        )

        strategy = RangeTradingStrategy(strategy_config)

        signal_generator = RangeTradingSignalGenerator(
            strategy=strategy,
            buy_threshold=params['buy_threshold'],
            sell_threshold=params['sell_threshold'],
            neutral_buffer=params['neutral_buffer']
        )

        return strategy, signal_generator

    def evaluate_objective(self, strategy: RangeTradingStrategy,
                          signal_generator: RangeTradingSignalGenerator,
                          train_data: pd.DataFrame) -> float:
        """Evaluate strategy performance using backtest."""

        try:
            if train_data.empty or len(train_data) < 100:
                self.logger.warning(f"Insufficient training data: {len(train_data)} rows")
                return -10.0

            self.logger.debug(f"Creating support/resistance data for {len(train_data)} data points")
            pivot_detector = PivotPointIndicator(window=20, min_strength=2)
            sr_data = pivot_detector.calculate(train_data)

            if sr_data.empty:
                self.logger.warning("No support/resistance data generated")
                return -10.0

            # Fit the strategy models
            model_fit_results = strategy.fit_models(train_data, sr_data)
            self.logger.debug(f"Model fitting completed: {model_fit_results is not None}")

            # Update predictions for signal generation
            strategy.update_model_predictions(train_data, sr_data)

            # Run backtest
            results = self.backtest_engine.run_backtest(
                strategy=strategy,
                signal_generator=signal_generator,
                data=train_data,
                initial_capital=100000
            )

            # Validate backtest results
            if results is None or results.metrics is None:
                self.logger.warning("Backtest returned no results")
                return -10.0

            if len(results.trades) == 0:
                self.logger.debug("No trades generated during backtest")
                # Return small negative score instead of -10 for no trades
                return -0.5

            # Calculate optimization metric
            if self.optimization_metric == 'sharpe_ratio':
                if results.metrics['volatility'] > 0:
                    return results.metrics['sharpe_ratio']
                else:
                    return -10.0  # Penalize zero volatility

            elif self.optimization_metric == 'calmar_ratio':
                if results.metrics['max_drawdown'] < -0.001:  # Avoid division by zero
                    return abs(results.metrics['total_return'] / results.metrics['max_drawdown'])
                else:
                    return results.metrics['total_return'] if results.metrics['total_return'] > 0 else -10.0

            elif self.optimization_metric == 'total_return':
                return results.metrics['total_return']

            elif self.optimization_metric == 'profit_factor':
                return results.metrics.get('profit_factor', -10.0)
            
            elif self.optimization_metric == 'win_rate':
                return results.metrics.get('win_rate', -1)

            else:
                raise ValueError(f"Unknown optimization metric: {self.optimization_metric}")

        except ValueError as e:
            self.logger.error(f"Value error during optimization: {e}")
            return -10.0
        except AttributeError as e:
            self.logger.error(f"Attribute error (possibly missing data): {e}")
            return -10.0
        except KeyError as e:
            self.logger.error(f"Key error (missing column/index): {e}")
            return -10.0
        except Exception as e:
            self.logger.error(f"Unexpected error during backtest evaluation: {type(e).__name__}: {e}")
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return -10.0  # Return poor score for failed backtests

    def optimize(self, train_data: pd.DataFrame,
                validation_data: Optional[pd.DataFrame] = None) -> OptimizationResult:
        """
        Run Bayesian optimization to find best parameters.

        Args:
            train_data: Training data for optimization
            validation_data: Optional validation data for final evaluation

        Returns:
            OptimizationResult with best parameters and study results
        """

        # Input validation
        if train_data.empty:
            raise ValueError("Training data cannot be empty")

        if len(train_data) < 500:
            self.logger.warning(f"Limited training data: {len(train_data)} rows. Consider using more data for robust optimization.")

        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in train_data.columns]
        if missing_columns:
            raise ValueError(f"Training data missing required columns: {missing_columns}")

        self.logger.info(f"Starting optimization with {len(train_data)} data points")
        self.logger.info(f"Optimization metric: {self.optimization_metric}")
        self.logger.info(f"Number of trials: {self.n_trials}")

        def objective(trial):
            trial_num = trial.number
            self.logger.debug(f"Starting trial {trial_num}")

            # Get parameter suggestions
            params = self.define_search_space(trial)
            self.logger.debug(f"Trial {trial_num} parameters: {params}")

            # Create strategy with suggested parameters
            strategy, signal_generator = self.create_strategy(params)

            # Evaluate on training data
            score = self.evaluate_objective(strategy, signal_generator, train_data)
            self.logger.debug(f"Trial {trial_num} training score: {score:.4f}")

            # Log intermediate results
            trial.set_user_attr('train_score', score)

            # Optional validation score
            if validation_data is not None:
                val_score = self.evaluate_objective(strategy, signal_generator, validation_data)
                trial.set_user_attr('validation_score', val_score)
                # Use weighted combination of train and validation scores
                score = 0.7 * score + 0.3 * val_score

            return score

        # Create and run optimization study
        sampler = optuna.samplers.TPESampler(seed=self.random_seed)
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            study_name=f'range_trading_optimization_{self.optimization_metric}'
        )

        # Add pruner for early stopping of poor trials
        study.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )

        # Create optimization history DataFrame
        trials_df = study.trials_dataframe()

        # Extract best parameters
        best_params = study.best_params
        best_value = study.best_value

        self.logger.info(f"Optimization completed. Best {self.optimization_metric}: {best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")

        return OptimizationResult(
            best_params=best_params,
            best_value=best_value,
            study=study,
            optimization_history=trials_df
        )

    def cross_validate_parameters(self, data: pd.DataFrame,
                                 params: Dict[str, Any],
                                 n_splits: int = 5) -> Dict[str, float]:
        """
        Perform time series cross-validation on parameters.

        Args:
            data: Full dataset
            params: Parameters to validate
            n_splits: Number of CV splits

        Returns:
            Cross-validation metrics
        """

        # Create strategy with given parameters
        strategy, signal_generator = self.create_strategy(params)

        # Perform time series split
        split_size = len(data) // (n_splits + 1)
        cv_scores = []

        for i in range(n_splits):
            # Define train and validation periods
            train_start = i * split_size
            train_end = (i + 1) * split_size
            val_end = (i + 2) * split_size

            train_data = data.iloc[train_start:train_end]
            val_data = data.iloc[train_end:val_end]

            # Evaluate on validation fold
            score = self.evaluate_objective(strategy, signal_generator, val_data)
            cv_scores.append(score)

        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores
        }

    def get_feature_importance(self, study: optuna.Study) -> pd.DataFrame:
        """
        Calculate parameter importance from optimization study.

        Args:
            study: Completed Optuna study

        Returns:
            DataFrame with parameter importance scores
        """

        try:
            importance = optuna.importance.get_param_importances(study)
            importance_df = pd.DataFrame([
                {'parameter': param, 'importance': score}
                for param, score in importance.items()
            ]).sort_values('importance', ascending=False)

            return importance_df

        except Exception as e:
            self.logger.warning(f"Could not calculate parameter importance: {e}")
            return pd.DataFrame()