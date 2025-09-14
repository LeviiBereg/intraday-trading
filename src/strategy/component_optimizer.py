from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import optuna
from dataclasses import dataclass
import logging

from ..regime_detection.hmm import GaussianMixtureRegimeDetector, SimpleHMMRegimeDetector
from ..regime_detection.volatility_regime import VolatilityRegimeDetector
from ..indicators.technical import PivotPointIndicator
from ..indicators.breakout_probability import CatBoostBreakoutPredictor


@dataclass
class ComponentOptimizationResult:
    """Results from component-specific optimization."""
    component_name: str
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: pd.DataFrame


class ComponentOptimizer(ABC):
    """Base class for individual component optimization."""

    def __init__(self, component_name: str, n_trials: int = 50):
        self.component_name = component_name
        self.n_trials = n_trials
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define parameter search space for the component."""
        pass

    @abstractmethod
    def create_component(self, params: Dict[str, Any]) -> Any:
        """Create component instance with parameters."""
        pass

    @abstractmethod
    def evaluate_component(self, component: Any, data: pd.DataFrame) -> float:
        """Evaluate component performance."""
        pass

    def optimize_component(self, data: pd.DataFrame) -> ComponentOptimizationResult:
        """Run optimization for this component."""

        def objective(trial):
            params = self.define_search_space(trial)
            component = self.create_component(params)
            return self.evaluate_component(component, data)

        study = optuna.create_study(
            direction='maximize',
            study_name=f'{self.component_name}_optimization'
        )

        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        return ComponentOptimizationResult(
            component_name=self.component_name,
            best_params=study.best_params,
            best_score=study.best_value,
            optimization_history=study.trials_dataframe()
        )


class RegimeDetectionOptimizer(ComponentOptimizer):
    """Optimizer for regime detection components."""

    def __init__(self, regime_type: str = 'hmm', n_trials: int = 50):
        """
        Initialize regime detection optimizer.

        Args:
            regime_type: Type of regime detector ('hmm' or 'volatility')
            n_trials: Number of optimization trials
        """
        super().__init__(f'regime_detection_{regime_type}', n_trials)
        self.regime_type = regime_type

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define search space for regime detection parameters."""

        params = {}

        if self.regime_type == 'hmm':
            params['n_components'] = trial.suggest_int('n_components', 2, 4)
            params['covariance_type'] = trial.suggest_categorical(
                'covariance_type', ['spherical', 'diag', 'full']
            )
            params['n_iter'] = trial.suggest_int('n_iter', 50, 200, step=25)
            params['random_state'] = 42

        elif self.regime_type == 'volatility':
            params['short_window'] = trial.suggest_int('short_window', 10, 30)
            params['long_window'] = trial.suggest_int('long_window', 40, 100)
            params['threshold'] = trial.suggest_float('threshold', 1.2, 3.0)

        return params

    def create_component(self, params: Dict[str, Any]):
        """Create regime detection component."""

        if self.regime_type == 'hmm':
            return GaussianMixtureRegimeDetector(
                n_states=params['n_components'],
                n_components=params['n_components'],
                covariance_type=params['covariance_type'],
                max_iter=params['n_iter'],
                random_state=params['random_state']
            )

        elif self.regime_type == 'volatility':
            return VolatilityRegimeDetector(
                short_window=params['short_window'],
                long_window=params['long_window'],
                threshold=params['threshold']
            )

    def evaluate_component(self, component, data: pd.DataFrame) -> float:
        """Evaluate regime detection quality."""

        try:
            # Prepare features and fit the model
            features = component.prepare_features(data)
            component.fit(features)

            # Predict regimes
            regimes = component.predict_regime(features)
            regimes = pd.Series(regimes, index=data.index[-len(regimes):])

            # Calculate evaluation metrics
            regime_stability = self._calculate_regime_stability(regimes)
            regime_separation = self._calculate_regime_separation(regimes, data)
            regime_coverage = self._calculate_regime_coverage(regimes)

            # Composite score
            score = (
                0.4 * regime_stability +
                0.4 * regime_separation +
                0.2 * regime_coverage
            )

            return score

        except Exception as e:
            self.logger.warning(f"Regime detection evaluation failed: {e}")
            return 0.0

    def _calculate_regime_stability(self, regimes: pd.Series) -> float:
        """Calculate regime stability score."""
        if len(regimes) == 0:
            return 0.0

        # Count regime transitions
        transitions = (regimes != regimes.shift()).sum() - 1  # Subtract initial non-transition
        transition_rate = transitions / len(regimes)

        # Prefer moderate transition rates (not too stable, not too noisy)
        optimal_rate = 0.1  # 10% of periods should be transitions
        stability_score = 1 - abs(transition_rate - optimal_rate) / optimal_rate

        return max(0, stability_score)

    def _calculate_regime_separation(self, regimes: pd.Series, data: pd.DataFrame) -> float:
        """Calculate how well regimes separate different market conditions."""

        if len(regimes) == 0 or 'close' not in data.columns:
            return 0.0

        try:
            # Calculate volatility for each regime
            data_with_regimes = data.copy()
            data_with_regimes['regime'] = regimes
            data_with_regimes['returns'] = data_with_regimes['close'].pct_change()

            regime_volatilities = []
            for regime in regimes.unique():
                if pd.notna(regime):
                    regime_returns = data_with_regimes[data_with_regimes['regime'] == regime]['returns']
                    if len(regime_returns) > 1:
                        regime_volatilities.append(regime_returns.std())

            # Higher variance in volatilities across regimes is better
            if len(regime_volatilities) > 1:
                separation_score = np.std(regime_volatilities) / np.mean(regime_volatilities)
                return min(1.0, separation_score / 2.0)  # Normalize to 0-1
            else:
                return 0.0

        except Exception:
            return 0.0

    def _calculate_regime_coverage(self, regimes: pd.Series) -> float:
        """Calculate regime coverage balance."""

        if len(regimes) == 0:
            return 0.0

        # Count observations per regime
        regime_counts = regimes.value_counts()
        total_obs = len(regimes)

        # Calculate coverage balance (prefer balanced regimes)
        regime_proportions = regime_counts / total_obs

        # Entropy-based measure of balance
        entropy = -np.sum(regime_proportions * np.log(regime_proportions + 1e-8))
        max_entropy = np.log(len(regime_proportions))

        return entropy / max_entropy if max_entropy > 0 else 0.0


class SupportResistanceOptimizer(ComponentOptimizer):
    """Optimizer for support/resistance detection."""

    def __init__(self, n_trials: int = 50):
        super().__init__('support_resistance', n_trials)

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define search space for pivot point parameters."""

        return {
            'window': trial.suggest_int('window', 10, 50),
            'min_strength': trial.suggest_int('min_strength', 1, 5)
        }

    def create_component(self, params: Dict[str, Any]):
        """Create pivot point indicator."""

        return PivotPointIndicator(
            window=params['window'],
            min_strength=params['min_strength']
        )

    def evaluate_component(self, component: PivotPointIndicator, data: pd.DataFrame) -> float:
        """Evaluate support/resistance quality."""

        try:
            # Calculate pivot points and levels
            df_with_indicators = component.calculate(data)

            # Evaluate quality metrics
            level_accuracy = self._calculate_level_accuracy(df_with_indicators)
            range_quality = self._calculate_range_consistency(df_with_indicators)
            breakout_detection = self._calculate_breakout_quality(df_with_indicators)

            # Composite score
            score = (
                0.4 * level_accuracy +
                0.3 * range_quality +
                0.3 * breakout_detection
            )

            return score

        except Exception as e:
            self.logger.warning(f"Support/Resistance evaluation failed: {e}")
            return 0.0

    def _calculate_level_accuracy(self, data: pd.DataFrame) -> float:
        """Calculate how accurately levels predict price reactions."""

        if 'dynamic_support' not in data.columns or 'dynamic_resistance' not in data.columns:
            return 0.0

        try:
            # Test how often price bounces from S/R levels
            close_prices = data['close']
            support_levels = data['dynamic_support']
            resistance_levels = data['dynamic_resistance']

            # Calculate distances to S/R levels
            support_distances = (close_prices - support_levels) / close_prices
            resistance_distances = (resistance_levels - close_prices) / close_prices

            # Count successful bounces (price approaches level then reverses)
            bounce_threshold = 0.01  # 1% of price
            successful_bounces = 0
            total_tests = 0

            for i in range(1, len(data) - 1):
                # Test support bounces
                if (support_distances.iloc[i] <= bounce_threshold and
                    support_distances.iloc[i] >= 0):
                    total_tests += 1
                    if close_prices.iloc[i+1] > close_prices.iloc[i]:  # Price bounced up
                        successful_bounces += 1

                # Test resistance bounces
                if (resistance_distances.iloc[i] <= bounce_threshold and
                    resistance_distances.iloc[i] >= 0):
                    total_tests += 1
                    if close_prices.iloc[i+1] < close_prices.iloc[i]:  # Price bounced down
                        successful_bounces += 1

            return successful_bounces / total_tests if total_tests > 0 else 0.0

        except Exception:
            return 0.0

    def _calculate_range_consistency(self, data: pd.DataFrame) -> float:
        """Calculate consistency of range definitions."""

        if 'range_width' not in data.columns:
            return 0.0

        try:
            range_widths = data['range_width'].dropna()

            if len(range_widths) == 0:
                return 0.0

            # Measure consistency (lower variance is better)
            consistency_score = 1 / (1 + range_widths.std())

            # Prefer reasonable range widths
            avg_width = range_widths.mean()
            if 0.01 <= avg_width <= 0.05:  # 1-5% ranges are reasonable
                width_score = 1.0
            else:
                width_score = max(0, 1 - abs(avg_width - 0.03) / 0.03)

            return 0.6 * consistency_score + 0.4 * width_score

        except Exception:
            return 0.0

    def _calculate_breakout_quality(self, data: pd.DataFrame) -> float:
        """Calculate quality of breakout identification."""

        try:
            breakouts = data.get('upside_breakout', pd.Series(False, index=data.index)) | \
                       data.get('downside_breakout', pd.Series(False, index=data.index))

            if breakouts.sum() == 0:
                return 0.5  # No breakouts detected, neutral score

            # Evaluate if breakouts lead to continued moves
            successful_breakouts = 0
            total_breakouts = breakouts.sum()

            for i in range(len(data) - 5):  # Need 5 periods ahead
                if breakouts.iloc[i]:
                    # Check if trend continues for next 5 periods
                    current_price = data['close'].iloc[i]
                    future_prices = data['close'].iloc[i+1:i+6]

                    if len(future_prices) == 5:
                        # If breakout was upside
                        if data.get('upside_breakout', pd.Series(False)).iloc[i]:
                            if future_prices.mean() > current_price * 1.01:
                                successful_breakouts += 1
                        # If breakout was downside
                        elif data.get('downside_breakout', pd.Series(False)).iloc[i]:
                            if future_prices.mean() < current_price * 0.99:
                                successful_breakouts += 1

            return successful_breakouts / total_breakouts if total_breakouts > 0 else 0.5

        except Exception:
            return 0.5


class BreakoutPredictorOptimizer(ComponentOptimizer):
    """Optimizer for breakout prediction model."""

    def __init__(self, n_trials: int = 50):
        super().__init__('breakout_predictor', n_trials)

    def define_search_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Define search space for CatBoost parameters."""

        return {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10)
        }

    def create_component(self, params: Dict[str, Any]):
        """Create breakout predictor."""

        upward_params = {
            'iterations': params['iterations'],
            'depth': params['depth'],
            'learning_rate': params['learning_rate'],
            'l2_leaf_reg': params['l2_leaf_reg'],
            'verbose': False
        }

        downward_params = {
            'iterations': params['iterations'],
            'depth': params['depth'],
            'learning_rate': params['learning_rate'],
            'l2_leaf_reg': params['l2_leaf_reg'],
            'verbose': False
        }

        return CatBoostBreakoutPredictor(
            upward_model_params=upward_params,
            downward_model_params=downward_params
        )

    def evaluate_component(self, component: CatBoostBreakoutPredictor, data: pd.DataFrame) -> float:
        """Evaluate breakout prediction accuracy."""

        try:
            # Use the fit method for training and evaluation
            training_results = component.fit(data, test_size=0.2, validate_model=True)

            if not training_results:
                return 0.0

            # Get average performance across both models
            upward_metrics = training_results.get('upward_model', {})
            downward_metrics = training_results.get('downward_model', {})

            # Extract ROC AUC scores
            upward_auc = upward_metrics.get('roc_auc', 0.0)
            downward_auc = downward_metrics.get('roc_auc', 0.0)

            # Use average AUC as the primary metric
            avg_auc = (upward_auc + downward_auc) / 2

            # Also consider F1 scores for balanced evaluation
            upward_f1 = upward_metrics.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.0)
            downward_f1 = downward_metrics.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.0)
            avg_f1 = (upward_f1 + downward_f1) / 2

            # Composite score favoring AUC (better for probability predictions)
            score = 0.7 * avg_auc + 0.3 * avg_f1

            return score

        except Exception as e:
            self.logger.warning(f"Breakout predictor evaluation failed: {e}")
            return 0.0



class MultiComponentOptimizer:
    """Orchestrates optimization across multiple strategy components."""

    def __init__(self, n_trials_per_component: int = 50):
        self.n_trials_per_component = n_trials_per_component
        self.logger = logging.getLogger(__name__)

    def optimize_all_components(self, data: pd.DataFrame) -> Dict[str, ComponentOptimizationResult]:
        """Optimize all strategy components sequentially."""

        results = {}

        # Optimize regime detection (HMM)
        self.logger.info("Optimizing HMM regime detection...")
        hmm_optimizer = RegimeDetectionOptimizer('hmm', self.n_trials_per_component)
        results['hmm_regime'] = hmm_optimizer.optimize_component(data)

        # Optimize regime detection (Volatility)
        self.logger.info("Optimizing volatility regime detection...")
        vol_optimizer = RegimeDetectionOptimizer('volatility', self.n_trials_per_component)
        results['volatility_regime'] = vol_optimizer.optimize_component(data)

        # Optimize support/resistance detection
        self.logger.info("Optimizing support/resistance detection...")
        sr_optimizer = SupportResistanceOptimizer(self.n_trials_per_component)
        results['support_resistance'] = sr_optimizer.optimize_component(data)

        # Optimize breakout predictor
        self.logger.info("Optimizing breakout predictor...")
        breakout_optimizer = BreakoutPredictorOptimizer(self.n_trials_per_component)
        results['breakout_predictor'] = breakout_optimizer.optimize_component(data)

        return results

    def get_best_parameters_combination(self,
                                      component_results: Dict[str, ComponentOptimizationResult]) -> Dict[str, Any]:
        """Combine best parameters from all components."""

        combined_params = {}

        for component_name, result in component_results.items():
            prefix = f"{component_name}_"
            for param_name, param_value in result.best_params.items():
                combined_params[f"{prefix}{param_name}"] = param_value

        return combined_params