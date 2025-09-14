#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from src.data.data_loader import YahooDataLoader
from config.settings import DataConfig
from src.indicators.technical import PivotPointIndicator

plt.rcParams['figure.figsize'] = (18, 12)


# # Data Preparation

# In[2]:


data_config = DataConfig(
    timeframes=["1h"],
    symbols=["SPY"],
    cache_enabled=True,
    cache_dir="../data/cache"
)

data_loader = YahooDataLoader(data_config)

end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"Loading price data for SPY from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")


# In[ ]:


symbol = "SPY"
timeframe = "1h"

spy_prices_df = data_loader.load_data(
    symbol=symbol,
    start_date=start_date,
    end_date=end_date,
    timeframe=timeframe
)

print(f"Successfully loaded {len(spy_prices_df)} rows of {timeframe} data for {symbol}")
print(f"Date range: {spy_prices_df.index.min()} to {spy_prices_df.index.max()}")
print(f"\nData shape: {spy_prices_df.shape}")
print(f"\nColumns: {list(spy_prices_df.columns)}")


# In[8]:


spy_prices_df.head()


# In[9]:


spy_prices_df.describe()


# In[11]:


spy_returns_df = pd.DataFrame(index=spy_prices_df.index)

spy_returns_df['open_return'] = spy_prices_df['open'].pct_change() * 100
spy_returns_df['high_return'] = spy_prices_df['high'].pct_change() * 100
spy_returns_df['low_return'] = spy_prices_df['low'].pct_change() * 100
spy_returns_df['close_return'] = spy_prices_df['close'].pct_change() * 100

spy_returns_df['intrabar_return'] = ((spy_prices_df['high'] - spy_prices_df['low']) / spy_prices_df['open']) * 100

spy_returns_df['body_return'] = ((spy_prices_df['close'] - spy_prices_df['open']) / spy_prices_df['open']) * 100

spy_returns_df = spy_returns_df.dropna()

print(f"Returns data shape: {spy_returns_df.shape}")


# In[12]:


spy_returns_df.head()


# In[13]:


spy_returns_df.describe()


# ## Prices Analysis

# In[14]:


fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'{symbol} Price Data - 1H Timeframe', fontsize=16, fontweight='bold')

axes[0, 0].plot(spy_prices_df.index, spy_prices_df['close'], label='Close', alpha=0.8, linewidth=1)
axes[0, 0].fill_between(spy_prices_df.index, spy_prices_df['low'], spy_prices_df['high'], 
                       alpha=0.3, label='High-Low Range')
axes[0, 0].set_title('Price Evolution Over Time')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].bar(spy_prices_df.index, spy_prices_df['volume'], alpha=0.6, width=0.02)
axes[0, 1].set_title('Trading Volume Over Time')
axes[0, 1].set_ylabel('Volume')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(spy_prices_df['close'], bins=50, alpha=0.7, edgecolor='black')
axes[1, 0].axvline(spy_prices_df['close'].mean(), color='red', linestyle='--', 
                  label=f'Mean: ${spy_prices_df["close"].mean():.2f}')
axes[1, 0].axvline(spy_prices_df['close'].median(), color='orange', linestyle='--', 
                  label=f'Median: ${spy_prices_df["close"].median():.2f}')
axes[1, 0].set_title('Close Price Distribution')
axes[1, 0].set_xlabel('Price ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

price_range = spy_prices_df['high'] - spy_prices_df['low']
axes[1, 1].hist(price_range, bins=50, alpha=0.7, edgecolor='black')
axes[1, 1].axvline(price_range.mean(), color='red', linestyle='--', 
                  label=f'Mean Range: ${price_range.mean():.2f}')
axes[1, 1].set_title('Intrabar Price Range Distribution (High-Low)')
axes[1, 1].set_xlabel('Price Range ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[15]:


print(f"Price Statistics for {symbol}:")
print(f"Close Price - Mean: ${spy_prices_df['close'].mean():.2f}, Std: ${spy_prices_df['close'].std():.2f}")
print(f"Price Range - Mean: ${price_range.mean():.2f}, Std: ${price_range.std():.2f}")
print(f"Volume - Mean: {spy_prices_df['volume'].mean():,.0f}, Std: {spy_prices_df['volume'].std():,.0f}")


# ## Returns Analysis

# In[16]:


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'{symbol} Returns Analysis - 1H Timeframe', fontsize=16, fontweight='bold')

axes[0, 0].plot(spy_returns_df.index, spy_returns_df['close_return'], alpha=0.7, linewidth=0.8)
axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.8)
axes[0, 0].set_title('Close Returns Over Time')
axes[0, 0].set_ylabel('Return (%)')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(spy_returns_df['close_return'], bins=50, alpha=0.7, edgecolor='black')
axes[0, 1].axvline(spy_returns_df['close_return'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {spy_returns_df["close_return"].mean():.3f}%')
axes[0, 1].axvline(0, color='orange', linestyle='--', alpha=0.8)
axes[0, 1].set_title('Close Returns Distribution')
axes[0, 1].set_xlabel('Return (%)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

from scipy import stats
stats.probplot(spy_returns_df['close_return'].dropna(), dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Close Returns Q-Q Plot (Normal)')
axes[0, 2].grid(True, alpha=0.3)

axes[1, 0].hist(spy_returns_df['body_return'], bins=50, alpha=0.7, edgecolor='black', color='green')
axes[1, 0].axvline(spy_returns_df['body_return'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {spy_returns_df["body_return"].mean():.3f}%')
axes[1, 0].axvline(0, color='orange', linestyle='--', alpha=0.8)
axes[1, 0].set_title('Body Returns Distribution (Close-Open)')
axes[1, 0].set_xlabel('Return (%)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(spy_returns_df['intrabar_return'], bins=50, alpha=0.7, edgecolor='black', color='purple')
axes[1, 1].axvline(spy_returns_df['intrabar_return'].mean(), color='red', linestyle='--', 
                  label=f'Mean: {spy_returns_df["intrabar_return"].mean():.3f}%')
axes[1, 1].set_title('Intrabar Range Distribution (High-Low/Open)')
axes[1, 1].set_xlabel('Range (%)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

rolling_vol = spy_returns_df['close_return'].rolling(window=20).std()
axes[1, 2].plot(rolling_vol.index, rolling_vol, alpha=0.8, color='red')
axes[1, 2].set_title('Rolling Volatility (20-period)')
axes[1, 2].set_ylabel('Volatility (%)')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[19]:


close_returns = spy_returns_df['close_return']
body_returns = spy_returns_df['body_return']
intrabar_returns = spy_returns_df['intrabar_return']


# In[20]:


print(f"1. CLOSE RETURNS STATISTICS:")
print(f"   Mean Return: {close_returns.mean():.4f}%")
print(f"   Std Deviation: {close_returns.std():.4f}%")
print(f"   Skewness: {close_returns.skew():.4f}")
print(f"   Kurtosis: {close_returns.kurtosis():.4f}")
print(f"   Min Return: {close_returns.min():.4f}%")
print(f"   Max Return: {close_returns.max():.4f}%")
print(f"   Sharpe Ratio (assuming 252*24 periods/year): {(close_returns.mean() / close_returns.std()) * np.sqrt(252*24):.4f}")


# In[21]:


print(f"2. BODY RETURNS STATISTICS (Close-Open):")
print(f"   Mean Return: {body_returns.mean():.4f}%")
print(f"   Std Deviation: {body_returns.std():.4f}%")
print(f"   Skewness: {body_returns.skew():.4f}")
print(f"   Kurtosis: {body_returns.kurtosis():.4f}")


# In[22]:


print(f"3. INTRABAR RANGE STATISTICS (High-Low/Open):")
print(f"   Mean Range: {intrabar_returns.mean():.4f}%")
print(f"   Std Deviation: {intrabar_returns.std():.4f}%")
print(f"   Min Range: {intrabar_returns.min():.4f}%")
print(f"   Max Range: {intrabar_returns.max():.4f}%")


# In[23]:


print(f"4. RISK METRICS:")
print(f"   Value at Risk (1%): {np.percentile(close_returns, 1):.4f}%")
print(f"   Value at Risk (5%): {np.percentile(close_returns, 5):.4f}%")
print(f"   Expected Shortfall (1%): {close_returns[close_returns <= np.percentile(close_returns, 1)].mean():.4f}%")
print(f"   Expected Shortfall (5%): {close_returns[close_returns <= np.percentile(close_returns, 5)].mean():.4f}%")


# In[24]:


print(f"5. DISTRIBUTION ANALYSIS:")
positive_returns = (close_returns > 0).sum()
negative_returns = (close_returns < 0).sum()
zero_returns = (close_returns == 0).sum()
total_returns = len(close_returns)

print(f"   Positive Returns: {positive_returns} ({positive_returns/total_returns*100:.1f}%)")
print(f"   Negative Returns: {negative_returns} ({negative_returns/total_returns*100:.1f}%)")
print(f"   Zero Returns: {zero_returns} ({zero_returns/total_returns*100:.1f}%)")
print(f"   Win Rate: {positive_returns/(positive_returns+negative_returns)*100:.1f}%")

if positive_returns > 0 and negative_returns > 0:
    avg_win = close_returns[close_returns > 0].mean()
    avg_loss = close_returns[close_returns < 0].mean()
    profit_factor = abs(avg_win / avg_loss)
    print(f"   Average Win: {avg_win:.4f}%")
    print(f"   Average Loss: {avg_loss:.4f}%")
    print(f"   Profit Factor: {profit_factor:.4f}")


# # Regime Detection

# In[29]:


from src.regime_detection.hmm import GaussianMixtureRegimeDetector


# We identify three market regimes:
# - Bullish breakout: Strong upward momentum with high volatility
# - Bearish breakout: Strong downward momentum with high volatility
# - Trading range: Low volatility sideways movement

# In[31]:


regime_detector = GaussianMixtureRegimeDetector(n_states=3, random_state=13)


# In[33]:


hmm_features = regime_detector.prepare_features(spy_prices_df)

print(f"Feature matrix shape: {hmm_features.shape}")
print(f"Features include: returns, volatility, volume ratios, price ratios, RSI, momentum")
print(f"Training period: {len(hmm_features)} observations")


# In[35]:


regime_detector.fit(hmm_features)
regime_characteristics = regime_detector.get_regime_characteristics()

print("\nRegime Characteristics:")
for regime_id, characteristics in regime_characteristics.items():
    print(f"\n{regime_id.upper()}:")
    print(f"  Weight: {characteristics['weight']:.3f}")
    print(f"  Mean Return: {characteristics['features']['returns']:.4f}")
    print(f"  Volatility: {characteristics['features']['volatility']:.4f}")
    print(f"  Volume Ratio: {characteristics['features']['volume_ratio']:.3f}")
    print(f"  RSI: {characteristics['features']['rsi']:.1f}")
    print(f"  Momentum: {characteristics['features']['momentum']:.4f}")


# In[36]:


regime_predictions = regime_detector.predict_regime(hmm_features)
regime_probabilities = regime_detector.get_regime_probabilities(hmm_features)

print(f"Predicted {len(regime_predictions)} regime states")

regime_start_idx = len(spy_prices_df) - len(regime_predictions)
regime_df = pd.DataFrame(index=spy_prices_df.index[regime_start_idx:])
regime_df['regime_state'] = regime_predictions
regime_df['regime_prob_0'] = regime_probabilities[:, 0]
regime_df['regime_prob_1'] = regime_probabilities[:, 1]
regime_df['regime_prob_2'] = regime_probabilities[:, 2]

print("\nRegime distribution:")
regime_counts = pd.Series(regime_predictions).value_counts().sort_index()
for state, count in regime_counts.items():
    pct = count / len(regime_predictions) * 100
    print(f"  Regime {state}: {count} observations ({pct:.1f}%)")


# In[37]:


def map_regimes_to_labels(characteristics):
    regime_labels = {}

    for regime_id, props in characteristics.items():
        features = props['features']
        returns = features['returns']
        volatility = features['volatility']
        momentum = features['momentum']

        if volatility > 0.015 and returns > 0.003 and momentum > 0.01:
            label = "bullish_breakout"
        elif volatility > 0.015 and returns < -0.003 and momentum < -0.01:
            label = "bearish_breakout"
        elif volatility < 0.012 and abs(returns) < 0.002:
            label = "trading_range"
        elif returns > 0:
            label = "bullish_trend"
        else:
            label = "bearish_trend"

        regime_labels[regime_id] = label

    return regime_labels

regime_labels = map_regimes_to_labels(regime_characteristics)
print("\nRegime Label Mapping:")
for regime_id, label in regime_labels.items():
    print(f"  {regime_id} -> {label}")

state_to_label = {i: regime_labels[f'regime_{i}'] for i in range(3)}
regime_df['regime_label'] = regime_df['regime_state'].map(state_to_label)

print("\nRegime distribution by label:")
label_counts = regime_df['regime_label'].value_counts()
for label, count in label_counts.items():
    pct = count / len(regime_df) * 100
    print(f"  {label}: {count} observations ({pct:.1f}%)")


# In[38]:


fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('HMM Regime Detection Analysis', fontsize=16, fontweight='bold')

ax = axes[0, 0]
regime_colors = {'bullish_breakout': 'green', 'bearish_breakout': 'red', 'trading_range': 'blue',
                'bullish_trend': 'lightgreen', 'bearish_trend': 'lightcoral'}

price_regime_df = spy_prices_df.iloc[regime_start_idx:].copy()
price_regime_df['regime_label'] = regime_df['regime_label']

for label, color in regime_colors.items():
    mask = price_regime_df['regime_label'] == label
    if mask.any():
        ax.scatter(price_regime_df[mask].index, price_regime_df[mask]['close'],
                  c=color, label=label, alpha=0.6, s=10)

ax.set_title('Price Evolution by Regime')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(regime_df.index, regime_df['regime_state'], linewidth=2)
ax.set_title('Regime State Transitions Over Time')
ax.set_ylabel('Regime State')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['State 0', 'State 1', 'State 2'])
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
prob_matrix = regime_df[['regime_prob_0', 'regime_prob_1', 'regime_prob_2']].values.T
im = ax.imshow(prob_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
ax.set_title('Regime Probability Evolution')
ax.set_ylabel('Regime State')
ax.set_xlabel('Time')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['State 0', 'State 1', 'State 2'])
plt.colorbar(im, ax=ax, label='Probability')

ax = axes[1, 1]
returns_regime_df = spy_returns_df.iloc[regime_start_idx:].copy()
returns_regime_df['regime_label'] = regime_df['regime_label']

regime_returns = []
regime_names = []
for label in regime_labels.values():
    if label in returns_regime_df['regime_label'].values:
        regime_returns.append(returns_regime_df[returns_regime_df['regime_label'] == label]['close_return'].values)
        regime_names.append(label)

ax.boxplot(regime_returns, labels=regime_names)
ax.set_title('Returns Distribution by Regime')
ax.set_ylabel('Return (%)')
ax.tick_params(axis='x', rotation=45)
ax.grid(True, alpha=0.3)

ax = axes[2, 0]
for label in regime_labels.values():
    if label in returns_regime_df['regime_label'].values:
        regime_data = returns_regime_df[returns_regime_df['regime_label'] == label]
        rolling_vol = regime_data['close_return'].rolling(window=20).std()
        ax.plot(regime_data.index, rolling_vol, label=label, alpha=0.7)

ax.set_title('Rolling Volatility by Regime')
ax.set_ylabel('Volatility (%)')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
regime_changes = regime_df['regime_state'].diff().fillna(0) != 0
regime_durations = []
current_regime = regime_df['regime_state'].iloc[0]
duration = 1

for i in range(1, len(regime_df)):
    if regime_df['regime_state'].iloc[i] == current_regime:
        duration += 1
    else:
        regime_durations.append(duration)
        current_regime = regime_df['regime_state'].iloc[i]
        duration = 1
regime_durations.append(duration)

ax.hist(regime_durations, bins=30, alpha=0.7, edgecolor='black')
ax.set_title('Regime Duration Distribution')
ax.set_xlabel('Duration (hours)')
ax.set_ylabel('Frequency')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[39]:


for label in regime_labels.values():
    if label in returns_regime_df['regime_label'].values:
        regime_data = returns_regime_df[returns_regime_df['regime_label'] == label]

        if len(regime_data) > 0:
            print(f"\n{label.upper().replace('_', ' ')}:")
            print(f"  Observations: {len(regime_data)}")
            print(f"  Mean Return: {regime_data['close_return'].mean():.4f}%")
            print(f"  Std Deviation: {regime_data['close_return'].std():.4f}%")
            print(f"  Skewness: {regime_data['close_return'].skew():.4f}")
            print(f"  Win Rate: {(regime_data['close_return'] > 0).mean()*100:.1f}%")

            if len(regime_data) > 1:
                sharpe = (regime_data['close_return'].mean() / regime_data['close_return'].std()) * np.sqrt(252*24)
                print(f"  Sharpe Ratio: {sharpe:.4f}")

print(f"\nMean regime duration: {np.mean(regime_durations):.1f} hours")
print(f"Median regime duration: {np.median(regime_durations):.1f} hours")
print(f"Max regime duration: {np.max(regime_durations)} hours")


# # Support and Resistance Detection
#
# This section implements support and resistance level detection for trading range regimes.
# The system identifies key levels that bound the trading ranges and validates them with volume analysis.

# In[40]:


# Initialize pivot point detector for support/resistance identification
pivot_detector = PivotPointIndicator(window=20, min_strength=2)

print("Initializing Support & Resistance Detection...")
print(f"Pivot detector parameters: window={pivot_detector.window}, min_strength={pivot_detector.min_strength}")


# In[41]:


# Calculate pivot points and support/resistance levels
print("Detecting pivot points and support/resistance levels...")
sr_data = pivot_detector.calculate(spy_prices_df)

print(f"Calculated pivot points for {len(sr_data)} price observations")
print(f"Pivot highs detected: {sr_data['pivot_high'].notna().sum()}")
print(f"Pivot lows detected: {sr_data['pivot_low'].notna().sum()}")

# Display recent support/resistance levels
recent_data = sr_data.tail(10)
print(f"\nRecent Support/Resistance Levels:")
print(f"Dynamic Support: ${recent_data['dynamic_support'].iloc[-1]:.2f}")
print(f"Dynamic Resistance: ${recent_data['dynamic_resistance'].iloc[-1]:.2f}")
print(f"Range Width: {recent_data['range_width'].iloc[-1]:.1%}")
print(f"Current Range Position: {recent_data['range_position'].iloc[-1]:.1%}")


# In[42]:


# Focus on trading range regime periods for support/resistance validation
if len(regime_df) > 0:
    # Align support/resistance data with regime data
    sr_regime_df = sr_data.iloc[regime_start_idx:].copy()
    sr_regime_df['regime_label'] = regime_df['regime_label']

    # Filter for trading range periods
    trading_range_periods = sr_regime_df[sr_regime_df['regime_label'] == 'trading_range']

    print(f"\\nTrading Range Analysis:")
    print(f"Trading range periods: {len(trading_range_periods)} observations")

    if len(trading_range_periods) > 0:
        print(f"Average range width in trading ranges: {trading_range_periods['range_width'].mean():.1%}")
        print(f"Average range position distribution: {trading_range_periods['range_position'].mean():.1%}")

        # Calculate range quality for trading range periods
        range_quality = pivot_detector.get_range_quality(spy_prices_df.iloc[regime_start_idx:])
        trading_range_quality = range_quality[sr_regime_df['regime_label'] == 'trading_range']

        print(f"Average range quality score: {trading_range_quality.mean():.3f}")
        print(f"High quality ranges (>0.7): {(trading_range_quality > 0.7).sum()} periods")
else:
    print("No regime data available for support/resistance validation")


# In[43]:


# Volume-weighted support/resistance validation
def validate_sr_levels_with_volume(price_data, sr_data, volume_threshold_multiplier=1.5):
    """Validate support/resistance levels using volume analysis"""

    validated_sr = sr_data.copy()

    # Calculate volume moving average
    volume_ma = price_data['volume'].rolling(window=20).mean()
    high_volume_threshold = volume_ma * volume_threshold_multiplier

    # Volume validation for support levels
    support_tests = []
    resistance_tests = []

    for i in range(20, len(price_data)):
        current_support = sr_data['dynamic_support'].iloc[i]
        current_resistance = sr_data['dynamic_resistance'].iloc[i]

        # Check for support tests (price approaches support with volume)
        recent_data = price_data.iloc[i-5:i+1]
        support_distance = abs(recent_data['low'] - current_support) / current_support
        support_test = (support_distance < 0.01) & (recent_data['volume'] > high_volume_threshold.iloc[i-5:i+1])

        # Check for resistance tests (price approaches resistance with volume)
        resistance_distance = abs(recent_data['high'] - current_resistance) / current_resistance
        resistance_test = (resistance_distance < 0.01) & (recent_data['volume'] > high_volume_threshold.iloc[i-5:i+1])

        support_tests.append(support_test.any())
        resistance_tests.append(resistance_test.any())

    # Add validation columns
    validated_sr['support_volume_validated'] = [False] * 20 + support_tests
    validated_sr['resistance_volume_validated'] = [False] * 20 + resistance_tests

    # Calculate validation scores
    validated_sr['sr_validation_score'] = (
        validated_sr['support_volume_validated'].astype(int) +
        validated_sr['resistance_volume_validated'].astype(int)
    ) / 2

    return validated_sr

print("Validating support/resistance levels with volume analysis...")
validated_sr_data = validate_sr_levels_with_volume(spy_prices_df, sr_data)

# Summary of volume validation
support_validated = validated_sr_data['support_volume_validated'].sum()
resistance_validated = validated_sr_data['resistance_volume_validated'].sum()
total_observations = len(validated_sr_data) - 20

print(f"Volume validation results:")
print(f"Support levels validated by volume: {support_validated}/{total_observations} ({support_validated/total_observations*100:.1f}%)")
print(f"Resistance levels validated by volume: {resistance_validated}/{total_observations} ({resistance_validated/total_observations*100:.1f}%)")
print(f"Average validation score: {validated_sr_data['sr_validation_score'].mean():.3f}")


# In[44]:


# Support and Resistance visualization
fig, axes = plt.subplots(3, 2, figsize=(18, 15))
fig.suptitle('Support & Resistance Analysis', fontsize=16, fontweight='bold')

# Price action with support/resistance levels
ax = axes[0, 0]
recent_periods = 500  # Show last 500 periods for clarity
recent_sr_data = validated_sr_data.tail(recent_periods)
recent_price_data = spy_prices_df.tail(recent_periods)

ax.plot(recent_price_data.index, recent_price_data['close'], label='Close Price', linewidth=1.5, alpha=0.8)
ax.plot(recent_sr_data.index, recent_sr_data['dynamic_support'], label='Dynamic Support',
        color='green', linestyle='--', alpha=0.7)
ax.plot(recent_sr_data.index, recent_sr_data['dynamic_resistance'], label='Dynamic Resistance',
        color='red', linestyle='--', alpha=0.7)

# Highlight volume-validated levels
support_validated_mask = recent_sr_data['support_volume_validated']
resistance_validated_mask = recent_sr_data['resistance_volume_validated']

ax.scatter(recent_sr_data[support_validated_mask].index,
          recent_sr_data[support_validated_mask]['dynamic_support'],
          color='green', marker='o', s=30, alpha=0.8, label='Volume Validated Support')
ax.scatter(recent_sr_data[resistance_validated_mask].index,
          recent_sr_data[resistance_validated_mask]['dynamic_resistance'],
          color='red', marker='o', s=30, alpha=0.8, label='Volume Validated Resistance')

ax.set_title('Price with Support & Resistance Levels')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)

# Range position over time
ax = axes[0, 1]
ax.plot(recent_sr_data.index, recent_sr_data['range_position'], linewidth=2, alpha=0.8)
ax.axhline(y=0.2, color='green', linestyle=':', alpha=0.7, label='Buy Zone (20%)')
ax.axhline(y=0.8, color='red', linestyle=':', alpha=0.7, label='Sell Zone (80%)')
ax.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Range Middle')
ax.set_title('Range Position Over Time')
ax.set_ylabel('Range Position')
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)

# Range width distribution
ax = axes[1, 0]
ax.hist(validated_sr_data['range_width'].dropna(), bins=50, alpha=0.7, edgecolor='black')
ax.axvline(validated_sr_data['range_width'].mean(), color='red', linestyle='--',
          label=f'Mean: {validated_sr_data["range_width"].mean():.1%}')
ax.set_title('Range Width Distribution')
ax.set_xlabel('Range Width (%)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# Range quality over time
ax = axes[1, 1]
range_quality_all = pivot_detector.get_range_quality(spy_prices_df)
ax.plot(range_quality_all.index, range_quality_all, alpha=0.8, linewidth=1)
ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='High Quality Threshold')
ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Medium Quality Threshold')
ax.set_title('Range Quality Score Over Time')
ax.set_ylabel('Quality Score')
ax.legend()
ax.grid(True, alpha=0.3)

# Volume validation heatmap
ax = axes[2, 0]
validation_matrix = np.column_stack([
    validated_sr_data['support_volume_validated'].astype(int),
    validated_sr_data['resistance_volume_validated'].astype(int)
]).T

im = ax.imshow(validation_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
ax.set_title('Volume Validation Heatmap')
ax.set_ylabel('Level Type')
ax.set_xlabel('Time')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Support', 'Resistance'])
plt.colorbar(im, ax=ax, label='Validated (1) / Not Validated (0)')

# Pivot points scatter plot
ax = axes[2, 1]
pivot_highs = validated_sr_data['pivot_high'].dropna()
pivot_lows = validated_sr_data['pivot_low'].dropna()

ax.scatter(pivot_lows.index, pivot_lows.values, color='green', marker='^',
          s=50, alpha=0.7, label=f'Pivot Lows ({len(pivot_lows)})')
ax.scatter(pivot_highs.index, pivot_highs.values, color='red', marker='v',
          s=50, alpha=0.7, label=f'Pivot Highs ({len(pivot_highs)})')

# Overlay price for context
ax.plot(recent_price_data.index, recent_price_data['close'],
       color='blue', alpha=0.3, linewidth=1)

ax.set_title('Detected Pivot Points')
ax.set_ylabel('Price ($)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# In[45]:


# Integration with regime detection - Trading range specific analysis
print("="*60)
print("SUPPORT & RESISTANCE ANALYSIS FOR TRADING RANGES")
print("="*60)

if len(regime_df) > 0 and 'trading_range' in regime_df['regime_label'].values:

    # Filter for trading range periods only
    sr_regime_aligned = validated_sr_data.iloc[regime_start_idx:].copy()
    sr_regime_aligned['regime_label'] = regime_df['regime_label']

    trading_range_sr = sr_regime_aligned[sr_regime_aligned['regime_label'] == 'trading_range']

    print(f"\\nTrading Range Regime Analysis:")
    print(f"Total trading range periods: {len(trading_range_sr)}")

    if len(trading_range_sr) > 0:
        print(f"\\nSupport & Resistance Characteristics in Trading Ranges:")
        print(f"Average range width: {trading_range_sr['range_width'].mean():.2%}")
        print(f"Median range width: {trading_range_sr['range_width'].median():.2%}")
        print(f"Range width std dev: {trading_range_sr['range_width'].std():.2%}")

        print(f"\\nRange Position Distribution:")
        position_stats = trading_range_sr['range_position'].describe()
        for stat, value in position_stats.items():
            print(f"  {stat}: {value:.3f}")

        print(f"\\nVolume Validation in Trading Ranges:")
        tr_support_validated = trading_range_sr['support_volume_validated'].sum()
        tr_resistance_validated = trading_range_sr['resistance_volume_validated'].sum()
        print(f"Support validations: {tr_support_validated}/{len(trading_range_sr)} ({tr_support_validated/len(trading_range_sr)*100:.1f}%)")
        print(f"Resistance validations: {tr_resistance_validated}/{len(trading_range_sr)} ({tr_resistance_validated/len(trading_range_sr)*100:.1f}%)")

        # Range quality in trading ranges
        tr_range_quality = pivot_detector.get_range_quality(spy_prices_df.iloc[regime_start_idx:])
        tr_quality_in_ranges = tr_range_quality[sr_regime_aligned['regime_label'] == 'trading_range']

        print(f"\\nRange Quality in Trading Ranges:")
        print(f"Average quality score: {tr_quality_in_ranges.mean():.3f}")
        print(f"High quality periods (>0.7): {(tr_quality_in_ranges > 0.7).sum()}/{len(tr_quality_in_ranges)} ({(tr_quality_in_ranges > 0.7).sum()/len(tr_quality_in_ranges)*100:.1f}%)")

        # Identify best trading opportunities
        high_quality_ranges = trading_range_sr[
            (tr_quality_in_ranges > 0.7) &
            (trading_range_sr['range_width'] > 0.02) &
            ((trading_range_sr['support_volume_validated']) | (trading_range_sr['resistance_volume_validated']))
        ]

        print(f"\\nHigh-Quality Trading Opportunities:")
        print(f"Periods meeting all criteria: {len(high_quality_ranges)}")
        if len(high_quality_ranges) > 0:
            print(f"Average range width in best opportunities: {high_quality_ranges['range_width'].mean():.2%}")
            print(f"Average validation score: {high_quality_ranges['sr_validation_score'].mean():.3f}")

else:
    print("No trading range regime periods detected for analysis")

print("\\n" + "="*60)


# In[46]:


# Export support/resistance data for strategy implementation
print("Preparing support/resistance data for strategy implementation...")

# Create final dataset with all indicators
final_sr_dataset = validated_sr_data.copy()

# Add regime information if available
if len(regime_df) > 0:
    regime_aligned = pd.DataFrame(index=final_sr_dataset.index)
    regime_aligned.loc[regime_df.index, 'regime_label'] = regime_df['regime_label']
    regime_aligned.loc[regime_df.index, 'regime_state'] = regime_df['regime_state']
    regime_aligned = regime_aligned.fillna('unknown')

    final_sr_dataset['regime_label'] = regime_aligned['regime_label']
    final_sr_dataset['regime_state'] = regime_aligned['regime_state']

# Add trading signals from pivot detector
trading_signals = pivot_detector.get_signals(spy_prices_df)
final_sr_dataset['pivot_signals'] = trading_signals

# Add breakout detection
breakout_signals = pivot_detector.identify_breakouts(spy_prices_df, breakout_threshold=0.015)
final_sr_dataset['upside_breakout'] = breakout_signals['upside_breakout']
final_sr_dataset['downside_breakout'] = breakout_signals['downside_breakout']

# Summary of final dataset
print(f"\\nFinal Support/Resistance Dataset:")
print(f"Total observations: {len(final_sr_dataset)}")
print(f"Columns: {len(final_sr_dataset.columns)}")

# Key statistics
recent_final = final_sr_dataset.tail(100)
print(f"\\nRecent Performance (Last 100 periods):")
print(f"Trading signals generated: {(recent_final['pivot_signals'] != 0).sum()}")
print(f"Upside breakouts detected: {recent_final['upside_breakout'].sum()}")
print(f"Downside breakouts detected: {recent_final['downside_breakout'].sum()}")
print(f"Average range quality: {pivot_detector.get_range_quality(spy_prices_df).tail(100).mean():.3f}")

print(f"\\nSupport/Resistance detection implementation complete!")


# In[47]:


get_ipython().system('jupyter nbconvert --to script workflow.ipynb')

