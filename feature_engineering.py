"""
Feature Engineering Module
Creates technical indicators and targets for all ETFs
"""

import pandas as pd
import numpy as np


def add_technical_indicators(df, ma_window=5):
    """
    Add technical indicators for each ETF
    
    Parameters:
    -----------
    df : pd.DataFrame
        Price data with ETF columns
    ma_window : int
        Moving average window (3 or 5)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added technical features
    """
    result = pd.DataFrame(index=df.index)
    
    etf_cols = [c for c in df.columns if c not in ['3MTBILL']]
    
    for col in etf_cols:
        # Price-based features
        result[f"{col}_RET"] = df[col].pct_change()
        result[f"{col}_MA{ma_window}"] = df[col].rolling(window=ma_window).mean()
        result[f"{col}_MA{ma_window}_DIFF"] = result[f"{col}_MA{ma_window}"].diff()
        result[f"{col}_EMA{ma_window}"] = df[col].ewm(span=ma_window).mean()
        result[f"{col}_EMA{ma_window}_DIFF"] = result[f"{col}_EMA{ma_window}"].diff()
        
        # Lagged features (past 5 days)
        for lag in range(1, 6):
            result[f"{col}_RET_LAG{lag}"] = result[f"{col}_RET"].shift(lag)
            result[f"{col}_MA_DIFF_LAG{lag}"] = result[f"{col}_MA{ma_window}_DIFF"].shift(lag)
        
        # Additional indicators
        result[f"{col}_VOL20"] = result[f"{col}_RET"].rolling(window=20).std()
        result[f"{col}_RSI"] = compute_rsi(df[col])
        result[f"{col}_BB_UPPER"], result[f"{col}_BB_LOWER"] = compute_bollinger_bands(df[col])
        result[f"{col}_MACD"], result[f"{col}_MACD_SIGNAL"] = compute_macd(df[col])
        result[f"{col}_ATR"] = compute_atr(df[col])
    
    # Add T-bill as feature
    if '3MTBILL' in df.columns:
        result['TBILL'] = df['3MTBILL']
    
    return result


def compute_rsi(series, window=14):
    """Compute Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_bollinger_bands(series, window=20, num_std=2):
    """Compute Bollinger Bands"""
    ma = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = ma + (std * num_std)
    lower = ma - (std * num_std)
    return upper, lower


def compute_macd(series, fast=12, slow=26, signal=9):
    """Compute MACD"""
    ema_fast = series.ewm(span=fast).mean()
    ema_slow = series.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal).mean()
    return macd, macd_signal


def compute_atr(series, window=14):
    """Compute Average True Range (simplified)"""
    high_low = series.diff().abs()
    atr = high_low.rolling(window=window).mean()
    return atr


def create_target(df, etf, ma_window=5):
    """
    Create target variable: MA_Diff(t+1) = MA(t+1) - MA(t)
    This is what we predict (next day's MA difference)
    """
    ma = df[etf].rolling(window=ma_window).mean()
    ma_diff = ma.diff()
    # Target is next day's MA_Diff (shift -1)
    target = ma_diff.shift(-1)
    return target


def prepare_all_features(df, ma_window=5):
    """
    Prepare complete feature set and targets for all ETFs
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw price data with ETF columns and 3MTBILL
    ma_window : int
        MA window (3 or 5)
    
    Returns:
    --------
    dict
        Dictionary with 'features' and 'targets' for each ETF
    """
    print(f"Preparing features for MA({ma_window})...")
    
    # Get ETF list (exclude T-bill)
    etf_list = [c for c in df.columns if c not in ['3MTBILL']]
    
    # Add technical indicators
    features_df = add_technical_indicators(df, ma_window)
    
    # Create targets for each ETF
    targets = {}
    features = {}
    
    for etf in etf_list:
        # Target: next day's MA_Diff
        target = create_target(df, etf, ma_window)
        
        # Features: all columns related to this ETF + macro features
        etf_feature_cols = [c for c in features_df.columns if etf in c or c == 'TBILL']
        X = features_df[etf_feature_cols].copy()
        
        # Add cross-ETF features (correlations, relative strength)
        for other_etf in etf_list:
            if other_etf != etf:
                X[f"REL_STRENGTH_{other_etf}"] = df[etf] / df[other_etf]
        
        # Align and drop NaN
        X['TARGET'] = target
        X = X.dropna()
        
        if len(X) > 100:  # Minimum data requirement
            features[etf] = X.drop('TARGET', axis=1)
            targets[etf] = X['TARGET']
            print(f"  {etf}: {len(X)} samples, {features[etf].shape[1]} features")
        else:
            print(f"  {etf}: Insufficient data ({len(X)} samples)")
    
    return {
        'features': features,
        'targets': targets,
        'full_features': features_df
    }


def z_score_normalize(df, train_mean=None, train_std=None):
    """Z-score normalization using training statistics"""
    if train_mean is None:
        train_mean = df.mean()
        train_std = df.std()
    
    normalized = (df - train_mean) / (train_std + 1e-10)
    return normalized, train_mean, train_std
