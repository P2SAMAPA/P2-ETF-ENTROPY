import pandas as pd
import numpy as np

TARGET_ETFS    = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]
BENCHMARK_ETFS = ["SPY", "AGG"]
TRAIN_PCT = 0.80
VAL_PCT   = 0.10


def compute_rsi(series, window=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def compute_bollinger_bands(series, window=20, num_std=2):
    ma  = series.rolling(window).mean()
    std = series.rolling(window).std()
    return ma + std * num_std, ma - std * num_std

def compute_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    return macd, macd.ewm(span=signal, adjust=False).mean()

def compute_atr(series, window=14):
    return series.diff().abs().rolling(window).mean()


def add_technical_indicators(df, ma_window=5):
    result   = pd.DataFrame(index=df.index)
    etf_cols = [c for c in df.columns if c in TARGET_ETFS]
    for col in etf_cols:
        ret = df[col].pct_change()
        result[f"{col}_RET"] = ret
        for w in [3, 5]:
            ma   = df[col].rolling(w).mean()
            diff = ma.diff()
            result[f"{col}_MA{w}"]      = ma
            result[f"{col}_MA{w}_DIFF"] = diff
            for lag in range(1, 6):
                result[f"{col}_MA{w}_DIFF_LAG{lag}"] = diff.shift(lag)
        for lag in range(1, 6):
            result[f"{col}_RET_LAG{lag}"] = ret.shift(lag)
        result[f"{col}_VOL20"]       = ret.rolling(20).std()
        result[f"{col}_RSI"]         = compute_rsi(df[col])
        bb_u, bb_l                   = compute_bollinger_bands(df[col])
        result[f"{col}_BB_UPPER"]    = bb_u
        result[f"{col}_BB_LOWER"]    = bb_l
        macd, macd_sig               = compute_macd(df[col])
        result[f"{col}_MACD"]        = macd
        result[f"{col}_MACD_SIGNAL"] = macd_sig
        result[f"{col}_ATR"]         = compute_atr(df[col])
    for col in etf_cols:
        for other in etf_cols:
            if other != col:
                result[f"{col}_REL_{other}"] = df[col] / (df[other] + 1e-10)
    if "3MTBILL" in df.columns:
        result["TBILL"] = df["3MTBILL"]
    return result


def create_target(df, etf, ma_window=5):
    ma = df[etf].rolling(ma_window).mean()
    return ma.diff().shift(-1)


def compute_split_indices(n):
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * (TRAIN_PCT + VAL_PCT))
    return train_end, val_end


def prepare_all_features(df, ma_window=5, year_start=2008):
    """Prepare features with 80/10/10 split starting from year_start."""
    # Filter data from year_start onwards
    df_filtered = df[df.index >= f"{year_start}-01-01"].copy()
    
    if len(df_filtered) < 100:  # Minimum data check
        return {"features": {}, "split_dates": {}, "features_test": {}}
    
    # Add technical indicators
    features_df = add_technical_indicators(df_filtered, ma_window)
    
    # Remove rows with NaN values
    features_df = features_df.dropna()
    
    # Align with original df to get targets
    df_aligned = df_filtered.loc[features_df.index]
    
    # Build features dict per ETF
    features = {}
    targets = {}
    for etf in TARGET_ETFS:
        if etf not in df_aligned.columns:
            continue
            
        # Feature columns for this ETF
        etf_feature_cols = [c for c in features_df.columns if c.startswith(f"{etf}_") or c == "TBILL"]
        if len(etf_feature_cols) == 0:
            continue
            
        X = features_df[etf_feature_cols]
        y = create_target(df_aligned, etf, ma_window)
        
        # Align X and y
        common_idx = X.index.intersection(y.dropna().index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 50:
            continue
            
        features[etf] = X
        targets[etf] = y
    
    if not features:
        return {"features": {}, "split_dates": {}, "features_test": {}}
    
    # Compute 80/10/10 split indices
    n = len(features[TARGET_ETFS[0]])
    train_end, val_end = compute_split_indices(n)
    
    # Split features and compute dates
    split_dates = {}
    features_test = {}
    
    for etf in features:
        X = features[etf]
        
        # Get dates for splits
        all_dates = X.index
        train_dates = all_dates[:train_end]
        val_dates = all_dates[train_end:val_end]
        test_dates = all_dates[val_end:]
        
        split_dates[etf] = {
            "train_end": train_dates[-1] if len(train_dates) > 0 else None,
            "val_end": val_dates[-1] if len(val_dates) > 0 else None,
            "oos_start": test_dates[0] if len(test_dates) > 0 else None,
            "oos_end": test_dates[-1] if len(test_dates) > 0 else None,
        }
        
        # Store test features for get_oos_dates to use
        features_test[etf] = X.iloc[val_end:]
    
    return {
        "features": features,
        "targets": targets,
        "split_dates": split_dates,
        "features_test": features_test,
        "train_end": train_end,
        "val_end": val_end,
    }


def get_oos_dates(data_dict):
    """Return (oos_start, oos_end) as pd.Timestamps from prepared data_dict.

    Corrected to ensure oos_end is the last index of test features.
    """
    # Defensive check to prevent TypeError
    if not isinstance(data_dict, dict):
        return None, None
    if "split_dates" not in data_dict or data_dict["split_dates"] is None:
        return None, None
    
    ref = next((e for e in TARGET_ETFS if e in data_dict["split_dates"]), None)
    if ref is None:
        return None, None

    sd = data_dict["split_dates"][ref]
    oos_start = pd.Timestamp(sd["oos_start"]) if sd.get("oos_start") else None

    # Use the last index of X_test to get true oos_end
    if ref in data_dict.get("features_test", {}):
        oos_end = data_dict["features_test"][ref].index[-1]
    else:
        oos_end = pd.Timestamp(sd["oos_end"]) if sd.get("oos_end") else None

    return oos_start, oos_end
