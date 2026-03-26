"""
Feature Engineering — SPEED OPTIMISED
Now supports dynamic ETF lists for Option A and Option B.
"""

import pandas as pd
import numpy as np

# Default ETF list for backward compatibility (Option A)
TARGET_ETFS    = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]
BENCHMARK_ETFS = ["SPY", "AGG"]
TRAIN_PCT = 0.80
VAL_PCT   = 0.10


# ── Technical indicators ──────────────────────────────────────────────────────

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


def add_technical_indicators(df, ma_window=5, etf_list=None):
    """
    Build feature matrix for the given ETF list.
    If etf_list is None, uses global TARGET_ETFS.
    """
    if etf_list is None:
        etf_list = TARGET_ETFS

    cols = {}
    etf_cols = [c for c in df.columns if c in etf_list]

    for col in etf_cols:
        price = df[col]
        ret   = price.pct_change()

        cols[f"{col}_RET"] = ret

        # Single MA window
        ma   = price.rolling(ma_window).mean()
        diff = ma.diff()
        cols[f"{col}_MA{ma_window}"]          = ma
        cols[f"{col}_MA{ma_window}_DIFF"]     = diff

        # Lags 1→5
        for lag in range(1, 6):
            cols[f"{col}_MA{ma_window}_DIFF_LAG{lag}"] = diff.shift(lag)
            cols[f"{col}_RET_LAG{lag}"]                = ret.shift(lag)

        cols[f"{col}_VOL20"]       = ret.rolling(20).std()
        cols[f"{col}_RSI"]         = compute_rsi(price)
        bb_u, bb_l                 = compute_bollinger_bands(price)
        cols[f"{col}_BB_UPPER"]    = bb_u
        cols[f"{col}_BB_LOWER"]    = bb_l
        macd, macd_sig             = compute_macd(price)
        cols[f"{col}_MACD"]        = macd
        cols[f"{col}_MACD_SIGNAL"] = macd_sig
        cols[f"{col}_ATR"]         = compute_atr(price)

    if "3MTBILL" in df.columns:
        cols["TBILL"] = df["3MTBILL"]

    # Build DataFrame once
    return pd.DataFrame(cols, index=df.index)


# ── Target: normalised MA_Diff / price ───────────────────────────────────────

def create_target(df, etf, ma_window=5):
    """
    Target = next-day MA_Diff / price  (percentage, same scale for all ETFs).
    """
    ma    = df[etf].rolling(ma_window).mean()
    diff  = ma.diff().shift(-1)
    price = df[etf]
    return diff / (price + 1e-10)


# ── Split indices ─────────────────────────────────────────────────────────────

def compute_split_indices(n):
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * (TRAIN_PCT + VAL_PCT))
    return train_end, val_end


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare_all_features(df, ma_window=5, year_start=2008, etf_list=None):
    """
    80/10/10 split, per-ETF indices, Z-score normalisation on train stats only.
    If etf_list is None, uses global TARGET_ETFS.
    """
    if etf_list is None:
        etf_list = TARGET_ETFS

    df_filtered = df[df.index >= f"{year_start}-01-01"].copy()

    if len(df_filtered) < 100:
        return {"features": {}, "split_dates": {}, "features_test": {}}

    features_df = add_technical_indicators(df_filtered, ma_window, etf_list=etf_list)
    features_df = features_df.dropna()
    df_aligned  = df_filtered.loc[features_df.index]

    features      = {}
    targets       = {}
    split_dates   = {}
    features_test = {}

    for etf in etf_list:
        if etf not in df_aligned.columns:
            continue

        etf_cols = [c for c in features_df.columns
                    if c.startswith(f"{etf}_") or c == "TBILL"]
        if not etf_cols:
            continue

        X_full = features_df[etf_cols]
        y_full = create_target(df_aligned, etf, ma_window)

        # Keep X_full on full features index — do NOT drop the last row just
        # because its forward target is NaN. The last row is the live signal
        # date (today's close → tomorrow's allocation). It has no forward
        # return yet, so y is NaN for it, but we still need it in X for
        # prediction. Drop NaN only from y; X keeps all rows.
        y_full     = y_full.loc[X_full.index.intersection(y_full.dropna().index)]

        if len(X_full) < 50:
            continue

        n                  = len(X_full)
        train_end, val_end = compute_split_indices(n)

        X_train = X_full.iloc[:train_end]
        X_val   = X_full.iloc[train_end:val_end]
        X_test  = X_full.iloc[val_end:]

        y_train = y_full.iloc[:train_end]
        y_val   = y_full.iloc[train_end:val_end]
        y_test  = y_full.iloc[val_end:]

        # Z-score normalise using TRAIN statistics only
        train_mean = X_train.mean()
        train_std  = X_train.std().replace(0, 1e-10)

        X_train_n = (X_train - train_mean) / train_std
        X_val_n   = (X_val   - train_mean) / train_std
        X_test_n  = (X_test  - train_mean) / train_std

        features[etf] = pd.concat([X_train_n, X_val_n, X_test_n])

        targets[etf] = {
            "train": y_train,
            "val":   y_val,
            "test":  y_test,
        }

        split_dates[etf] = {
            "train_start": str(X_train.index[0].date()),
            "train_end":   str(X_train.index[-1].date()),
            "val_start":   str(X_val.index[0].date())   if len(X_val)  > 0 else "",
            "val_end":     str(X_val.index[-1].date())  if len(X_val)  > 0 else "",
            "oos_start":   str(X_test.index[0].date())  if len(X_test) > 0 else "",
            "oos_end":     str(X_test.index[-1].date()) if len(X_test) > 0 else "",
            "n_train":     train_end,
            "n_val":       val_end - train_end,
            "n_test":      n - val_end,
        }

        features_test[etf] = X_test_n

    if not features:
        return {"features": {}, "split_dates": {}, "features_test": {}}

    # Use the first ETF in the list as reference for lengths (for backward compatibility)
    ref           = etf_list[0] if etf_list[0] in features else next(iter(features))
    n_ref         = len(features[ref])
    train_end_ref, val_end_ref = compute_split_indices(n_ref)

    return {
        "features":      features,
        "targets":       targets,
        "split_dates":   split_dates,
        "features_test": features_test,
        "train_end":     train_end_ref,
        "val_end":       val_end_ref,
    }


def get_oos_dates(data_dict, etf_list=None):
    """
    Get OOS start and end dates. If etf_list is provided, use it; otherwise use global TARGET_ETFS.
    """
    if not isinstance(data_dict, dict):
        return None, None
    if not data_dict.get("split_dates"):
        return None, None

    if etf_list is None:
        etf_list = TARGET_ETFS

    ref = next((e for e in etf_list if e in data_dict["split_dates"]), None)
    if ref is None:
        return None, None

    sd        = data_dict["split_dates"][ref]
    oos_start = pd.Timestamp(sd["oos_start"]) if sd.get("oos_start") else None

    # Read oos_end from the full features index — this includes the last
    # live trading day even when its forward target is NaN (no tomorrow yet).
    # features_test stops at val_end and may exclude the current live row.
    if ref in data_dict.get("features", {}):
        oos_end = data_dict["features"][ref].index[-1]
    elif ref in data_dict.get("features_test", {}):
        oos_end = data_dict["features_test"][ref].index[-1]
    else:
        oos_end = pd.Timestamp(sd["oos_end"]) if sd.get("oos_end") else None

    return oos_start, oos_end
