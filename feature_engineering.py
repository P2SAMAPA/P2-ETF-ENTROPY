"""
Feature Engineering — TUNED
Changes vs original:
  - create_target(): normalised to MA_Diff / price (return %, not dollar amount)
    → same scale across all 7 ETFs, models can rank fairly
  - add_technical_indicators(): MA_DIFF and RET lags extended 5→10
  - prepare_all_features(): per-ETF split indices (not shared first-ETF index)
  - split_dates stores strings for json.dump compatibility
"""

import pandas as pd
import numpy as np

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


def add_technical_indicators(df, ma_window=5):
    """
    Build feature matrix.
    MA_DIFF and RET lags extended to 10 (was 5) for longer autocorrelation memory.
    """
    result   = pd.DataFrame(index=df.index)
    etf_cols = [c for c in df.columns if c in TARGET_ETFS]

    for col in etf_cols:
        price = df[col]
        ret   = price.pct_change()

        result[f"{col}_RET"] = ret

        for w in [3, 5]:
            ma   = price.rolling(w).mean()
            diff = ma.diff()
            result[f"{col}_MA{w}"]      = ma
            result[f"{col}_MA{w}_DIFF"] = diff
            # Extended lags: 1→10
            for lag in range(1, 11):
                result[f"{col}_MA{w}_DIFF_LAG{lag}"] = diff.shift(lag)

        # Return lags 1→10
        for lag in range(1, 11):
            result[f"{col}_RET_LAG{lag}"] = ret.shift(lag)

        result[f"{col}_VOL20"]       = ret.rolling(20).std()
        result[f"{col}_RSI"]         = compute_rsi(price)
        bb_u, bb_l                   = compute_bollinger_bands(price)
        result[f"{col}_BB_UPPER"]    = bb_u
        result[f"{col}_BB_LOWER"]    = bb_l
        macd, macd_sig               = compute_macd(price)
        result[f"{col}_MACD"]        = macd
        result[f"{col}_MACD_SIGNAL"] = macd_sig
        result[f"{col}_ATR"]         = compute_atr(price)

    # Cross-ETF relative price ratios
    for col in etf_cols:
        for other in etf_cols:
            if other != col:
                result[f"{col}_REL_{other}"] = df[col] / (df[other] + 1e-10)

    if "3MTBILL" in df.columns:
        result["TBILL"] = df["3MTBILL"]

    return result


# ── Target: normalised MA_Diff / price ───────────────────────────────────────

def create_target(df, etf, ma_window=5):
    """
    Target = next-day MA_Diff / price  (percentage, same scale for all ETFs).
    Previously returned raw MA_Diff dollar value — broke cross-ETF comparisons.
    """
    ma    = df[etf].rolling(ma_window).mean()
    diff  = ma.diff().shift(-1)          # next-day MA_Diff
    price = df[etf]
    return diff / (price + 1e-10)        # normalise to return %


# ── Split indices ─────────────────────────────────────────────────────────────

def compute_split_indices(n):
    train_end = int(n * TRAIN_PCT)
    val_end   = int(n * (TRAIN_PCT + VAL_PCT))
    return train_end, val_end


# ── Main entry point ──────────────────────────────────────────────────────────

def prepare_all_features(df, ma_window=5, year_start=2008):
    """
    80/10/10 split, per-ETF indices, Z-score normalisation on train stats only.
    split_dates values are strings (json-serialisable).
    """
    df_filtered = df[df.index >= f"{year_start}-01-01"].copy()

    if len(df_filtered) < 100:
        return {"features": {}, "split_dates": {}, "features_test": {}}

    features_df = add_technical_indicators(df_filtered, ma_window)
    features_df = features_df.dropna()
    df_aligned  = df_filtered.loc[features_df.index]

    features      = {}
    targets       = {}
    split_dates   = {}
    features_test = {}

    for etf in TARGET_ETFS:
        if etf not in df_aligned.columns:
            continue

        etf_cols = [c for c in features_df.columns
                    if c.startswith(f"{etf}_") or c == "TBILL"]
        if not etf_cols:
            continue

        X_full = features_df[etf_cols]
        y_full = create_target(df_aligned, etf, ma_window)

        common_idx = X_full.index.intersection(y_full.dropna().index)
        X_full = X_full.loc[common_idx]
        y_full = y_full.loc[common_idx]

        if len(X_full) < 50:
            continue

        # Per-ETF split (avoids length mismatch after dropna)
        n                      = len(X_full)
        train_end, val_end     = compute_split_indices(n)

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

        # String dates for json.dump
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

    # Expose per-split feature dicts for ma_optimizer
    ref        = TARGET_ETFS[0] if TARGET_ETFS[0] in features else next(iter(features))
    n_ref      = len(features[ref])
    train_end_ref, val_end_ref = compute_split_indices(n_ref)

    return {
        "features":      features,
        "targets":       targets,
        "split_dates":   split_dates,
        "features_test": features_test,
        "train_end":     train_end_ref,
        "val_end":       val_end_ref,
    }


def get_oos_dates(data_dict):
    if not isinstance(data_dict, dict):
        return None, None
    if not data_dict.get("split_dates"):
        return None, None

    ref = next((e for e in TARGET_ETFS if e in data_dict["split_dates"]), None)
    if ref is None:
        return None, None

    sd        = data_dict["split_dates"][ref]
    oos_start = pd.Timestamp(sd["oos_start"]) if sd.get("oos_start") else None

    if ref in data_dict.get("features_test", {}):
        oos_end = data_dict["features_test"][ref].index[-1]
    else:
        oos_end = pd.Timestamp(sd["oos_end"]) if sd.get("oos_end") else None

    return oos_start, oos_end
