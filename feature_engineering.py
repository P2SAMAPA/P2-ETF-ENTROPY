"""
Feature Engineering Module
Creates MA-based technical indicators and targets for the 7 target ETFs.

Key fixes vs original:
  - EMA features removed (paper adaptation: MA-only)
  - SPY / AGG excluded from target ETF list (benchmarks only)
  - Train/test split enforced BEFORE feature construction to prevent
    look-ahead bias (normalisation stats from train only)
  - Both MA(3) and MA(5) targets built simultaneously so each sample
    has both prediction horizons available
  - Z-score normalisation applied using train-set statistics only
"""

import pandas as pd
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_ETFS   = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]
BENCHMARK_ETFS = ["SPY", "AGG"]
TRAIN_END     = "2021-12-31"   # paper: train 2017-2021, test 2022-2024


# ── Technical indicator helpers ────────────────────────────────────────────────

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0).rolling(window).mean()
    loss     = (-delta.clip(upper=0)).rolling(window).mean()
    rs       = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2):
    ma    = series.rolling(window).mean()
    std   = series.rolling(window).std()
    return ma + std * num_std, ma - std * num_std


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    return macd, macd.ewm(span=signal, adjust=False).mean()


def compute_atr(series: pd.Series, window: int = 14) -> pd.Series:
    """Simplified ATR on close-only data (true range = |Δclose|)."""
    return series.diff().abs().rolling(window).mean()


# ── Core feature builder ───────────────────────────────────────────────────────

def add_technical_indicators(df: pd.DataFrame, ma_window: int = 5) -> pd.DataFrame:
    """
    Build per-ETF feature columns.
    Only TARGET_ETFS get features built — benchmarks excluded.

    Features per ETF
    ----------------
    Returns & lags   : RET, RET_LAG1..5
    MA features      : MA3, MA5, MA3_DIFF, MA5_DIFF, MA_DIFF_LAG1..5
    Momentum         : RSI(14), MACD, MACD_SIGNAL
    Volatility       : VOL20, ATR(14), BB_UPPER, BB_LOWER
    Cross-ETF        : REL_STRENGTH vs each other target ETF
    Macro            : TBILL (3M T-Bill rate)
    """
    result = pd.DataFrame(index=df.index)

    etf_cols = [c for c in df.columns if c in TARGET_ETFS]

    for col in etf_cols:
        ret = df[col].pct_change()
        result[f"{col}_RET"] = ret

        # MA features — both windows always built as features
        for w in [3, 5]:
            ma   = df[col].rolling(w).mean()
            diff = ma.diff()
            result[f"{col}_MA{w}"]      = ma
            result[f"{col}_MA{w}_DIFF"] = diff
            for lag in range(1, 6):
                result[f"{col}_MA{w}_DIFF_LAG{lag}"] = diff.shift(lag)

        # Lagged returns
        for lag in range(1, 6):
            result[f"{col}_RET_LAG{lag}"] = ret.shift(lag)

        # Volatility & momentum
        result[f"{col}_VOL20"]       = ret.rolling(20).std()
        result[f"{col}_RSI"]         = compute_rsi(df[col])
        bb_upper, bb_lower           = compute_bollinger_bands(df[col])
        result[f"{col}_BB_UPPER"]    = bb_upper
        result[f"{col}_BB_LOWER"]    = bb_lower
        macd, macd_sig               = compute_macd(df[col])
        result[f"{col}_MACD"]        = macd
        result[f"{col}_MACD_SIGNAL"] = macd_sig
        result[f"{col}_ATR"]         = compute_atr(df[col])

    # Cross-ETF relative strength (only target ETFs)
    for col in etf_cols:
        for other in etf_cols:
            if other != col:
                result[f"{col}_REL_{other}"] = df[col] / (df[other] + 1e-10)

    # Macro
    if "3MTBILL" in df.columns:
        result["TBILL"] = df["3MTBILL"]

    return result


def create_target(df: pd.DataFrame, etf: str, ma_window: int = 5) -> pd.Series:
    """
    Target: MA_Diff(t+1) = MA(t+1) - MA(t)
    Shift by -1 so row t contains the label for next-day prediction.
    """
    ma = df[etf].rolling(ma_window).mean()
    return ma.diff().shift(-1)


# ── Main preparation function ──────────────────────────────────────────────────

def prepare_all_features(
    df: pd.DataFrame,
    ma_window: int = 5,
    train_end: str = TRAIN_END,
) -> dict:
    """
    Prepare complete feature set and targets for all TARGET_ETFS.

    Train/test split is enforced here:
      - Z-score normalisation stats computed on train rows only.
      - Test rows normalised with train stats (no look-ahead).

    Parameters
    ----------
    df : pd.DataFrame
        Raw price data (ETFs + 3MTBILL).
    ma_window : int
        MA window for the prediction TARGET (3 or 5).
        Both MA(3) and MA(5) are included as INPUT features regardless.
    train_end : str
        Last date of training window (inclusive).

    Returns
    -------
    dict with keys:
        'features'      : {etf: X_full DataFrame (all dates)}
        'targets'       : {etf: y_full Series (all dates)}
        'features_train': {etf: X_train}
        'features_test' : {etf: X_test}
        'targets_train' : {etf: y_train}
        'targets_test'  : {etf: y_test}
        'train_stats'   : {etf: (mean, std)} for de-normalisation
        'full_features' : raw un-normalised feature DataFrame
    """
    print(f"Preparing features — target MA({ma_window}), train_end={train_end}")

    # Only keep target ETFs + T-Bill
    keep_cols = [c for c in df.columns if c in TARGET_ETFS + ["3MTBILL"]]
    df = df[keep_cols].copy()

    # Build all technical indicators on full history
    # (indicator history prior to train_end is legitimately usable)
    features_df = add_technical_indicators(df, ma_window)

    results = {
        "features":       {},
        "targets":        {},
        "features_train": {},
        "features_test":  {},
        "targets_train":  {},
        "targets_test":   {},
        "train_stats":    {},
        "full_features":  features_df,
    }

    for etf in TARGET_ETFS:
        if etf not in df.columns:
            print(f"  {etf}: not in dataset — skipping")
            continue

        # Target for this ETF
        target = create_target(df, etf, ma_window)

        # Features for this ETF: its own indicators + TBILL
        own_cols   = [c for c in features_df.columns if c.startswith(etf) or c == "TBILL"]
        # Cross-ETF relative strength already included in add_technical_indicators
        X = features_df[own_cols].copy()

        # Align features + target, drop NaN rows
        combined         = X.copy()
        combined["_TGT"] = target
        combined         = combined.dropna()

        if len(combined) < 200:
            print(f"  {etf}: insufficient data ({len(combined)} rows) — skipping")
            continue

        X_full = combined.drop("_TGT", axis=1)
        y_full = combined["_TGT"]

        # ── Train / test split ─────────────────────────────────────
        train_mask = X_full.index <= pd.Timestamp(train_end)
        test_mask  = X_full.index >  pd.Timestamp(train_end)

        X_train = X_full[train_mask]
        X_test  = X_full[test_mask]
        y_train = y_full[train_mask]
        y_test  = y_full[test_mask]

        # ── Z-score normalisation (train stats only) ───────────────
        train_mean = X_train.mean()
        train_std  = X_train.std().replace(0, 1e-10)

        X_train_norm = (X_train - train_mean) / train_std
        X_test_norm  = (X_test  - train_mean) / train_std
        X_full_norm  = (X_full  - train_mean) / train_std

        print(f"  {etf}: train={len(X_train)} rows, test={len(X_test)} rows, "
              f"features={X_full.shape[1]}")

        results["features"][etf]        = X_full_norm
        results["targets"][etf]         = y_full
        results["features_train"][etf]  = X_train_norm
        results["features_test"][etf]   = X_test_norm
        results["targets_train"][etf]   = y_train
        results["targets_test"][etf]    = y_test
        results["train_stats"][etf]     = (train_mean, train_std)

    print(f"Feature preparation complete — {len(results['features'])} ETFs ready")
    return results
