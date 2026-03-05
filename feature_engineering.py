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
    """
    Prepare features with 80/10/10 split starting from year_start.

    Returns dict with keys:
        features, targets                        — full normalised
        features_train, features_val, features_test
        targets_train,  targets_val,  targets_test
        train_stats   {etf: (mean, std)}
        split_dates   {etf: {train_start, train_end, val_start,
                              val_end, oos_start, oos_end,
                              n_train, n_val, n_test}}
        full_features, year_start
    """
    df = df[df.index >= f"{year_start}-01-01"].copy()
    keep = [c for c in df.columns if c in TARGET_ETFS + ["3MTBILL"]]
    df   = df[keep]

    if len(df) < 300:
        return {"features": {}, "split_dates": {}, "features_train": {},
                "features_val": {}, "features_test": {}, "targets_train": {},
                "targets_val": {}, "targets_test": {}, "targets": {},
                "train_stats": {}, "full_features": pd.DataFrame(),
                "year_start": year_start}

    features_df = add_technical_indicators(df, ma_window)

    results = {
        "features": {}, "targets": {},
        "features_train": {}, "features_val": {}, "features_test": {},
        "targets_train":  {}, "targets_val":  {}, "targets_test":  {},
        "train_stats": {}, "split_dates": {},
        "full_features": features_df,
        "year_start": year_start,
    }

    for etf in TARGET_ETFS:
        if etf not in df.columns:
            continue

        target   = create_target(df, etf, ma_window)
        own_cols = [c for c in features_df.columns if c.startswith(f"{etf}_") or c == "TBILL"]
        if not own_cols:
            continue

        X = features_df[own_cols].copy()

        # Align X and y, drop NaN rows
        combined         = X.copy()
        combined["_TGT"] = target
        combined         = combined.dropna()

        if len(combined) < 300:
            print(f"  {etf}: insufficient data ({len(combined)} rows) — skipping")
            continue

        X_full = combined.drop("_TGT", axis=1)
        y_full = combined["_TGT"]
        n      = len(X_full)

        # ── 80/10/10 split (computed per ETF — no cross-ETF length assumption) ──
        train_end_idx, val_end_idx = compute_split_indices(n)

        X_train = X_full.iloc[:train_end_idx]
        X_val   = X_full.iloc[train_end_idx:val_end_idx]
        X_test  = X_full.iloc[val_end_idx:]
        y_train = y_full.iloc[:train_end_idx]
        y_val   = y_full.iloc[train_end_idx:val_end_idx]
        y_test  = y_full.iloc[val_end_idx:]

        # ── Z-score normalisation — train stats only ──────────────────
        train_mean = X_train.mean()
        train_std  = X_train.std().replace(0, 1e-10)

        X_train_n = (X_train - train_mean) / train_std
        X_val_n   = (X_val   - train_mean) / train_std
        X_test_n  = (X_test  - train_mean) / train_std
        X_full_n  = (X_full  - train_mean) / train_std

        # ── split_dates: store as strings for json.dump compatibility ─
        results["split_dates"][etf] = {
            "train_start": str(X_train.index[0].date()),
            "train_end":   str(X_train.index[-1].date()),
            "val_start":   str(X_val.index[0].date()),
            "val_end":     str(X_val.index[-1].date()),
            "oos_start":   str(X_test.index[0].date()),
            "oos_end":     str(X_test.index[-1].date()),
            "n_train":     len(X_train),
            "n_val":       len(X_val),
            "n_test":      len(X_test),
        }

        results["features"][etf]       = X_full_n
        results["targets"][etf]        = y_full
        results["features_train"][etf] = X_train_n
        results["features_val"][etf]   = X_val_n
        results["features_test"][etf]  = X_test_n
        results["targets_train"][etf]  = y_train
        results["targets_val"][etf]    = y_val
        results["targets_test"][etf]   = y_test
        results["train_stats"][etf]    = (train_mean, train_std)

    print(f"Feature prep complete: {len(results['features'])} ETFs | "
          f"MA({ma_window}) | year_start={year_start}")
    return results


def get_oos_dates(data_dict):
    """Return (oos_start, oos_end) as pd.Timestamps."""
    if not isinstance(data_dict, dict):
        return None, None
    if not data_dict.get("split_dates"):
        return None, None
    ref = next((e for e in TARGET_ETFS if e in data_dict["split_dates"]), None)
    if ref is None:
        return None, None
    sd = data_dict["split_dates"][ref]
    oos_start = pd.Timestamp(sd["oos_start"]) if sd.get("oos_start") else None
    oos_end   = pd.Timestamp(sd["oos_end"])   if sd.get("oos_end")   else None
    return oos_start, oos_end
