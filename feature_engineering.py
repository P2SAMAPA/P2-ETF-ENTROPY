import pandas as pd
import numpy as np

def add_technical_indicators(df, ma_window):

    result = df.copy()

    for col in df.columns:
        if col == "3MTBILL":
            continue

        result[f"{col}_MA"] = df[col].rolling(ma_window).mean()
        result[f"{col}_MA_DIFF"] = result[f"{col}_MA"].diff()
        result[f"{col}_EMA"] = df[col].ewm(span=ma_window).mean()
        result[f"{col}_EMA_DIFF"] = result[f"{col}_EMA"].diff()

        result[f"{col}_RSI"] = compute_rsi(df[col])
        result[f"{col}_ATR"] = df[col].rolling(14).std()

    return result.dropna()

def compute_rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))
