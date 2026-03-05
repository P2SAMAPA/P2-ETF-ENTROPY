#!/usr/bin/env python3
"""
Training script for GitHub Actions — TUNED
Prepares features for MA windows [2,3,4,5,7,10], runs optimize_ma_window.
"""

import sys
import time
import numpy as np
import pandas as pd

from data_loader import load_dataset
from feature_engineering import prepare_all_features
from ma_optimizer import optimize_ma_window, MA_WINDOWS


def main():
    t0 = time.time()
    print("Starting weekly retraining …")

    df = load_dataset()
    print(f"Dataset shape: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")

    etf_list = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]
    required = etf_list + ["3MTBILL"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    price_df = df[etf_list].copy()
    tbill_df = df["3MTBILL"].copy()

    # Prepare features for every MA window
    data_dict = {}
    for ma in MA_WINDOWS:
        print(f"\nPreparing features MA({ma}) …")
        t = time.time()
        data_dict[ma] = prepare_all_features(df, ma_window=ma)
        print(f"  MA({ma}) features ready in {round(time.time()-t,1)}s")

    # Optimise + train
    best_ma, results = optimize_ma_window(
        etf_list  = etf_list,
        data_dict = data_dict,
        price_df  = price_df,
        tbill_df  = tbill_df,
    )

    total = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"Training complete in {total}s  |  Best MA: MA({best_ma})")
    for w, r in sorted(results.items()):
        print(f"  MA({w}): val ann_return = {r*100:.2f}%")
    print("="*60)


if __name__ == "__main__":
    main()
