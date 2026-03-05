#!/usr/bin/env python3
"""
Training script for GitHub Actions
Production-safe version
"""

import sys
import numpy as np
import pandas as pd

from data_loader import load_dataset
from feature_engineering import prepare_all_features
from ma_optimizer import optimize_ma_window


def main():

    print("Starting weekly retraining...")

    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    df = load_dataset()

    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # --------------------------------------------------
    # 2. Define ETF universe
    # --------------------------------------------------
    etf_list = ['TLT', 'VNQ', 'GLD', 'SLV', 'VCIT', 'HYG', 'LQD']

    # Ensure required columns exist
    required_cols = etf_list + ["3MTBILL"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in dataset: {missing}")

    price_df = df[etf_list].copy()
    tbill_df = df["3MTBILL"].copy()

    # --------------------------------------------------
    # 3. Prepare features for MA windows
    # --------------------------------------------------
    data_dict = {}

    for ma in [3, 5]:
        print(f"Preparing features for MA({ma})...")
        data_dict[ma] = prepare_all_features(df, ma_window=ma)

    # --------------------------------------------------
    # 4. Run MA Optimization
    # --------------------------------------------------
    best_ma, results = optimize_ma_window(
        etf_list=etf_list,
        data_dict=data_dict,
        price_df=price_df,
        tbill_df=tbill_df
    )

    # --------------------------------------------------
    # 5. Print results cleanly
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Training complete. Best MA window: {best_ma}")
    print(f"MA(3) annualized return: {results[3]:.4f}")
    print(f"MA(5) annualized return: {results[5]:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
