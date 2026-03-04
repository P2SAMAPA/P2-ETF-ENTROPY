#!/usr/bin/env python3
"""
Training script for GitHub Actions
Avoids complex YAML escaping
"""

import sys
import numpy as np
import pandas as pd
from data_loader import load_dataset
from feature_engineering import prepare_all_features
from ma_optimizer import optimize_ma_window

def main():
    print("Starting weekly retraining...")
    
    # Load data
    df = load_dataset()
    print(f"Loaded dataset with shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Prepare features for both MA windows
    data_dict = {}
    for ma in [3, 5]:
        print(f"Preparing features for MA({ma})...")
        data_dict[ma] = prepare_all_features(df, ma_window=ma)
    
    # Split indices (80/10/10)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    train_idx = np.arange(0, train_end)
    val_idx = np.arange(train_end, val_end)
    test_idx = np.arange(val_end, n)
    
    print(f"Data splits: Train {len(train_idx)}, Val {len(val_idx)}, Test {len(test_idx)}")
    
    # Run optimization
    etf_list = ['TLT', 'VNQ', 'GLD', 'SLV', 'VCIT', 'HYG', 'LQD']
    
    best_ma, results = optimize_ma_window(
        etf_list, data_dict, df[etf_list], 
        train_idx, val_idx, test_idx
    )
    
    print(f"Training complete. Best MA window: {best_ma}")
    print(f"MA(3) return: {results[3]['annualized_return']:.4f}")
    print(f"MA(5) return: {results[5]['annualized_return']:.4f}")

if __name__ == "__main__":
    main()
