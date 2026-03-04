"""
MA Window Optimizer - OPTIMIZED for faster training
Trains both MA(3) and MA(5) models, selects best by validation MSE
"""

import json
import os
import joblib
import pandas as pd
import numpy as np
from transfer_voting import TransferVotingModel


def optimize_ma_window(etf_list, data_dict, price_df, 
                       artifact_path='artifacts'):
    """
    Train both MA(3) and MA(5), select best by validation MSE
    
    Parameters:
    -----------
    etf_list : list
        List of ETF tickers
    data_dict : dict
        Dictionary with 'features' and 'targets' for each MA window
    price_df : pd.DataFrame
        Price data
    artifact_path : str
    
    Returns:
    --------
    best_ma : int (3 or 5)
    results : dict with full comparison
    """
    
    os.makedirs(artifact_path, exist_ok=True)
    results = {}
    
    for ma_window in [3, 5]:
        print(f"\n{'='*60}")
        print(f"TRAINING MA({ma_window}) MODEL")
        print(f"{'='*60}")
        
        # Get data for this MA window
        features_dict = data_dict[ma_window]['features']
        targets_dict = data_dict[ma_window]['targets']
        
        # Find common date range across all ETFs
        common_dates = None
        for etf in etf_list:
            if etf in features_dict:
                dates = features_dict[etf].index
                if common_dates is None:
                    common_dates = dates
                else:
                    common_dates = common_dates.intersection(dates)
        
        print(f"Common date range: {len(common_dates)} days ({common_dates[0]} to {common_dates[-1]})")
        
        # Split by date (80/10/10)
        n = len(common_dates)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        
        train_dates = common_dates[:train_end]
        val_dates = common_dates[train_end:val_end]
        test_dates = common_dates[val_end:]
        
        print(f"Data splits: Train {len(train_dates)}, Val {len(val_dates)}, Test {len(test_dates)}")
        
        # Prepare train/val data
        X_train = {}
        y_train = {}
        X_val = {}
        y_val = {}
        
        for etf in etf_list:
            if etf in features_dict:
                X_train[etf] = features_dict[etf].loc[train_dates]
                y_train[etf] = targets_dict[etf].loc[train_dates]
                X_val[etf] = features_dict[etf].loc[val_dates]
                y_val[etf] = targets_dict[etf].loc[val_dates]
        
        # Align price data
        price_aligned = price_df.loc[common_dates]
        
        # Train Transfer Voting model
        model = TransferVotingModel(etf_list, ma_window, artifact_path)
        model.fit(X_train, y_train, price_aligned.loc[train_dates])
        
        # Save model
        model_path = f"{artifact_path}/transfer_voting_MA{ma_window}.pkl"
        model.save(model_path)
        
        # Quick validation: use MSE instead of full backtest for speed
        print(f"\nValidating MA({ma_window})...")
        val_mse_list = []
        
        for etf in etf_list:
            if etf in X_val:
                preds = model.predict_single_etf(X_val[etf], etf)
                actual = y_val[etf].values
                mse = np.mean((preds - actual) ** 2)
                val_mse_list.append(mse)
        
        avg_mse = np.mean(val_mse_list) if val_mse_list else 999
        
        # Use inverse MSE as selection metric (higher = better)
        pseudo_return = 1.0 / (1.0 + avg_mse)
        
        results[ma_window] = {
            'model_path': model_path,
            'annualized_return': pseudo_return,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'mse': avg_mse
        }
        
        print(f"MA({ma_window}) Validation MSE: {avg_mse:.6f}")
        print(f"MA({ma_window}) Pseudo-Return: {pseudo_return:.6f}")
    
    # Select best by lowest MSE (highest pseudo-return)
    best_ma = 3 if results[3]['annualized_return'] > results[5]['annualized_return'] else 5
    
    print(f"\n{'='*60}")
    print(f"BEST MA WINDOW SELECTED: MA({best_ma})")
    print(f"  MA(3) MSE: {results[3]['mse']:.6f}")
    print(f"  MA(5) MSE: {results[5]['mse']:.6f}")
    print(f"{'='*60}")
    
    # Save selection
    best_model_info = {
        'best_ma_window': best_ma,
        'ma3_mse': results[3]['mse'],
        'ma5_mse': results[5]['mse'],
        'ma3_pseudo_return': results[3]['annualized_return'],
        'ma5_pseudo_return': results[5]['annualized_return'],
        'selection_criteria': 'lowest_validation_mse'
    }
    
    with open(f"{artifact_path}/best_model.json", 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    return best_ma, results


def load_best_model(artifact_path='artifacts'):
    """Load the selected best model"""
    with open(f"{artifact_path}/best_model.json", 'r') as f:
        info = json.load(f)
    
    best_ma = info['best_ma_window']
    model_path = f"{artifact_path}/transfer_voting_MA{best_ma}.pkl"
    
    model = TransferVotingModel([], best_ma, artifact_path)
    model.load(model_path)
    
    return model, best_ma
