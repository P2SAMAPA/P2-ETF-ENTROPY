"""
MA Window Optimizer
Trains both MA(3) and MA(5) models, compares OOS performance,
selects best by Annualized Return (not risk-adjusted)
"""

import json
import os
import joblib
import pandas as pd
import numpy as np
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics


def optimize_ma_window(etf_list, data_dict, price_df, 
                       artifact_path='artifacts'):
    """
    Train both MA(3) and MA(5), select best by Annualized Return on validation set
    
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
        
        # Split by date (80/10/10) instead of index
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
        
        for etf in etf_list:
            if etf in features_dict:
                # Select by date index
                X_train[etf] = features_dict[etf].loc[train_dates]
                y_train[etf] = targets_dict[etf].loc[train_dates]
                X_val[etf] = features_dict[etf].loc[val_dates]
        
        # Align price data to common dates
        price_aligned = price_df.loc[common_dates]
        
        # Train Transfer Voting model
        model = TransferVotingModel(etf_list, ma_window, artifact_path)
        model.fit(X_train, y_train, price_aligned.loc[train_dates])
        
        # Save model
        model_path = f"{artifact_path}/transfer_voting_MA{ma_window}.pkl"
        model.save(model_path)
        
        # Validate on validation set
        print(f"\nValidating MA({ma_window}) on validation set...")
        
        # Generate predictions for validation period
        val_predictions = {}
        for etf in etf_list:
            if etf in X_val:
                val_predictions[etf] = model.predict_single_etf(X_val[etf], etf)
        
        # Run quick strategy backtest on validation
        val_metrics = run_quick_evaluation(
            val_predictions, 
            price_aligned.loc[val_dates], 
            val_dates
        )
        
        results[ma_window] = {
            'model_path': model_path,
            'annualized_return': val_metrics['annualized_return'],
            'sharpe_ratio': val_metrics['sharpe_ratio'],
            'max_drawdown': val_metrics['max_drawdown']
        }
        
        print(f"MA({ma_window}) Validation Results:")
        print(f"  Annualized Return: {val_metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {val_metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {val_metrics['max_drawdown']:.2%}")
    
    # Select best by Annualized Return
    best_ma = 3 if results[3]['annualized_return'] > results[5]['annualized_return'] else 5
    
    print(f"\n{'='*60}")
    print(f"BEST MA WINDOW SELECTED: MA({best_ma})")
    print(f"  MA(3) Return: {results[3]['annualized_return']:.2%}")
    print(f"  MA(5) Return: {results[5]['annualized_return']:.2%}")
    print(f"{'='*60}")
    
    # Save selection
    best_model_info = {
        'best_ma_window': best_ma,
        'ma3_return': results[3]['annualized_return'],
        'ma5_return': results[5]['annualized_return'],
        'selection_criteria': 'highest_annualized_return'
    }
    
    with open(f"{artifact_path}/best_model.json", 'w') as f:
        json.dump(best_model_info, f, indent=2)
    
    return best_ma, results


def run_quick_evaluation(predictions, price_df, dates):
    """
    Quick evaluation on validation set for MA selection
    Simplified strategy: pick ETF with highest predicted MA_Diff each day
    """
    # Convert predictions to DataFrame
    pred_df = pd.DataFrame({k: pd.Series(v, index=dates) for k, v in predictions.items()})
    
    # Simple strategy: pick ETF with highest predicted MA_Diff each day
    selected = pred_df.idxmax(axis=1)
    
    # Calculate returns
    daily_returns = price_df.pct_change().shift(-1)
    
    strategy_returns = []
    for i, date in enumerate(dates[:-1]):  # Exclude last day
        etf = selected.loc[date]
        if pd.notna(etf) and etf in daily_returns.columns:
            next_date = dates[i + 1]
            if next_date in daily_returns.index:
                ret = daily_returns.loc[date, etf]  # Return from date to next_date
                strategy_returns.append(ret)
    
    strategy_returns = pd.Series(strategy_returns)
    
    # Calculate metrics
    if len(strategy_returns) == 0 or strategy_returns.std() == 0:
        return {
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0
        }
    
    total_return = (1 + strategy_returns).prod() - 1
    n_days = len(strategy_returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    # Max drawdown
    equity = (1 + strategy_returns).cumprod()
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()
    
    return {
        'annualized_return': ann_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd
    }


def load_best_model(artifact_path='artifacts'):
    """Load the selected best model"""
    with open(f"{artifact_path}/best_model.json", 'r') as f:
        info = json.load(f)
    
    best_ma = info['best_ma_window']
    model_path = f"{artifact_path}/transfer_voting_MA{best_ma}.pkl"
    
    model = TransferVotingModel([], best_ma, artifact_path)
    model.load(model_path)
    
    return model, best_ma
