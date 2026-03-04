"""
MA Window Optimizer
Trains both MA(3) and MA(5) models, compares OOS performance,
selects best by Annualized Return (not risk-adjusted)
"""

import json
import os
import joblib
import pandas as pd
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics


def optimize_ma_window(etf_list, data_dict, price_df, train_idx, val_idx, test_idx, 
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
    train_idx, val_idx, test_idx : array-like
        Indices for splits
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
        
        # Prepare data for this MA window
        X_train = {}
        y_train = {}
        X_val = {}
        
        for etf in etf_list:
            features = data_dict[ma_window]['features'][etf]
            targets = data_dict[ma_window]['targets'][etf]
            
            X_train[etf] = features.iloc[train_idx]
            y_train[etf] = targets.iloc[train_idx]
            X_val[etf] = features.iloc[val_idx]
        
        # Train Transfer Voting model
        model = TransferVotingModel(etf_list, ma_window, artifact_path)
        model.fit(X_train, y_train, price_df.iloc[train_idx])
        
        # Save model
        model_path = f"{artifact_path}/transfer_voting_MA{ma_window}.pkl"
        model.save(model_path)
        
        # Validate on validation set (quick backtest)
        print(f"\nValidating MA({ma_window}) on validation set...")
        
        # Generate predictions for validation period
        val_predictions = {}
        for etf in etf_list:
            val_predictions[etf] = model.predict_single_etf(X_val[etf], etf)
        
        # Run quick strategy backtest on validation
        val_metrics = run_quick_evaluation(
            val_predictions, 
            price_df.iloc[val_idx], 
            data_dict[ma_window]['features'],
            val_idx
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
    
    # Select best by Annualized Return (as per your requirement)
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


def run_quick_evaluation(predictions, price_df, features_dict, val_idx):
    """
    Quick evaluation on validation set for MA selection
    Simplified version of full backtest
    """
    # Simple strategy: pick ETF with highest predicted MA_Diff each day
    pred_df = pd.DataFrame(predictions, index=price_df.index)
    
    # Convert predictions to expected returns (approximate)
    expected_returns = pred_df.div(price_df[price_df.columns], axis=1)
    
    # Select best ETF each day
    selected = expected_returns.idxmax(axis=1)
    
    # Calculate returns
    daily_returns = price_df.pct_change().shift(-1)  # Next day return
    
    strategy_returns = []
    for i, date in enumerate(pred_df.index):
        if i < len(pred_df) - 1:
            etf = selected.loc[date]
            if pd.notna(etf) and etf in daily_returns.columns:
                next_day = pred_df.index[i + 1]
                if next_day in daily_returns.index:
                    ret = daily_returns.loc[next_day, etf]
                    strategy_returns.append(ret)
    
    strategy_returns = pd.Series(strategy_returns)
    
    # Calculate metrics
    total_return = (1 + strategy_returns).prod() - 1
    n_days = len(strategy_returns)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    
    # Sharpe (assume 0 for simplicity in quick eval, or use mean/variance)
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std() if strategy_returns.std() > 0 else 0
    
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
