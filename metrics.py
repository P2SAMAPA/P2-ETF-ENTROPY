"""
Metrics Module
Calculates all strategy performance metrics for OOS period
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_metrics(equity_curve, returns, risk_free, audit_trail=None):
    """
    Calculate comprehensive performance metrics
    
    Parameters:
    -----------
    equity_curve : pd.Series or pd.DataFrame
        Equity curve of strategy (starting at 1.0)
    returns : pd.Series
        Daily strategy returns
    risk_free : pd.Series
        Daily risk-free rate (decimal)
    audit_trail : pd.DataFrame, optional
        Audit trail for hit ratio calculation
        
    Returns:
    --------
    metrics : dict
        Dictionary of all calculated metrics
    """
    
    metrics = {}
    
    # 1. Annualized Return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    n_days = len(returns)
    n_years = n_days / 252
    
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    metrics['annualized_return'] = ann_return
    metrics['total_return'] = total_return
    
    # 2. Sharpe Ratio (using 3M T-Bill)
    excess_returns = returns - risk_free
    if excess_returns.std() > 0:
        sharpe = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    else:
        sharpe = 0
    metrics['sharpe_ratio'] = sharpe
    
    # 3. Max Drawdown (Peak to Trough)
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    
    # Find peak date (start of this drawdown)
    if isinstance(max_dd_date, pd.Timestamp):
        peak_before = equity_curve.loc[:max_dd_date].idxmax()
    else:
        peak_before = None
    
    metrics['max_drawdown'] = max_dd
    metrics['max_drawdown_date'] = max_dd_date
    metrics['max_drawdown_peak_date'] = peak_before
    
    # 4. Worst Daily Drawdown (single day)
    worst_daily_dd = returns.min()
    worst_daily_date = returns.idxmin()
    metrics['worst_daily_return'] = worst_daily_dd
    metrics['worst_daily_date'] = worst_daily_date
    
    # 5. Volatility (annualized)
    volatility = returns.std() * np.sqrt(252)
    metrics['volatility'] = volatility
    
    # 6. Hit Ratio (last 15 trading days positive predictions)
    if audit_trail is not None and len(audit_trail) >= 15:
        last_15 = audit_trail.tail(15)
        positive_predictions = (last_15['actual_return'] > 0).sum()
        hit_ratio = positive_predictions / 15
        metrics['hit_ratio_15d'] = hit_ratio
        metrics['hit_ratio_count'] = positive_predictions
    else:
        metrics['hit_ratio_15d'] = None
        metrics['hit_ratio_count'] = 0
    
    # 7. Calmar Ratio (AnnReturn / MaxDD)
    if max_dd != 0:
        calmar = ann_return / abs(max_dd)
    else:
        calmar = np.inf if ann_return > 0 else 0
    metrics['calmar_ratio'] = calmar
    
    # 8. Sortino Ratio (downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = np.sqrt(252) * returns.mean() / downside_returns.std()
    else:
        sortino = np.inf if returns.mean() > 0 else 0
    metrics['sortino_ratio'] = sortino
    
    # 9. Win Rate (overall)
    wins = (returns > 0).sum()
    total_days = len(returns)
    metrics['win_rate'] = wins / total_days if total_days > 0 else 0
    
    # 10. Average Win / Average Loss
    avg_win = returns[returns > 0].mean() if wins > 0 else 0
    avg_loss = returns[returns < 0].mean() if (returns < 0).sum() > 0 else 0
    metrics['avg_win'] = avg_win
    metrics['avg_loss'] = avg_loss
    metrics['win_loss_ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    return metrics


def format_metrics_for_display(metrics):
    """
    Format metrics for nice Streamlit display
    """
    formatted = {
        'Annualized Return': f"{metrics['annualized_return']:.2%}",
        'Sharpe Ratio': f"{metrics['sharpe_ratio']:.2f}",
        'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
        'Max DD Date': str(metrics['max_drawdown_date'])[:10] if metrics['max_drawdown_date'] else 'N/A',
        'Worst Daily DD': f"{metrics['worst_daily_return']:.2%}",
        'Worst Daily Date': str(metrics['worst_daily_date'])[:10] if metrics['worst_daily_date'] else 'N/A',
        'Hit Ratio (15d)': f"{metrics['hit_ratio_15d']:.1%}" if metrics['hit_ratio_15d'] is not None else 'N/A',
        'Volatility': f"{metrics['volatility']:.2%}",
        'Win Rate': f"{metrics['win_rate']:.1%}",
        'Calmar Ratio': f"{metrics['calmar_ratio']:.2f}",
        'Total Return': f"{metrics['total_return']:.2%}"
    }
    return formatted


def calculate_benchmark_metrics(price_df, tbill_df, test_dates, benchmark='SPY'):
    """
    Calculate metrics for benchmark buy-and-hold
    """
    if benchmark not in price_df.columns:
        return None
    
    bench_prices = price_df.loc[test_dates, benchmark]
    bench_returns = bench_prices.pct_change().dropna()
    
    # Align dates
    bench_returns = bench_returns.iloc[:len(test_dates)-1]
    rf_aligned = risk_free.iloc[:len(bench_returns)]
    
    equity = (1 + bench_returns).cumprod()
    
    return calculate_metrics(equity, bench_returns, rf_aligned)
