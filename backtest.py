"""
Backtest Module
Runs full OOS backtest with proper transaction cost accounting
"""

import pandas as pd
import numpy as np
from strategy_engine import StrategyEngine


def run_backtest(predictions_dict, price_df, tbill_df, strategy_engine, 
                 test_dates, benchmarks=['SPY', 'AGG']):
    """
    Run full OOS backtest
    
    Parameters:
    -----------
    predictions_dict : dict
        {etf: pd.Series} of predictions for test period
    price_df : pd.DataFrame
        Price data for all ETFs
    tbill_df : pd.Series or pd.DataFrame
        3-month T-bill rates (annualized %)
    strategy_engine : StrategyEngine
        Configured strategy engine
    test_dates : pd.DatetimeIndex
        OOS test dates
    benchmarks : list
        Benchmark ETFs to compare against
        
    Returns:
    --------
    results : dict
        Full backtest results including equity curve, metrics, audit trail
    """
    
    # Reset strategy state
    strategy_engine.reset()
    
    # Generate signals
    print("Generating trading signals...")
    signals = strategy_engine.generate_signals(predictions_dict, price_df, test_dates)
    
    # Calculate strategy returns
    print("Calculating strategy returns...")
    strategy_returns = []
    equity = 1.0
    equity_curve = [equity]
    
    audit_trail = []
    
    for i, date in enumerate(test_dates):
        if i == len(test_dates) - 1:
            # Last day, no forward return
            break
        
        next_date = test_dates[i + 1]
        selected = signals.loc[date, 'selected_etf']
        
        # Get return for selected position
        if selected == 'CASH' or selected is None:
            # Cash return = daily T-bill equivalent
            if isinstance(tbill_df, pd.DataFrame):
                tbill_rate = tbill_df.loc[date, '3MTBILL'] / 100  # Convert from %
            else:
                tbill_rate = tbill_df.loc[date] / 100
            
            daily_rf = (1 + tbill_rate) ** (1/252) - 1
            daily_return = daily_rf
            actual_etf_return = daily_rf
        else:
            # ETF return
            if next_date in price_df.index and selected in price_df.columns:
                price_today = price_df.loc[date, selected]
                price_next = price_df.loc[next_date, selected]
                
                if pd.notna(price_today) and pd.notna(price_next) and price_today > 0:
                    daily_return = (price_next / price_today) - 1
                    actual_etf_return = daily_return
                    
                    # Apply transaction cost if there was a switch
                    if signals.loc[date, 'switch_reason'] is not None:
                        daily_return -= strategy_engine.transaction_cost
                else:
                    daily_return = 0
                    actual_etf_return = 0
            else:
                daily_return = 0
                actual_etf_return = 0
        
        # Update equity
        equity *= (1 + daily_return)
        equity_curve.append(equity)
        strategy_returns.append(daily_return)
        
        # Record audit trail
        audit_trail.append({
            'date': date,
            'next_date': next_date,
            'predicted_etf': selected,
            'switch_reason': signals.loc[date, 'switch_reason'],
            'expected_return': signals.loc[date, 'expected_return'],
            'z_score': signals.loc[date, 'z_score'],
            'in_cash': signals.loc[date, 'in_cash'],
            'actual_return': actual_etf_return,
            'strategy_return': daily_return,
            'equity': equity
        })
    
    # Create results DataFrame
    equity_df = pd.DataFrame({
        'strategy': equity_curve[:-1]  # Align with returns
    }, index=test_dates[:-1])
    
    # Add benchmark equity curves
    for bench in benchmarks:
        if bench in price_df.columns:
            bench_prices = price_df.loc[test_dates, bench]
            bench_returns = bench_prices.pct_change().shift(-1).dropna()
            bench_equity = (1 + bench_returns).cumprod()
            # Normalize to start at 1.0
            bench_equity = bench_equity / bench_equity.iloc[0]
            equity_df[bench] = bench_equity.iloc[:len(equity_df)]
    
    # Calculate returns series
    returns_series = pd.Series(strategy_returns, index=test_dates[:-1])
    
    # Get T-bill series for risk-free
    if isinstance(tbill_df, pd.DataFrame):
        rf_series = tbill_df.loc[returns_series.index, '3MTBILL'] / 100 / 252  # Daily
    else:
        rf_series = tbill_df.loc[returns_series.index] / 100 / 252
    
    results = {
        'equity_curve': equity_df,
        'returns': returns_series,
        'risk_free': rf_series,
        'signals': signals,
        'audit_trail': pd.DataFrame(audit_trail),
        'n_trades': (signals['switch_reason'].notna()).sum(),
        'n_switches': (signals['switch_reason'] == 'BETTER_OPPORTUNITY').sum(),
        'n_tsl_triggers': (signals['switch_reason'] == 'TSL_TRIGGERED').sum(),
        'n_cash_exits': (signals['switch_reason'] == 'ALL_NEGATIVE').sum()
    }
    
    return results


def run_quick_backtest(predictions_dict, price_df, tbill_df, dates, tsl_pct=0.15, 
                       transaction_cost_bps=25, z_threshold=1.0):
    """
    Quick backtest with default strategy engine (for validation/testing)
    """
    etf_list = [k for k in predictions_dict.keys() if k not in ['SPY', 'AGG', '3MTBILL']]
    
    engine = StrategyEngine(
        etf_list=etf_list,
        tsl_pct=tsl_pct,
        transaction_cost_bps=transaction_cost_bps,
        z_score_threshold=z_threshold
    )
    
    return run_backtest(predictions_dict, price_df, tbill_df, engine, dates)
