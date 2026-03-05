import pandas as pd
import numpy as np
import pandas_market_calendars as mcal
from datetime import datetime

def get_nyse_trading_days(start_date="2008-01-01"):
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(start_date=start_date, end_date=datetime.today())
    return schedule.index

def get_latest_trading_day():
    trading_days = get_nyse_trading_days()
    return trading_days[-1].date()

def annualized_return(equity_curve):
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / 252
    return (1 + total_return) ** (1 / years) - 1

def sharpe_ratio(returns, rf_daily):
    excess = returns - rf_daily
    return np.sqrt(252) * excess.mean() / excess.std()

def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    return max_dd, max_dd_date

def get_next_trading_day(current_date):
    """
    Get the next trading day after the given date.
    """
    trading_days = get_nyse_trading_days()
    current = pd.Timestamp(current_date)
    future_days = trading_days[trading_days > current]
    if len(future_days) > 0:
        return future_days[0]
    else:
        return current + pd.Timedelta(days=1)

# ── NEW HELPERS ───────────────────────────────────────────────────────────────

def get_hero_next_date(predictions, etfs):
    """
    Return the correct next date for hero box, based on last prediction index.
    """
    last_pred_date = sorted(list(predictions[etfs[0]].index))[-1]
    return get_next_trading_day(last_pred_date)

def get_oos_index(equity, returns, rf_series, oos_start, oos_end):
    """
    Return aligned OOS indices for equity, returns, and risk-free rates.
    Ensures slices are not empty after look-ahead adjustments.
    """
    oos_mask = (equity.index >= oos_start) & (equity.index <= oos_end)
    oos_index = equity.index[oos_mask]

    # Keep only dates that exist in returns and risk-free
    oos_index = oos_index.intersection(returns.index)
    
    equity_oos  = equity.loc[oos_index]
    returns_oos = returns.loc[oos_index]
    rf_oos      = rf_series.loc[oos_index]
    
    return equity_oos, returns_oos, rf_oos
