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
    
    Parameters:
    -----------
    current_date : pd.Timestamp or datetime
        The current date to find the next trading day from
        
    Returns:
    --------
    pd.Timestamp
        The next valid trading day
    """
    trading_days = get_nyse_trading_days()
    # Convert to pandas Timestamp for comparison
    current = pd.Timestamp(current_date)
    
    # Find the next trading day strictly after current_date
    future_days = trading_days[trading_days > current]
    
    if len(future_days) > 0:
        return future_days[0]
    else:
        # If no future trading days found (edge case), return current + 1 day
        return current + pd.Timedelta(days=1)
