import pandas as pd
import numpy as np
from datetime import datetime
from functools import lru_cache
import pytz

try:
    import pandas_market_calendars as mcal
    NYSE_AVAILABLE = True
except ImportError:
    NYSE_AVAILABLE = False


@lru_cache(maxsize=1)
def _get_nyse_schedule():
    """Cached NYSE schedule — fetched once per process lifetime."""
    if not NYSE_AVAILABLE:
        return None
    nyse  = mcal.get_calendar("NYSE")
    end   = (datetime.today() + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    sched = nyse.schedule(start_date="2008-01-01", end_date=end)
    # Return tz-naive normalised index
    idx = sched.index.normalize()
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def get_latest_trading_day():
    idx = _get_nyse_schedule()
    if idx is not None:
        return idx[-1].date()
    d = datetime.today().date()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def get_next_trading_day(reference_date=None):
    """
    Return the correct next actionable NYSE session.

    Logic (mirrors your other projects):
      - If TODAY is a NYSE trading day AND current EST time < 4:00pm → return TODAY
      - Otherwise return the next NYSE trading day after today
    """
    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now(eastern)
    today   = pd.Timestamp(now_est.date())

    idx = _get_nyse_schedule()
    if idx is not None:
        # Is today a trading day?
        if today in idx:
            if now_est.hour < 16:          # before market close → signal is for today
                return today.date()
        # Find next trading day after today
        future = idx[idx > today]
        if len(future) > 0:
            return future[0].date()

    # Fallback: skip weekends
    candidate = today + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)
    return candidate.date()


def get_hero_next_date(predictions=None, etfs=None):
    """
    Next actionable NYSE session for the hero allocation box.
    Anchored to TODAY — not to last prediction date.
    """
    return get_next_trading_day()


def get_oos_index(equity, returns, rf_series, oos_start, oos_end):
    """
    Slice equity, returns, rf_series to OOS window.
    Uses intersection to avoid KeyErrors when indices differ.
    """
    mask    = (equity.index >= oos_start) & (equity.index <= oos_end)
    oos_idx = equity.index[mask]
    oos_idx = oos_idx.intersection(returns.index).intersection(rf_series.index)
    return equity.loc[oos_idx], returns.loc[oos_idx], rf_series.loc[oos_idx]


def annualized_return(equity_curve):
    if len(equity_curve) < 2:
        return 0.0
    total = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / 252
    return float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0


def sharpe_ratio(returns, rf_daily):
    excess = returns - rf_daily
    std    = excess.std()
    return float(np.sqrt(252) * excess.mean() / std) if std > 1e-10 else 0.0


def max_drawdown(equity_curve):
    """Returns (max_drawdown_pct, trough_date)."""
    peak     = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    dd_val   = float(drawdown.min())
    dd_date  = drawdown.idxmin()
    return dd_val, dd_date
