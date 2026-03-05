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
def _get_nyse_schedule(start_date="2008-01-01"):
    """Cached NYSE schedule — fetched once per process."""
    if not NYSE_AVAILABLE:
        return None
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=start_date,
        end_date=(datetime.today() + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    )
    return schedule.index


def get_latest_trading_day():
    idx = _get_nyse_schedule()
    if idx is not None:
        return idx[-1].date()
    # Fallback: skip weekends
    d = datetime.today().date()
    while d.weekday() >= 5:
        d -= pd.Timedelta(days=1)
    return d


def get_next_trading_day(current_date):
    """
    Return the next NYSE trading day AFTER current_date.
    Uses today's actual date as anchor so the hero box is always forward-looking.
    """
    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now(eastern)
    today   = pd.Timestamp(now_est.date())

    idx = _get_nyse_schedule()
    if idx is not None:
        # Normalise tz
        trading_days = idx.normalize()
        if trading_days.tz is not None:
            trading_days = trading_days.tz_localize(None)
        future = trading_days[trading_days > today]
        if len(future) > 0:
            return future[0].date()

    # Fallback: skip weekends
    candidate = today + pd.Timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += pd.Timedelta(days=1)
    return candidate.date()


def get_hero_next_date(predictions, etfs):
    """
    Return the next NYSE trading day from TODAY (not from last prediction date).
    The hero box should always show the next actionable session.
    """
    # Find any valid ETF in predictions
    ref = next((e for e in etfs if e in predictions and len(predictions[e]) > 0), None)
    if ref is None:
        return get_next_trading_day(datetime.today().date())
    # Anchor to today — we want the next session from now
    return get_next_trading_day(datetime.today().date())


def get_oos_index(equity, returns, rf_series, oos_start, oos_end):
    """
    Slice equity, returns, and rf_series to the OOS window.
    Intersection ensures no KeyErrors even if indices differ slightly.
    """
    mask      = (equity.index >= oos_start) & (equity.index <= oos_end)
    oos_idx   = equity.index[mask]
    oos_idx   = oos_idx.intersection(returns.index).intersection(rf_series.index)

    return equity.loc[oos_idx], returns.loc[oos_idx], rf_series.loc[oos_idx]


def annualized_return(equity_curve):
    if len(equity_curve) < 2:
        return 0.0
    total = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / 252
    return (1 + total) ** (1 / years) - 1 if years > 0 else 0.0


def sharpe_ratio(returns, rf_daily):
    excess = returns - rf_daily
    std    = excess.std()
    return float(np.sqrt(252) * excess.mean() / std) if std > 0 else 0.0


def max_drawdown(equity_curve):
    peak     = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min(), drawdown.idxmin()
