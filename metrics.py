"""
Metrics Module — fixed
-----------------------
Key fixes:
  - All keys renamed to match streamlit_app.py:
      annualized_return → ann_return
      sharpe_ratio      → sharpe
      max_drawdown      → max_dd  (value)
      max_drawdown_date → max_dd_date
  - equity_curve forced to Series before .iloc indexing
  - hit_ratio uses 'actual_return' column which now contains
    real daily returns (including 0 on CASH days); excludes CASH days
    from the hit-ratio calculation so it measures ETF prediction accuracy
"""

import pandas as pd
import numpy as np


def calculate_metrics(equity_curve, returns, risk_free, audit_trail=None):
    """
    Parameters
    ----------
    equity_curve : pd.Series or pd.DataFrame
        Portfolio value starting at 1.0.  If DataFrame, 'strategy' column used.
    returns : pd.Series
        Daily strategy returns (decimal).
    risk_free : pd.Series
        Daily risk-free rate (decimal, already /252).
    audit_trail : pd.DataFrame, optional

    Returns
    -------
    dict  — all keys match streamlit_app.py expectations
    """

    # ── Force Series ──────────────────────────────────────────────────────────
    if isinstance(equity_curve, pd.DataFrame):
        equity_curve = equity_curve["strategy"]

    metrics = {}

    # 1. Annualized return
    n_days   = max(len(returns), 1)
    n_years  = n_days / 252
    start_v  = float(equity_curve.iloc[0])
    end_v    = float(equity_curve.iloc[-1])
    total_r  = end_v / start_v - 1 if start_v > 0 else 0.0
    ann_r    = float((1 + total_r) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    metrics["ann_return"]   = ann_r
    metrics["total_return"] = total_r

    # 2. Sharpe
    excess = returns - risk_free
    std    = float(excess.std())
    sharpe = float(np.sqrt(252) * excess.mean() / std) if std > 1e-10 else 0.0
    metrics["sharpe"] = sharpe

    # 3. Max Drawdown (peak-to-trough)
    peak       = equity_curve.cummax()
    drawdown   = (equity_curve - peak) / peak
    max_dd     = float(drawdown.min())
    max_dd_date = drawdown.idxmin()

    # Peak date of this drawdown
    try:
        peak_date = equity_curve.loc[:max_dd_date].idxmax()
    except Exception:
        peak_date = None

    metrics["max_dd"]            = max_dd
    metrics["max_dd_date"]       = max_dd_date
    metrics["max_dd_peak_date"]  = peak_date

    # 4. Worst single day
    metrics["worst_daily_return"] = float(returns.min())
    metrics["worst_daily_date"]   = returns.idxmin()

    # 5. Volatility
    metrics["volatility"] = float(returns.std() * np.sqrt(252))

    # 6. Win rate (all OOS days)
    wins              = int((returns > 0).sum())
    metrics["win_rate"]    = wins / n_days if n_days > 0 else 0.0

    # 7. Hit ratio — last 15 INVESTED days (excludes CASH)
    if audit_trail is not None and len(audit_trail) >= 1:
        invested = audit_trail[audit_trail.get("in_cash", pd.Series(False,
                       index=audit_trail.index)) == False]
        last15   = invested.tail(15)
        if len(last15) >= 5:
            positive = int((last15["actual_return"] > 0).sum())
            metrics["hit_ratio_15d"]    = positive / len(last15)
            metrics["hit_ratio_count"]  = positive
        else:
            metrics["hit_ratio_15d"]   = None
            metrics["hit_ratio_count"] = 0
    else:
        metrics["hit_ratio_15d"]   = None
        metrics["hit_ratio_count"] = 0

    # 8. Calmar
    metrics["calmar_ratio"] = (
        ann_r / abs(max_dd) if max_dd < -1e-10
        else (float("inf") if ann_r > 0 else 0.0)
    )

    # 9. Sortino
    down = returns[returns < 0]
    d_std = float(down.std()) if len(down) > 0 else 0.0
    metrics["sortino_ratio"] = (
        float(np.sqrt(252) * returns.mean() / d_std) if d_std > 1e-10
        else (float("inf") if returns.mean() > 0 else 0.0)
    )

    # 10. Avg win / loss
    avg_win  = float(returns[returns > 0].mean()) if wins > 0 else 0.0
    avg_loss = float(returns[returns < 0].mean()) if (returns < 0).sum() > 0 else 0.0
    metrics["avg_win"]       = avg_win
    metrics["avg_loss"]      = avg_loss
    metrics["win_loss_ratio"] = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    return metrics


def format_metrics_for_display(metrics):
    return {
        "Annualized Return": f"{metrics['ann_return']:.2%}",
        "Sharpe Ratio":      f"{metrics['sharpe']:.2f}",
        "Max Drawdown":      f"{metrics['max_dd']:.2%}",
        "Max DD Date":       str(metrics.get('max_dd_date', ''))[:10],
        "Worst Daily DD":    f"{metrics['worst_daily_return']:.2%}",
        "Win Rate":          f"{metrics['win_rate']:.1%}",
        "Calmar Ratio":      f"{metrics.get('calmar_ratio', 0):.2f}",
        "Total Return":      f"{metrics['total_return']:.2%}",
    }
