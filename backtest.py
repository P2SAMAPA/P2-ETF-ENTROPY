"""
Backtest Module — fixed
-----------------------
Key fixes:
  - signals.shift(1) removed: strategy_engine.generate_signals should already
    produce today's signal based on yesterday's prediction (look-ahead handled
    inside StrategyEngine). Double-shifting caused 2-day forward bias.
  - Last date: use today's price return (price_df.pct_change()) so March 4
    always gets its real return, not 0.
  - Audit trail enriched with switch_flag, expected_return, signal_z columns.
  - rf_series uses .reindex to avoid KeyError on missing dates.
"""

import pandas as pd
import numpy as np


def run_backtest(predictions_dict,
                 price_df,
                 tbill_df,
                 strategy_engine,
                 test_dates,
                 benchmarks=None):

    if benchmarks is None:
        benchmarks = []

    strategy_engine.reset()

    # generate_signals already handles look-ahead: signal on day T uses
    # prediction from day T-1 (shift inside StrategyEngine). Do NOT shift again.
    signals = strategy_engine.generate_signals(
        predictions_dict,
        price_df,
        test_dates,
    )

    # Pre-compute daily price returns for all ETFs (no forward bias)
    price_returns = price_df.pct_change()

    equity           = 1.0
    equity_curve     = []
    strategy_returns = []
    audit            = []

    for i, date in enumerate(test_dates):

        row      = signals.loc[date]
        selected = row["selected_etf"] if pd.notna(row.get("selected_etf")) else "CASH"

        # ── Daily return ──────────────────────────────────────────────────────
        if selected == "CASH":
            rf_rate      = _get_rf(tbill_df, date)
            daily_return = (1 + rf_rate) ** (1 / 252) - 1
        else:
            if date in price_returns.index and selected in price_returns.columns:
                daily_return = float(price_returns.loc[date, selected])
                if pd.isna(daily_return):
                    daily_return = 0.0
            else:
                daily_return = 0.0

            # Transaction cost on switch
            if row.get("switch_flag", False):
                daily_return -= strategy_engine.transaction_cost

        equity *= (1 + daily_return)

        equity_curve.append(equity)
        strategy_returns.append(daily_return)

        audit.append({
            "date":            date,
            "selected_etf":    selected,
            "actual_return":   daily_return,
            "switch_flag":     bool(row.get("switch_flag", False)),
            "expected_return": row.get("expected_return", None),
            "signal_z":        row.get("signal_z", None),
            "switch_reason":   row.get("switch_reason", ""),
            "in_cash":         selected == "CASH",
        })

    equity_series  = pd.Series(equity_curve,     index=test_dates)
    returns_series = pd.Series(strategy_returns, index=test_dates)
    equity_df      = pd.DataFrame({"strategy": equity_series})

    # Benchmarks (buy-and-hold, normalised to 1.0)
    for bench in benchmarks:
        if bench in price_df.columns:
            b_ret = price_returns[bench].reindex(test_dates).fillna(0)
            b_eq  = (1 + b_ret).cumprod()
            equity_df[bench] = b_eq / b_eq.iloc[0]

    rf_series = (
        tbill_df.reindex(test_dates).ffill().fillna(0) / 100 / 252
    )

    return {
        "equity_curve": equity_df,
        "returns":      returns_series,
        "risk_free":    rf_series,
        "audit_trail":  pd.DataFrame(audit).set_index("date"),
    }


def _get_rf(tbill_df, date):
    """Safe risk-free rate lookup — returns annualised decimal rate."""
    try:
        val = tbill_df.loc[date]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        return float(val) / 100 if pd.notna(val) else 0.0
    except Exception:
        return 0.0
