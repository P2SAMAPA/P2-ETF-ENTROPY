import pandas as pd
import numpy as np


def run_backtest(predictions_dict,
                 price_df,
                 tbill_df,
                 strategy_engine,
                 test_dates,
                 benchmarks=["SPY", "AGG"]):

    strategy_engine.reset()

    signals = strategy_engine.generate_signals(
        predictions_dict,
        price_df,
        test_dates
    )

    # FIX 1 — remove look-ahead bias (execute signal next day)
    signals = signals.shift(1)

    equity = 1.0
    equity_curve = []
    strategy_returns = []
    audit = []

    for i, date in enumerate(test_dates):

        selected = signals.loc[date, "selected_etf"]

        if i < len(test_dates) - 1:
            next_date = test_dates[i + 1]

            if selected == "CASH":
                rf = tbill_df.loc[date] / 100
                daily_return = (1 + rf) ** (1 / 252) - 1
            else:
                price_today = price_df.loc[date, selected]
                price_next = price_df.loc[next_date, selected]

                daily_return = (price_next / price_today) - 1

                # Apply transaction cost on switch
                if signals.loc[date, "switch_flag"]:
                    daily_return -= strategy_engine.transaction_cost

            equity *= (1 + daily_return)
        else:
            # Final day (no next price available)
            daily_return = 0

        equity_curve.append(equity)
        strategy_returns.append(daily_return)

        # FIX 2 — correct column name expected by app.py
        audit.append({
            "date": date,
            "selected_etf": selected,
            "actual_return": daily_return
        })

    equity_series = pd.Series(equity_curve, index=test_dates)
    returns_series = pd.Series(strategy_returns, index=test_dates)

    equity_df = pd.DataFrame({"strategy": equity_series})

    # Benchmarks
    for bench in benchmarks:
        if bench in price_df.columns:
            bench_ret = price_df[bench].pct_change().shift(-1)
            bench_equity = (1 + bench_ret.loc[test_dates]).cumprod()
            bench_equity /= bench_equity.iloc[0]
            equity_df[bench] = bench_equity

    rf_series = tbill_df.loc[test_dates] / 100 / 252

    return {
        "equity_curve": equity_df,
        "returns": returns_series,
        "risk_free": rf_series,
        # FIX 3 — set datetime index so Streamlit filtering works
        "audit_trail": pd.DataFrame(audit).set_index("date")
    }
