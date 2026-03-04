import pandas as pd
import numpy as np


def run_backtest(predictions_dict, price_df, tbill_df,
                 strategy_engine, test_dates,
                 benchmarks=["SPY", "AGG"]):

    strategy_engine.reset()
    signals = strategy_engine.generate_signals(
        predictions_dict, price_df, test_dates
    )

    equity = 1.0
    equity_curve = []
    strategy_returns = []
    audit = []

    for i in range(len(test_dates) - 1):

        date = test_dates[i]
        next_date = test_dates[i + 1]

        selected = signals.loc[date, "selected_etf"]

        if selected == "CASH":
            rf = tbill_df.loc[date] / 100
            daily_return = (1 + rf) ** (1 / 252) - 1
        else:
            price_today = price_df.loc[date, selected]
            price_next = price_df.loc[next_date, selected]
            daily_return = (price_next / price_today) - 1

            if signals.loc[date, "switch_reason"] is not None:
                daily_return -= strategy_engine.transaction_cost

        equity *= (1 + daily_return)

        equity_curve.append(equity)
        strategy_returns.append(daily_return)

        audit.append({
            "date": date,
            "predicted_etf": selected,
            "actual_return": daily_return
        })

    equity_series = pd.Series(equity_curve, index=test_dates[:-1])
    returns_series = pd.Series(strategy_returns, index=test_dates[:-1])

    # Benchmarks
    equity_df = pd.DataFrame({"strategy": equity_series})

    for bench in benchmarks:
        if bench in price_df.columns:
            bench_ret = price_df[bench].pct_change().shift(-1)
            bench_equity = (1 + bench_ret.loc[test_dates[:-1]]).cumprod()
            bench_equity /= bench_equity.iloc[0]
            equity_df[bench] = bench_equity

    rf_series = tbill_df.loc[test_dates[:-1]] / 100 / 252

    return {
        "equity_curve": equity_df,
        "returns": returns_series,
        "risk_free": rf_series,
        "audit_trail": pd.DataFrame(audit)
    }
