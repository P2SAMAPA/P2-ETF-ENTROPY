import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from utils import get_latest_trading_day
from backtest import run_backtest
from metrics import calculate_metrics
from strategy_engine import StrategyEngine
from feature_engineering import prepare_all_features
from data_loader import load_dataset
from transfer_voting import TransferVotingModel
import json
import os


st.set_page_config(layout="wide")


def get_next_trading_day(last_date):
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=last_date,
        end_date=last_date + pd.Timedelta(days=10)
    )
    future = schedule.index[schedule.index > last_date]
    return future[0]


def color_returns(val):
    return "color: green" if val > 0 else "color: red"


st.title("ETF Transfer Voting Engine")

year_start = st.sidebar.slider("Start Year", 2008, 2025, 2008)

tsl = st.sidebar.slider("TSL %", 10, 25, 15) / 100
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)

df = load_dataset()
df = df[df.index.year >= year_start]

model_info = json.load(open("artifacts/best_model.json"))
best_ma = model_info["best_ma_window"]

model = TransferVotingModel([], best_ma, "artifacts")
model.load(f"artifacts/transfer_voting_MA{best_ma}.pkl")

data_dict = prepare_all_features(df, best_ma)
features = data_dict["features"]

etf_list = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]

common_dates = None
for etf in etf_list:
    dates = features[etf].index
    common_dates = dates if common_dates is None else common_dates.intersection(dates)

n = len(common_dates)
test_dates = common_dates[int(n * 0.9):]

predictions = {}
for etf in etf_list:
    X_test = features[etf].loc[test_dates]
    preds = model.predict_single_etf(X_test, etf)
    predictions[etf] = pd.Series(preds, index=test_dates)

engine = StrategyEngine(etf_list, tsl_pct=tsl,
                        transaction_cost_bps=tx_cost)

results = run_backtest(
    predictions,
    df[etf_list + ["SPY", "AGG"]],
    df["3MTBILL"],
    engine,
    test_dates
)

metrics = calculate_metrics(
    results["equity_curve"]["strategy"],
    results["returns"],
    results["risk_free"],
    results["audit_trail"]
)

# Top Prediction
latest_date = df.index[-1]
next_day = get_next_trading_day(latest_date)

st.subheader("Next Trading Day Prediction")
st.write(f"**Date:** {next_day.date()}")
st.write(f"**Active MA Window:** MA({best_ma})")

# Metrics
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Ann Return", f"{metrics['annualized_return']:.2%}")
col2.metric("Sharpe", f"{metrics['sharpe_ratio']:.2f}")
col3.metric("Max DD", f"{metrics['max_drawdown']:.2%}")
col4.metric("Worst Daily", f"{metrics['worst_daily_return']:.2%}")
col5.metric("Hit Ratio 15d", f"{metrics['hit_ratio_15d']:.2%}")

# Equity Curve
st.subheader("Equity Curve (OOS)")
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(results["equity_curve"]["strategy"], label="Strategy", linewidth=2)
ax.plot(results["equity_curve"]["SPY"], label="SPY")
ax.plot(results["equity_curve"]["AGG"], label="AGG")
ax.legend()
ax.grid()

st.pyplot(fig)

# Audit Trail
st.subheader("Last 15 Trading Days")

audit = results["audit_trail"].tail(15).copy()
audit["actual_return"] = audit["actual_return"].astype(float)

st.dataframe(
    audit[["date", "predicted_etf", "actual_return"]]
    .style.applymap(color_returns, subset=["actual_return"]),
    use_container_width=True
)
