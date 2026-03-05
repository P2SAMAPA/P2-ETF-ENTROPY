"""
ETF Transfer Voting Engine
Robust Production Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from huggingface_hub import hf_hub_download
import pytz
from datetime import datetime

from data_loader import load_dataset, load_metadata
from update_data import incremental_update
from feature_engineering import prepare_all_features
from strategy_engine import StrategyEngine
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

st.set_page_config(page_title="ETF Transfer Voting Engine", layout="wide")
st.title("📈 ETF Transfer Voting Engine")

HF_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]


# --------------------------------------------------
# EST CLOCK
# --------------------------------------------------

def get_next_nyse_session():
    eastern = pytz.timezone("US/Eastern")
    now_est = datetime.now(eastern)

    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=now_est.date(),
        end_date=now_est.date() + pd.Timedelta(days=10)
    )

    today = pd.Timestamp(now_est.date())

    if today in schedule.index:
        if now_est.hour >= 16:
            future = schedule.index[schedule.index > today]
            return future[0]
        else:
            return today
    else:
        future = schedule.index[schedule.index > today]
        return future[0]


# --------------------------------------------------
# SIDEBAR DATA MANAGEMENT
# --------------------------------------------------

st.sidebar.header("🔄 Data Management")

metadata = load_metadata()

if metadata:
    st.sidebar.info(f"Dataset updated till {metadata['last_data_update']}")
else:
    st.sidebar.warning("Metadata not found")

if st.sidebar.button("Refresh Dataset"):
    with st.spinner("Updating dataset..."):
        incremental_update()
        st.sidebar.success("Dataset refreshed")
        st.rerun()


# --------------------------------------------------
# STRATEGY CONTROLS
# --------------------------------------------------

st.sidebar.header("⚙️ Strategy Controls")

year_start = st.sidebar.slider("Start Year", 2008, 2025, 2008)
tsl_pct = st.sidebar.slider("Trailing Stop Loss (%)", 10, 25, 15)
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)
z_threshold = st.sidebar.slider("Z-Score Re-entry", 0.5, 2.0, 1.0)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return load_dataset()

df = load_data()
df = df[df.index.year >= year_start]


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model_and_meta():
    os.makedirs("artifacts", exist_ok=True)

    meta_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="models/best_model.json",
        repo_type="dataset",
        local_dir="artifacts"
    )

    with open(meta_path, "r") as f:
        model_info = json.load(f)

    best_ma = model_info["best_ma_window"]

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"models/transfer_voting_MA{best_ma}.pkl",
        repo_type="dataset",
        local_dir="artifacts"
    )

    model = TransferVotingModel(ETF_LIST, best_ma, "artifacts")
    model.load(model_path)

    return model, model_info

model, model_info = load_model_and_meta()


# --------------------------------------------------
# OOS WINDOW SAFE HANDLING
# --------------------------------------------------

if "oos_start_date" in model_info:
    oos_start = pd.to_datetime(model_info["oos_start_date"])
    oos_end = pd.to_datetime(model_info["oos_end_date"])
else:
    # fallback: last 10%
    n = len(df)
    oos_start = df.index[int(n * 0.9)]
    oos_end = df.index[-1]


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

data_dict = prepare_all_features(df, ma_window=model_info["best_ma_window"])
features = data_dict["features"]

common_dates = None
for etf in ETF_LIST:
    idx = features[etf].index
    common_dates = idx if common_dates is None else common_dates.intersection(idx)

common_dates = common_dates.sort_values()

price_df = df[ETF_LIST].loc[common_dates]
tbill_df = df["3MTBILL"].loc[common_dates]


# --------------------------------------------------
# PREDICTIONS
# --------------------------------------------------

predictions = {}
for etf in ETF_LIST:
    X = features[etf].loc[common_dates]
    preds = model.predict_single_etf(X, etf)
    predictions[etf] = pd.Series(preds, index=common_dates)


# --------------------------------------------------
# BACKTEST
# --------------------------------------------------

engine = StrategyEngine(
    ETF_LIST,
    tsl_pct=tsl_pct,
    transaction_cost_bps=tx_cost,
    z_score_threshold=z_threshold
)

results = run_backtest(
    predictions,
    price_df,
    tbill_df,
    engine,
    common_dates
)

equity_curve = results["equity_curve"].loc[oos_start:oos_end]
returns_oos = results["returns"].loc[oos_start:oos_end]
risk_free_oos = results["risk_free"].loc[oos_start:oos_end]

metrics = calculate_metrics(
    equity_curve["strategy"],
    returns_oos,
    risk_free_oos,
    results["audit_trail"]
)


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

st.subheader("Next Allocation")

next_session = get_next_nyse_session()

latest_features = {etf: features[etf].iloc[-1:] for etf in ETF_LIST}
expected_returns = {}

for etf in ETF_LIST:
    pred = model.predict_single_etf(latest_features[etf], etf)[0]
    current_price = df.iloc[-1][etf]
    expected_return = pred / current_price
    expected_returns[etf] = expected_return

predicted_etf = max(expected_returns, key=expected_returns.get)

st.markdown(f"### {next_session.date()} → {predicted_etf}")

st.markdown("---")
st.subheader("Equity Curve (OOS Only)")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(equity_curve["strategy"])
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.subheader("Last 15 OOS Trades")
st.dataframe(results["audit_trail"].tail(15))
