#!/usr/bin/env python3
"""
P2-ETF-ENTROPY
Production Streamlit App
HF-backed inference architecture
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import timedelta

import pandas_market_calendars as mcal
from huggingface_hub import hf_hub_download

from feature_engineering import prepare_all_features
from transfer_voting import TransferVotingModel
from strategy_engine import StrategyEngine
from backtest import run_backtest

# ==========================================================
# CONFIG
# ==========================================================

HF_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ['TLT', 'VNQ', 'GLD', 'SLV', 'VCIT', 'HYG', 'LQD']

st.set_page_config(layout="wide")

# ==========================================================
# LOAD FROM HUGGINGFACE
# ==========================================================

@st.cache_resource
def load_from_hf():

    # Download dataset
    raw_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="raw_data.parquet",
        repo_type="dataset"
    )

    # Download best model metadata
    best_model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="models/best_model.json",
        repo_type="dataset"
    )

    with open(best_model_path, "r") as f:
        best_info = json.load(f)

    best_ma = best_info["best_ma_window"]

    # Download only selected transfer model
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"models/transfer_voting_MA{best_ma}.pkl",
        repo_type="dataset"
    )

    model = joblib.load(model_path)
    raw_df = pd.read_parquet(raw_path)

    return raw_df, model, best_ma


raw_df, model, best_ma = load_from_hf()

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

data_dict = prepare_all_features(raw_df, ma_window=best_ma)

features_dict = data_dict["features"]
targets_dict = data_dict["targets"]

# Common dates
common_dates = None
for etf in ETF_LIST:
    dates = features_dict[etf].index
    common_dates = dates if common_dates is None else common_dates.intersection(dates)

common_dates = pd.DatetimeIndex(sorted(common_dates))

price_df = raw_df[ETF_LIST].loc[common_dates]
tbill_df = raw_df["3MTBILL"].loc[common_dates]

# ==========================================================
# GENERATE PREDICTIONS
# ==========================================================

predictions_dict = {}

for etf in ETF_LIST:
    X = features_dict[etf].loc[common_dates]
    preds = model.predict_single_etf(X, etf)
    predictions_dict[etf] = pd.Series(preds, index=common_dates)

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("Strategy Controls")

tsl_slider = st.sidebar.slider("Trailing Stop Loss (%)", 10, 25, 15)
transaction_slider = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)
z_slider = st.sidebar.slider("Z-Score Re-entry Threshold", 0.5, 2.0, 1.0)

# ==========================================================
# RUN STRATEGY
# ==========================================================

engine = StrategyEngine(
    etf_list=ETF_LIST,
    tsl_pct=tsl_slider,
    transaction_cost_bps=transaction_slider,
    z_score_threshold=z_slider
)

results = run_backtest(
    predictions_dict,
    price_df,
    tbill_df,
    engine,
    common_dates
)

equity_curve = results["equity_curve"]
audit_df = results["audit_trail"]

# ==========================================================
# HERO BOX
# ==========================================================

last_prediction_date = common_dates.max()

nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(
    start_date=last_prediction_date + timedelta(days=1),
    end_date=last_prediction_date + timedelta(days=10)
)

next_trading_day = schedule.index[0].date()
latest_signal = audit_df.iloc[-1]["predicted_etf"]

st.markdown("## 📈 Next Trading Day Prediction")

st.markdown(f"""
<div style="
background:#111827;
padding:30px;
border-radius:12px;
text-align:center;
font-size:26px;
font-weight:600;
color:white;">
Predicted ETF for {next_trading_day}:<br><br>
<span style="font-size:40px;color:#22C55E;">
{latest_signal}
</span>
</div>
""", unsafe_allow_html=True)

# ==========================================================
# EQUITY CURVE
# ==========================================================

st.markdown("## 📊 Strategy vs Benchmarks")
st.line_chart(equity_curve)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

returns = results["returns"]

annual_return = (1 + returns.mean()) ** 252 - 1
annual_vol = returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol if annual_vol > 0 else 0

rolling_max = equity_curve["strategy"].cummax()
drawdown = equity_curve["strategy"] / rolling_max - 1

max_dd = drawdown.min()
max_dd_date = drawdown.idxmin().date()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Annual Return", f"{annual_return*100:.2f}%")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
col4.metric("Max DD Date", f"{max_dd_date}")

# ==========================================================
# AUDIT TABLE
# ==========================================================

st.markdown("## 🧾 Audit Trail")

audit_display = audit_df.copy()
audit_display["actual_return"] *= 100

def color_returns(val):
    if val > 0:
        return "color:#00B050;font-weight:600;"
    elif val < 0:
        return "color:#C00000;font-weight:600;"
    return ""

styled = (
    audit_display.style
    .format({"actual_return": "{:.2f}%"})
    .applymap(color_returns, subset=["actual_return"])
)

st.dataframe(styled, use_container_width=True)

# ==========================================================
# METHODOLOGY
# ==========================================================

with st.expander("📘 Methodology", expanded=False):
    st.markdown(f"""
### Model
- Transfer Voting Regression
- Optimized MA window: **MA({best_ma})**

### Allocation Logic
- Cross-sectional expected return ranking
- Peak-based trailing stop
- Z-score re-entry
- Transaction cost friction

### Risk Controls
- Move to cash when all predictions negative
- 2× transaction cost switching filter

### Deployment
- Training: GitHub Actions
- Storage: HuggingFace Dataset
- Inference: Streamlit
""")

st.markdown("---")
st.markdown("P2-ETF-ENTROPY | Production Quant Allocation Framework")
