"""
Streamlit App - ETF Transfer Voting Strategy
Production Version (HF + Cloud Safe)
ETF Transfer Voting Engine - Production Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from huggingface_hub import hf_hub_download

from data_loader import load_dataset
from data_loader import load_dataset, load_metadata
from update_data import incremental_update
from feature_engineering import prepare_all_features
from strategy_engine import StrategyEngine
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics
from utils import get_latest_trading_day


# ---------------------------------------------------
# --------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ETF Transfer Voting Engine",
    page_icon="📈",
    layout="wide"
)
# --------------------------------------------------

st.set_page_config(layout="wide", page_icon="📈")
st.title("📈 ETF Transfer Voting Engine")


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def get_next_trading_day(last_date):
    nyse = mcal.get_calendar("NYSE")
    schedule = nyse.schedule(
        start_date=last_date,
        end_date=last_date + pd.Timedelta(days=10)
    )
    future = schedule.index[schedule.index > last_date]
    return future[0] if len(future) > 0 else None


def color_returns(val):
    if val > 0:
        return "color: green"
    elif val < 0:
        return "color: red"
    return "color: black"


@st.cache_resource
def load_model_from_hf():
    os.makedirs("artifacts", exist_ok=True)

    meta_path = hf_hub_download(
        repo_id="P2SAMAPA/etf-entropy-dataset",
        filename="models/best_model.json",
        repo_type="dataset",
        local_dir="artifacts"
    )
# --------------------------------------------------
# DATA REFRESH SECTION
# --------------------------------------------------

    with open(meta_path, "r") as f:
        model_info = json.load(f)
def check_data_freshness():
    metadata = load_metadata()
    if metadata is None:
        return False, "No dataset found", None

    best_ma = model_info["best_ma_window"]
    last_update = pd.to_datetime(metadata["last_data_update"])
    latest_trading = pd.to_datetime(get_latest_trading_day())

    model_filename = f"models/transfer_voting_MA{best_ma}.pkl"
    if last_update >= latest_trading:
        return True, f"Dataset updated till {last_update.date()}", last_update.date()
    else:
        return False, f"Dataset stale: {last_update.date()}", last_update.date()

    model_path = hf_hub_download(
        repo_id="P2SAMAPA/etf-entropy-dataset",
        filename=model_filename,
        repo_type="dataset",
        local_dir="artifacts"
    )

    model = TransferVotingModel([], best_ma, "artifacts")
    model.load(model_path)
st.sidebar.header("🔄 Data Management")

    return model, best_ma
is_fresh, freshness_msg, last_date = check_data_freshness()
st.sidebar.info(freshness_msg)

if st.sidebar.button("Refresh Dataset"):
    with st.spinner("Checking for incremental updates..."):
        if is_fresh:
            st.sidebar.success(freshness_msg)
        else:
            incremental_update()
            st.sidebar.success("Dataset updated successfully.")
            st.rerun()

@st.cache_data
def load_data_cached():
    return load_dataset()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
# --------------------------------------------------
# STRATEGY CONTROLS
# --------------------------------------------------

st.sidebar.header("⚙️ Strategy Controls")

year_start = st.sidebar.slider("Start Year", 2008, 2025, 2008)

tsl_pct = st.sidebar.slider("Trailing Stop Loss (%)", 10, 25, 15) / 100
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)
z_threshold = st.sidebar.slider("Z-Score Re-entry", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.write("Data Source: HuggingFace Dataset")
st.sidebar.write("Model: Transfer Voting")

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

# ---------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------

df = load_data_cached()
df = load_dataset()

if df is None:
    st.error("Dataset not found.")
    st.error("Dataset not available.")
    st.stop()

df = df[df.index.year >= year_start]

try:
    model, best_ma = load_model_from_hf()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# --------------------------------------------------
# LOAD MODEL FROM HUGGINGFACE
# --------------------------------------------------

os.makedirs("artifacts", exist_ok=True)

meta_path = hf_hub_download(
    repo_id="P2SAMAPA/etf-entropy-dataset",
    filename="models/best_model.json",
    repo_type="dataset",
    local_dir="artifacts"
)

with open(meta_path, "r") as f:
    model_info = json.load(f)

# ---------------------------------------------------
# PREP FEATURES
# ---------------------------------------------------
best_ma = model_info["best_ma_window"]

model_path = hf_hub_download(
    repo_id="P2SAMAPA/etf-entropy-dataset",
    filename=f"models/transfer_voting_MA{best_ma}.pkl",
    repo_type="dataset",
    local_dir="artifacts"
)

model = TransferVotingModel([], best_ma, "artifacts")
model.load(model_path)


# --------------------------------------------------
# FEATURE PREPARATION
# --------------------------------------------------

etf_list = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]

@@ -151,9 +139,9 @@ def load_data_cached():
    predictions[etf] = pd.Series(preds, index=test_dates)


# ---------------------------------------------------
# RUN BACKTEST
# ---------------------------------------------------
# --------------------------------------------------
# BACKTEST
# --------------------------------------------------

engine = StrategyEngine(
    etf_list,
@@ -178,124 +166,164 @@ def load_data_cached():
)


# ---------------------------------------------------
# TABS
# ---------------------------------------------------
# --------------------------------------------------
# HERO PREDICTION BOX
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Audit Trail", "📖 Methodology"])
latest_date = df.index[-1]

latest_features = {etf: features[etf].iloc[-1:] for etf in etf_list}

# ---------------------------------------------------
# DASHBOARD TAB
# ---------------------------------------------------
expected_returns = {}
for etf in etf_list:
    pred = model.predict_single_etf(latest_features[etf], etf)[0]
    price = df.loc[latest_date, etf]
    expected_returns[etf] = pred / price if price > 0 else 0

with tab1:
predicted_etf = max(expected_returns, key=expected_returns.get)
expected_ret = expected_returns[predicted_etf]

    st.subheader("Next Trading Day Prediction")
nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(
    start_date=latest_date,
    end_date=latest_date + pd.Timedelta(days=10)
)
future = schedule.index[schedule.index > latest_date]
next_trading_day = future[0]

    latest_date = df.index[-1]
    next_day = get_next_trading_day(latest_date)
st.markdown("## 📅 Next Trading Day Allocation")

    colA, colB = st.columns(2)
st.markdown(f"""
<div style="background-color:#f4f8fb;
            padding:35px;
            border-radius:18px;
            text-align:center;
            border:3px solid #1f77b4;">
    <div style="font-size:18px;margin-bottom:10px;">
        {next_trading_day.date()}
    </div>
    <div style="font-size:54px;font-weight:bold;color:#1f77b4;">
        {predicted_etf}
    </div>
    <div style="font-size:18px;margin-top:10px;">
        Expected Return: {expected_ret:.2%}
    </div>
</div>
""", unsafe_allow_html=True)

    with colA:
        st.write(f"**Latest Data:** {latest_date.date()}")
        if next_day:
            st.write(f"**Next Trading Day:** {next_day.date()}")

    with colB:
        st.write(f"**Active MA Window:** MA({best_ma})")
        st.write(f"**ETFs Tracked:** {len(etf_list)}")
# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

    st.markdown("---")
    st.subheader("Performance Metrics (OOS)")
st.markdown("---")
st.subheader("Performance Metrics (OOS)")

    col1, col2, col3, col4, col5 = st.columns(5)
col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col4.metric("Worst Daily Return", f"{metrics['worst_daily_return']:.2%}")
    col5.metric("Hit Ratio (15d)", f"{metrics['hit_ratio_15d']:.2%}")
col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
col4.metric("Worst Daily Return", f"{metrics['worst_daily_return']:.2%}")
col5.metric("Hit Ratio (15d)", f"{metrics['hit_ratio_15d']:.2%}")

    st.markdown("---")
    st.subheader("Equity Curve")
st.caption(f"Worst Daily Return Date: {metrics.get('worst_daily_date')}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(results["equity_curve"]["strategy"], label="Strategy", linewidth=2)
    ax.plot(results["equity_curve"]["SPY"], label="SPY", alpha=0.7)
    ax.plot(results["equity_curve"]["AGG"], label="AGG", alpha=0.7)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
# --------------------------------------------------
# EQUITY CURVE
# --------------------------------------------------

st.markdown("---")
st.subheader("Equity Curve")

# ---------------------------------------------------
# AUDIT TAB
# ---------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(results["equity_curve"]["strategy"], label="Strategy", linewidth=2)
ax.plot(results["equity_curve"]["SPY"], label="SPY", alpha=0.7)
ax.plot(results["equity_curve"]["AGG"], label="AGG", alpha=0.7)
ax.legend()
ax.grid(True)

with tab2:
st.pyplot(fig)

    st.subheader("Last 15 Trading Days")

    audit = results["audit_trail"].tail(15).copy()
    audit["actual_return"] = audit["actual_return"].astype(float)
# --------------------------------------------------
# AUDIT TRAIL
# --------------------------------------------------

    styled = audit[["date", "predicted_etf", "actual_return"]].style.applymap(
        color_returns, subset=["actual_return"]
    )
st.markdown("---")
st.subheader("Last 15 Trading Days Audit Trail")

    st.dataframe(styled, use_container_width=True)
audit = results["audit_trail"].tail(15).copy()
st.dataframe(
    audit[["date", "predicted_etf", "actual_return"]],
    use_container_width=True
)


# ---------------------------------------------------
# METHODOLOGY TAB
# ---------------------------------------------------
# --------------------------------------------------
# COLLAPSIBLE METHODOLOGY
# --------------------------------------------------

with tab3:
st.markdown("---")

    st.subheader("Strategy Methodology")
with st.expander("📖 View Strategy Methodology", expanded=False):

    st.markdown("""
### 📄 Research Foundation
**Flexible Target Prediction for Quantitative Trading in the American Stock Market**  
Journal: *Entropy (2026)*  

**Flexible Target Prediction for Quantitative Trading in the American Stock Market:  
A Hybrid Framework Integrating Ensemble Models, Fusion Models and Transfer Learning**  
Journal: *Entropy (2026)*

---

### 🎯 Target Engineering
Instead of predicting raw prices (high entropy), we predict:
### 🎯 Core Innovation

Instead of predicting raw closing prices (high entropy and noisy),  
the model predicts:

MA_Diff(t+1) = MA(t+1) − MA(t)
**MA_Diff(t+1) = MA(t+1) − MA(t)**

This reduces noise and improves signal stability.
This reduces variance and improves predictive stability.

MA(3) and MA(5) are tested; the window with higher out-of-sample  
annualized return is selected.

---

### 🧠 Model Architecture
- RandomForest  

**Base Models**
- Random Forest  
- XGBoost  
- LightGBM  
- AdaBoost  
- Decision Tree  

Simple average voting → Transfer Voting weighting using similarity.
**Ensemble Layer**
- Simple average voting  

**Transfer Voting**
- Cross-ETF similarity weighting  
- Dynamic Time Warping (DTW)  

---

### ⚙️ Trading Logic
1. Select ETF with highest expected return  
2. If all negative → allocate to CASH (3M T-Bill)  
3. Apply Trailing Stop Loss  
4. Re-entry via Z-score threshold  

1. Select ETF with highest predicted expected return  
2. If all predictions negative → allocate to CASH (3M T-Bill)  
3. Apply trailing stop loss  
4. Re-enter via Z-score threshold  
5. Apply transaction cost friction  

---

### 🔄 Automation

- Daily update: 00:30 UTC  
- Weekly retraining  
- HuggingFace dataset storage  
- GitHub Actions CI/CD  
- GitHub Actions CI/CD pipeline  
""")
