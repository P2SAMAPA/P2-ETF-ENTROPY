"""
ETF Transfer Voting Engine
Production Version (GitHub Training + HF Storage + Streamlit Inference)
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
from huggingface_hub import hf_hub_download

from data_loader import load_dataset, load_metadata
from update_data import incremental_update
from feature_engineering import prepare_all_features
from strategy_engine import StrategyEngine
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics
from utils import get_latest_trading_day


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="ETF Transfer Voting Engine",
    page_icon="📈",
    layout="wide"
)

st.title("📈 ETF Transfer Voting Engine")

HF_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]


# --------------------------------------------------
# HELPERS
# --------------------------------------------------

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
    return ""


# --------------------------------------------------
# DATA REFRESH SECTION
# --------------------------------------------------

st.sidebar.header("🔄 Data Management")


def check_data_freshness():
    metadata = load_metadata()
    if metadata is None:
        return False, "No dataset metadata found.", None

    last_update = pd.to_datetime(metadata["last_data_update"])
    latest_trading = pd.to_datetime(get_latest_trading_day())

    if last_update >= latest_trading:
        return True, f"Dataset updated till {last_update.date()}", last_update.date()
    else:
        return False, f"Dataset stale (last: {last_update.date()})", last_update.date()


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


# --------------------------------------------------
# STRATEGY CONTROLS
# --------------------------------------------------

st.sidebar.header("⚙️ Strategy Controls")

year_start = st.sidebar.slider("Start Year", 2008, 2025, 2008)
tsl_pct = st.sidebar.slider("Trailing Stop Loss (%)", 10, 25, 15)
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)
z_threshold = st.sidebar.slider("Z-Score Re-entry", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.write("Data Source: HuggingFace Dataset")
st.sidebar.write("Model: Transfer Voting")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    return load_dataset()


df = load_data()

if df is None:
    st.error("Dataset not found.")
    st.stop()

df = df[df.index.year >= year_start]


# --------------------------------------------------
# LOAD MODEL FROM HF
# --------------------------------------------------

@st.cache_resource
def load_model_from_hf():

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

    return model, best_ma


model, best_ma = load_model_from_hf()


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

data_dict = prepare_all_features(df, ma_window=best_ma)
features = data_dict["features"]

common_dates = None
for etf in ETF_LIST:
    idx = features[etf].index
    if common_dates is None:
        common_dates = idx
    else:
        common_dates = common_dates.intersection(idx)

common_dates = common_dates.sort_values()

price_df = df[ETF_LIST].loc[common_dates]
tbill_df = df["3MTBILL"].loc[common_dates]


# --------------------------------------------------
# GENERATE PREDICTIONS
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

metrics = calculate_metrics(
    results["equity_curve"]["strategy"],
    results["returns"],
    results["risk_free"],
    results["audit_trail"]
)


# --------------------------------------------------
# TABS
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Audit Trail", "📖 Methodology"])


# ==================================================
# DASHBOARD TAB
# ==================================================

with tab1:

    latest_date = common_dates[-1]
    next_day = get_next_trading_day(latest_date)

    latest_features = {etf: features[etf].iloc[-1:] for etf in ETF_LIST}

    expected_returns = {}
    for etf in ETF_LIST:
        pred = model.predict_single_etf(latest_features[etf], etf)[0]
        price = df.loc[latest_date, etf]
        expected_returns[etf] = pred / price if price > 0 else 0

    predicted_etf = max(expected_returns, key=expected_returns.get)
    expected_ret = expected_returns[predicted_etf]

    st.subheader("Next Trading Day Allocation")

    st.markdown(f"""
    <div style="background-color:#f4f8fb;
                padding:35px;
                border-radius:18px;
                text-align:center;
                border:3px solid #1f77b4;">
        <div style="font-size:18px;margin-bottom:10px;">
            {next_day.date() if next_day else ""}
        </div>
        <div style="font-size:54px;font-weight:bold;color:#1f77b4;">
            {predicted_etf}
        </div>
        <div style="font-size:18px;margin-top:10px;">
            Expected Return: {expected_ret:.2%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Performance Metrics (OOS)")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col4.metric("Worst Daily Return", f"{metrics['worst_daily_return']:.2%}")
    col5.metric("Hit Ratio (15d)", f"{metrics['hit_ratio_15d']:.2%}")

    st.caption(f"Worst Daily Return Date: {metrics.get('worst_daily_date')}")

    st.markdown("---")
    st.subheader("Equity Curve")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(results["equity_curve"]["strategy"], label="Strategy", linewidth=2)
    ax.plot(results["equity_curve"]["SPY"], label="SPY", alpha=0.7)
    ax.plot(results["equity_curve"]["AGG"], label="AGG", alpha=0.7)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ==================================================
# AUDIT TAB
# ==================================================

with tab2:

    st.subheader("Last 15 Trading Days")

    audit = results["audit_trail"].tail(15).copy()
    audit["actual_return"] = audit["actual_return"].astype(float)

    styled = audit[["date", "predicted_etf", "actual_return"]].style.applymap(
        color_returns, subset=["actual_return"]
    )

    st.dataframe(styled, use_container_width=True)


# ==================================================
# METHODOLOGY TAB
# ==================================================

with tab3:

    st.subheader("Strategy Methodology")

    st.markdown("""
### 📄 Research Foundation
**Flexible Target Prediction for Quantitative Trading in the American Stock Market**  
Journal: *Entropy (2026)*  

---

### 🎯 Core Innovation
Instead of predicting raw closing prices (high entropy),  
the model predicts:

**MA_Diff(t+1) = MA(t+1) − MA(t)**

This reduces noise and improves predictive stability.

MA(3) and MA(5) are tested; the window with higher  
out-of-sample annualized return is selected.

---

### 🧠 Model Architecture
Base Models:
- Random Forest  
- XGBoost  
- LightGBM  
- AdaBoost  
- Decision Tree  

Transfer Voting:
- Cross-ETF similarity weighting  
- Dynamic Time Warping (DTW)

---

### ⚙️ Trading Logic
1. Select ETF with highest predicted expected return  
2. If all predictions negative → allocate to CASH (3M T-Bill)  
3. Apply trailing stop loss  
4. Re-enter via Z-score threshold  
5. Apply transaction cost friction  

---

### 🔄 Automation
- Daily incremental data update  
- Weekly retraining (GitHub Actions)  
- HuggingFace dataset storage  
- CI/CD pipeline  
""")
