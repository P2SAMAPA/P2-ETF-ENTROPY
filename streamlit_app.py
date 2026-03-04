"""
Streamlit App - ETF Transfer Voting Strategy
Production Version (HF + Cloud Safe)
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
from feature_engineering import prepare_all_features
from strategy_engine import StrategyEngine
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="ETF Transfer Voting Engine",
    page_icon="📈",
    layout="wide"
)

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

    with open(meta_path, "r") as f:
        model_info = json.load(f)

    best_ma = model_info["best_ma_window"]

    model_filename = f"models/transfer_voting_MA{best_ma}.pkl"

    model_path = hf_hub_download(
        repo_id="P2SAMAPA/etf-entropy-dataset",
        filename=model_filename,
        repo_type="dataset",
        local_dir="artifacts"
    )

    model = TransferVotingModel([], best_ma, "artifacts")
    model.load(model_path)

    return model, best_ma


@st.cache_data
def load_data_cached():
    return load_dataset()


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("⚙️ Strategy Controls")

year_start = st.sidebar.slider("Start Year", 2008, 2025, 2008)

tsl_pct = st.sidebar.slider("Trailing Stop Loss (%)", 10, 25, 15) / 100
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)
z_threshold = st.sidebar.slider("Z-Score Re-entry", 0.5, 2.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.write("Data Source: HuggingFace Dataset")
st.sidebar.write("Model: Transfer Voting")


# ---------------------------------------------------
# LOAD DATA + MODEL
# ---------------------------------------------------

df = load_data_cached()

if df is None:
    st.error("Dataset not found.")
    st.stop()

df = df[df.index.year >= year_start]

try:
    model, best_ma = load_model_from_hf()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()


# ---------------------------------------------------
# PREP FEATURES
# ---------------------------------------------------

etf_list = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]

data_dict = prepare_all_features(df, best_ma)
features = data_dict["features"]

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


# ---------------------------------------------------
# RUN BACKTEST
# ---------------------------------------------------

engine = StrategyEngine(
    etf_list,
    tsl_pct=tsl_pct,
    transaction_cost_bps=tx_cost,
    z_score_threshold=z_threshold
)

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


# ---------------------------------------------------
# TABS
# ---------------------------------------------------

tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📋 Audit Trail", "📖 Methodology"])


# ---------------------------------------------------
# DASHBOARD TAB
# ---------------------------------------------------

with tab1:

    st.subheader("Next Trading Day Prediction")

    latest_date = df.index[-1]
    next_day = get_next_trading_day(latest_date)

    colA, colB = st.columns(2)

    with colA:
        st.write(f"**Latest Data:** {latest_date.date()}")
        if next_day:
            st.write(f"**Next Trading Day:** {next_day.date()}")

    with colB:
        st.write(f"**Active MA Window:** MA({best_ma})")
        st.write(f"**ETFs Tracked:** {len(etf_list)}")

    st.markdown("---")
    st.subheader("Performance Metrics (OOS)")

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
    col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    col4.metric("Worst Daily Return", f"{metrics['worst_daily_return']:.2%}")
    col5.metric("Hit Ratio (15d)", f"{metrics['hit_ratio_15d']:.2%}")

    st.markdown("---")
    st.subheader("Equity Curve")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(results["equity_curve"]["strategy"], label="Strategy", linewidth=2)
    ax.plot(results["equity_curve"]["SPY"], label="SPY", alpha=0.7)
    ax.plot(results["equity_curve"]["AGG"], label="AGG", alpha=0.7)
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)


# ---------------------------------------------------
# AUDIT TAB
# ---------------------------------------------------

with tab2:

    st.subheader("Last 15 Trading Days")

    audit = results["audit_trail"].tail(15).copy()
    audit["actual_return"] = audit["actual_return"].astype(float)

    styled = audit[["date", "predicted_etf", "actual_return"]].style.applymap(
        color_returns, subset=["actual_return"]
    )

    st.dataframe(styled, use_container_width=True)


# ---------------------------------------------------
# METHODOLOGY TAB
# ---------------------------------------------------

with tab3:

    st.subheader("Strategy Methodology")

    st.markdown("""
### 📄 Research Foundation
**Flexible Target Prediction for Quantitative Trading in the American Stock Market**  
Journal: *Entropy (2026)*  

---

### 🎯 Target Engineering
Instead of predicting raw prices (high entropy), we predict:

MA_Diff(t+1) = MA(t+1) − MA(t)

This reduces noise and improves signal stability.

---

### 🧠 Model Architecture
- RandomForest  
- XGBoost  
- LightGBM  
- AdaBoost  
- Decision Tree  

Simple average voting → Transfer Voting weighting using similarity.

---

### ⚙️ Trading Logic
1. Select ETF with highest expected return  
2. If all negative → allocate to CASH (3M T-Bill)  
3. Apply Trailing Stop Loss  
4. Re-entry via Z-score threshold  
5. Apply transaction cost friction  

---

### 🔄 Automation
- Daily update: 00:30 UTC  
- Weekly retraining  
- HuggingFace dataset storage  
- GitHub Actions CI/CD  
""")
