"""
ETF Transfer Voting Engine - Production Version
"""

import streamlit as st
import pandas as pd
import os
import json
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

st.set_page_config(layout="wide", page_icon="📈")
st.title("📈 ETF Transfer Voting Engine")


# --------------------------------------------------
# DATA REFRESH SECTION
# --------------------------------------------------

def check_data_freshness():
    metadata = load_metadata()
    if metadata is None:
        return False, "No dataset found", None

    last_update = pd.to_datetime(metadata["last_data_update"])
    latest_trading = pd.to_datetime(get_latest_trading_day())

    if last_update >= latest_trading:
        return True, f"Dataset updated till {last_update.date()}", last_update.date()
    else:
        return False, f"Dataset stale: {last_update.date()}", last_update.date()


st.sidebar.header("🔄 Data Management")

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
tsl_pct = st.sidebar.slider("Trailing Stop Loss (%)", 10, 25, 15) / 100
tx_cost = st.sidebar.slider("Transaction Cost (bps)", 10, 75, 25)
z_threshold = st.sidebar.slider("Z-Score Re-entry", 0.5, 2.0, 1.0)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = load_dataset()

if df is None:
    st.error("Dataset not available.")
    st.stop()

df = df[df.index.year >= year_start]


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


# --------------------------------------------------
# BACKTEST
# --------------------------------------------------

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


# --------------------------------------------------
# HERO PREDICTION BOX
# --------------------------------------------------

latest_date = df.index[-1]

latest_features = {etf: features[etf].iloc[-1:] for etf in etf_list}

expected_returns = {}
for etf in etf_list:
    pred = model.predict_single_etf(latest_features[etf], etf)[0]
    price = df.loc[latest_date, etf]
    expected_returns[etf] = pred / price if price > 0 else 0

predicted_etf = max(expected_returns, key=expected_returns.get)
expected_ret = expected_returns[predicted_etf]

nyse = mcal.get_calendar("NYSE")
schedule = nyse.schedule(
    start_date=latest_date,
    end_date=latest_date + pd.Timedelta(days=10)
)
future = schedule.index[schedule.index > latest_date]
next_trading_day = future[0]

st.markdown("## 📅 Next Trading Day Allocation")

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


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

st.markdown("---")
st.subheader("Performance Metrics (OOS)")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Annualized Return", f"{metrics['annualized_return']:.2%}")
col2.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
col3.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
col4.metric("Worst Daily Return", f"{metrics['worst_daily_return']:.2%}")
col5.metric("Hit Ratio (15d)", f"{metrics['hit_ratio_15d']:.2%}")

st.caption(f"Worst Daily Return Date: {metrics.get('worst_daily_date')}")


# --------------------------------------------------
# EQUITY CURVE
# --------------------------------------------------

st.markdown("---")
st.subheader("Equity Curve")

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(results["equity_curve"]["strategy"], label="Strategy", linewidth=2)
ax.plot(results["equity_curve"]["SPY"], label="SPY", alpha=0.7)
ax.plot(results["equity_curve"]["AGG"], label="AGG", alpha=0.7)
ax.legend()
ax.grid(True)

st.pyplot(fig)


# --------------------------------------------------
# AUDIT TRAIL
# --------------------------------------------------

st.markdown("---")
st.subheader("Last 15 Trading Days Audit Trail")

audit = results["audit_trail"].tail(15).copy()
st.dataframe(
    audit[["date", "predicted_etf", "actual_return"]],
    use_container_width=True
)


# --------------------------------------------------
# COLLAPSIBLE METHODOLOGY
# --------------------------------------------------

st.markdown("---")

with st.expander("📖 View Strategy Methodology", expanded=False):

    st.markdown("""
### 📄 Research Foundation

**Flexible Target Prediction for Quantitative Trading in the American Stock Market:  
A Hybrid Framework Integrating Ensemble Models, Fusion Models and Transfer Learning**  
Journal: *Entropy (2026)*

---

### 🎯 Core Innovation

Instead of predicting raw closing prices (high entropy and noisy),  
the model predicts:

**MA_Diff(t+1) = MA(t+1) − MA(t)**

This reduces variance and improves predictive stability.

MA(3) and MA(5) are tested; the window with higher out-of-sample  
annualized return is selected.

---

### 🧠 Model Architecture

**Base Models**
- Random Forest  
- XGBoost  
- LightGBM  
- AdaBoost  
- Decision Tree  

**Ensemble Layer**
- Simple average voting  

**Transfer Voting**
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

- Daily update: 00:30 UTC  
- Weekly retraining  
- HuggingFace dataset storage  
- GitHub Actions CI/CD pipeline  
""")
