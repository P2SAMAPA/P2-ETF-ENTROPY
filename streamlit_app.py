"""
P2-ETF-ENTROPY Streamlit App
Professional Production Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import pandas_market_calendars as mcal
from datetime import timedelta

from strategy_engine import StrategyEngine
from backtest import run_backtest
from optimize_ma import load_best_model

st.set_page_config(layout="wide")

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():
    price_df = pd.read_parquet("data/price_data.parquet")
    tbill_df = pd.read_parquet("data/tbill_data.parquet")["3MTBILL"]
    predictions = joblib.load("artifacts/latest_predictions.pkl")
    return price_df, tbill_df, predictions


price_df, tbill_df, predictions_dict = load_data()

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("Strategy Controls")

tsl_slider = st.sidebar.slider(
    "Trailing Stop Loss (%)",
    min_value=10,
    max_value=25,
    value=15
)

transaction_slider = st.sidebar.slider(
    "Transaction Cost (bps)",
    min_value=10,
    max_value=75,
    value=25
)

z_threshold_slider = st.sidebar.slider(
    "Z-Score Re-entry Threshold",
    min_value=0.5,
    max_value=2.0,
    value=1.0
)

# ==========================================================
# BACKTEST DATE SPLIT
# ==========================================================

n = len(price_df)
train_end = int(n * 0.8)
val_end = int(n * 0.9)

test_dates = price_df.index[val_end:]  # FULL OOS RANGE

# ==========================================================
# STRATEGY ENGINE
# ==========================================================

etf_list = list(predictions_dict.keys())

engine = StrategyEngine(
    etf_list=etf_list,
    tsl_pct=tsl_slider,
    transaction_cost_bps=transaction_slider,
    z_score_threshold=z_threshold_slider
)

results = run_backtest(
    predictions_dict,
    price_df,
    tbill_df,
    engine,
    test_dates
)

equity_curve = results["equity_curve"]
audit_df = results["audit_trail"].copy()

# ==========================================================
# HERO BOX (NEXT TRADING DAY)
# ==========================================================

latest_data_date = price_df.index.max()

nyse = mcal.get_calendar("NYSE")

schedule = nyse.schedule(
    start_date=latest_data_date + timedelta(days=1),
    end_date=latest_data_date + timedelta(days=10)
)

next_trading_day = schedule.index[0].date()

latest_signal = results["audit_trail"].iloc[-1]["predicted_etf"]

st.markdown("## 📈 Next Trading Day Prediction")

st.markdown(
    f"""
    <div style="
        background-color:#111827;
        padding:30px;
        border-radius:10px;
        text-align:center;
        font-size:24px;
        font-weight:600;
        color:white;">
        Predicted ETF for {next_trading_day}: <br><br>
        <span style="font-size:36px; color:#22C55E;">
        {latest_signal}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# EQUITY CURVE
# ==========================================================

st.markdown("## 📊 Strategy vs Benchmarks")

st.line_chart(equity_curve)

# ==========================================================
# PERFORMANCE METRICS
# ==========================================================

returns = results["returns"]

annualized_return = (1 + returns.mean()) ** 252 - 1
annualized_vol = returns.std() * np.sqrt(252)
sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

rolling_max = equity_curve["strategy"].cummax()
drawdown = equity_curve["strategy"] / rolling_max - 1
max_dd = drawdown.min()
max_dd_date = drawdown.idxmin().date()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Annual Return", f"{annualized_return*100:.2f}%")
col2.metric("Sharpe Ratio", f"{sharpe:.2f}")
col3.metric("Max Drawdown", f"{max_dd*100:.2f}%")
col4.metric("Max DD Date", f"{max_dd_date}")

# ==========================================================
# AUDIT TRAIL TABLE (PROFESSIONAL STYLING)
# ==========================================================

st.markdown("## 🧾 Audit Trail")

audit_df["actual_return"] = audit_df["actual_return"] * 100

def style_returns(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color:#00B050; font-weight:600;"
    elif val < 0:
        return "color:#C00000; font-weight:600;"
    return ""

styled_audit = (
    audit_df.style
    .format({"actual_return": "{:.2f}%"})
    .applymap(style_returns, subset=["actual_return"])
)

st.dataframe(styled_audit, use_container_width=True)

# ==========================================================
# COLLAPSIBLE METHODOLOGY
# ==========================================================

with st.expander("📘 Methodology", expanded=False):
    st.markdown("""
    ### Model
    - Transfer Voting Regression across 7 ETFs
    - MA(3) vs MA(5) window optimization
    - Selection based on lowest validation MSE
    
    ### Allocation Logic
    - Cross-sectional ranking by expected return
    - Trailing Stop Loss (peak-based)
    - Z-score re-entry filter
    - Transaction cost friction applied
    
    ### Risk Controls
    - Move to cash when all predictions negative
    - 2x transaction cost threshold for switching
    - Professional out-of-sample backtest
    
    ### Benchmarks
    - SPY (Equity proxy)
    - AGG (Bond proxy)
    """)

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("P2-ETF-ENTROPY | Professional Quant Allocation Framework")
