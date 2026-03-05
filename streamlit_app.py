"""
P2-ETF-ENTROPY Streamlit App
Fully Stabilized Production Version
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pandas_market_calendars as mcal
from datetime import timedelta

from strategy_engine import StrategyEngine
from backtest import run_backtest

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
# ENSURE DATE ALIGNMENT (CRITICAL FIX)
# ==========================================================

# Use prediction coverage — not full price history
prediction_dates = None
for series in predictions_dict.values():
    prediction_dates = (
        series.index if prediction_dates is None
        else prediction_dates.intersection(series.index)
    )

prediction_dates = pd.DatetimeIndex(sorted(prediction_dates))

price_df = price_df.loc[prediction_dates]
tbill_df = tbill_df.loc[prediction_dates]

# ==========================================================
# SIDEBAR CONTROLS
# ==========================================================

st.sidebar.header("Strategy Controls")

tsl_slider = st.sidebar.slider(
    "Trailing Stop Loss (%)",
    10, 25, 15
)

transaction_slider = st.sidebar.slider(
    "Transaction Cost (bps)",
    10, 75, 25
)

z_threshold_slider = st.sidebar.slider(
    "Z-Score Re-entry Threshold",
    0.5, 2.0, 1.0
)

# ==========================================================
# BACKTEST RANGE (USE FULL PREDICTION WINDOW)
# ==========================================================

test_dates = prediction_dates

# ==========================================================
# RUN STRATEGY
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
# HERO BOX (FIXED DATE LOGIC)
# ==========================================================

# Use last prediction date — NOT price max
last_prediction_date = prediction_dates.max()

nyse = mcal.get_calendar("NYSE")

schedule = nyse.schedule(
    start_date=last_prediction_date + timedelta(days=1),
    end_date=last_prediction_date + timedelta(days=10)
)

next_trading_day = schedule.index[0].date()

latest_signal = audit_df.iloc[-1]["predicted_etf"]

st.markdown("## 📈 Next Trading Day Prediction")

st.markdown(
    f"""
    <div style="
        background-color:#111827;
        padding:30px;
        border-radius:12px;
        text-align:center;
        font-size:24px;
        font-weight:600;
        color:white;">
        Predicted ETF for {next_trading_day}:<br><br>
        <span style="font-size:38px; color:#22C55E;">
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
# AUDIT TRAIL (PROFESSIONAL STYLING)
# ==========================================================

st.markdown("## 🧾 Audit Trail")

audit_display = audit_df.copy()
audit_display["actual_return"] = audit_display["actual_return"] * 100

def style_returns(val):
    if pd.isna(val):
        return ""
    if val > 0:
        return "color:#00B050; font-weight:600;"
    elif val < 0:
        return "color:#C00000; font-weight:600;"
    return ""

styled = (
    audit_display.style
    .format({"actual_return": "{:.2f}%"})
    .applymap(style_returns, subset=["actual_return"])
)

st.dataframe(styled, use_container_width=True)

# ==========================================================
# COLLAPSIBLE METHODOLOGY
# ==========================================================

with st.expander("📘 Methodology", expanded=False):
    st.markdown("""
### Model
- Transfer Voting regression across ETF universe
- MA(3) vs MA(5) optimization
- Annualized OOS return selection

### Allocation Logic
- Cross-sectional ranking by expected return
- Peak-based trailing stop loss
- Z-score controlled re-entry
- Transaction cost friction

### Risk Controls
- Move to cash when all expected returns negative
- 2× transaction cost switching threshold
- Fully out-of-sample backtest

### Benchmarks
- SPY (Equity proxy)
- AGG (Bond proxy)
""")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")
st.markdown("P2-ETF-ENTROPY | Professional Quant Allocation Framework")
