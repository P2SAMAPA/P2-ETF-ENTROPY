"""
ETF Transfer Voting Engine — Streamlit App
==========================================
Clean rebuild with:
  - Proper @st.cache_data / @st.cache_resource to avoid re-running
    backtest on every slider interaction
  - Full transfer voting applied at inference (source_feature_dict passed)
  - Rich metrics dashboard: Sharpe, Ann Return, MaxDD, Win Rate, trade table
  - Next allocation signal with per-ETF expected return breakdown
  - Benchmark comparison (buy-and-hold SPY)
  - MA window indicator (from model_info)
  - Plotly charts (interactive)
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from huggingface_hub import hf_hub_download
import pytz
from datetime import datetime

from data_loader import load_dataset, load_metadata
from update_data import incremental_update
from feature_engineering import prepare_all_features, TARGET_ETFS
from strategy_engine import StrategyEngine
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics
from utils import get_next_trading_day

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Transfer Voting Engine",
    page_icon="📈",
    layout="wide",
)

HF_REPO  = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = TARGET_ETFS   # ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]

ETF_COLORS = {
    "TLT":  "#1f77b4",
    "VNQ":  "#ff7f0e",
    "GLD":  "#ffd700",
    "SLV":  "#aec7e8",
    "VCIT": "#2ca02c",
    "HYG":  "#d62728",
    "LQD":  "#9467bd",
    "CASH": "#7f7f7f",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://huggingface.co/front/assets/huggingface_logo.svg",
    width=40
)
st.sidebar.title("ETF Entropy Engine")

# Data management
st.sidebar.header("🔄 Data Management")
metadata = load_metadata()
if metadata and metadata.get("last_data_update"):
    st.sidebar.info(f"Data updated: **{metadata['last_data_update']}**")
    if metadata.get("last_training_date"):
        st.sidebar.caption(f"Model trained: {metadata['last_training_date']}")
else:
    st.sidebar.warning("Metadata not found")

if st.sidebar.button("🔄 Refresh Dataset"):
    with st.spinner("Updating dataset..."):
        incremental_update()
        st.cache_data.clear()
    st.sidebar.success("Dataset refreshed ✅")
    st.rerun()

# Strategy controls
st.sidebar.header("⚙️ Strategy Controls")
year_start   = st.sidebar.slider("Start Year",           2008, 2024, 2012)
tsl_pct      = st.sidebar.slider("Trailing Stop Loss (%)", 10,   25,   15)
tx_cost      = st.sidebar.slider("Transaction Cost (bps)", 10,   75,   25)
z_threshold  = st.sidebar.slider("Z-Score Re-entry",     0.50, 2.00, 0.71, step=0.01)


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def _load_data():
    return load_dataset()


@st.cache_resource(show_spinner="Loading model...")
def _load_model():
    os.makedirs("artifacts", exist_ok=True)

    meta_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="models/best_model.json",
        repo_type="dataset",
        local_dir="artifacts",
    )
    with open(meta_path) as f:
        model_info = json.load(f)

    best_ma = model_info["best_ma_window"]

    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"models/transfer_voting_MA{best_ma}.pkl",
        repo_type="dataset",
        local_dir="artifacts",
    )
    model = TransferVotingModel(ETF_LIST, best_ma, "artifacts")
    model.load(model_path)
    return model, model_info


@st.cache_data(
    show_spinner="Preparing features...",
    hash_funcs={pd.DataFrame: lambda df: df.shape},
)
def _prepare_features(df_shape_key, ma_window, year_start):
    # Re-load inside cached fn to avoid hashing large DataFrame
    df = _load_data()
    df = df[df.index.year >= year_start]
    return prepare_all_features(df, ma_window=ma_window)


@st.cache_data(
    show_spinner="Running backtest...",
    hash_funcs={pd.DataFrame: lambda df: df.shape},
)
def _run_full_backtest(df_shape_key, ma_window, year_start, tsl_pct, tx_cost, z_threshold):
    df       = _load_data()
    df       = df[df.index.year >= year_start]
    model, _ = _load_model()

    data_dict  = prepare_all_features(df, ma_window=ma_window)
    price_df   = df[ETF_LIST]
    tbill_s    = df["3MTBILL"]

    # Predict using full transfer voting on ALL dates (not just test)
    X_dict = data_dict["features"]
    pred_raw = model.predict_all_etfs(X_dict)

    predictions = {
        etf: pd.Series(preds, index=X_dict[etf].index)
        for etf, preds in pred_raw.items()
    }

    common_dates = sorted(
        set.intersection(*[set(s.index) for s in predictions.values()])
    )
    price_aligned = price_df.loc[price_df.index.isin(common_dates)]
    tbill_aligned = tbill_s.loc[tbill_s.index.isin(common_dates)]

    engine = StrategyEngine(
        ETF_LIST,
        tsl_pct=tsl_pct,
        transaction_cost_bps=tx_cost,
        z_score_threshold=z_threshold,
    )

    results = run_backtest(predictions, price_aligned, tbill_aligned, engine, common_dates)
    return results, predictions, data_dict


# ── Load everything ───────────────────────────────────────────────────────────

df_raw            = _load_data()
model, model_info = _load_model()
best_ma           = model_info["best_ma_window"]

# Use (shape, ma, year_start) as cache key to avoid hashing full DataFrame
df_key = (df_raw.shape, best_ma, year_start)

with st.spinner("Computing..."):
    results, predictions, data_dict = _run_full_backtest(
        df_key, best_ma, year_start, tsl_pct, tx_cost, z_threshold
    )

# OOS window
df_filtered = df_raw[df_raw.index.year >= year_start]

if "oos_start_date" in model_info and "oos_end_date" in model_info:
    oos_start = pd.to_datetime(model_info["oos_start_date"])
    oos_end   = pd.to_datetime(model_info["oos_end_date"])
else:
    n         = len(df_filtered)
    oos_start = df_filtered.index[int(n * 0.9)]
    oos_end   = df_filtered.index[-1]

equity_curve = results["equity_curve"]
oos_mask     = (equity_curve.index >= oos_start) & (equity_curve.index <= oos_end)
equity_oos   = equity_curve.loc[oos_mask]
returns_oos  = results["returns"].loc[oos_mask]
risk_free_oos = results["risk_free"].loc[oos_mask]

metrics = calculate_metrics(
    equity_oos["strategy"],
    returns_oos,
    risk_free_oos,
    results["audit_trail"],
)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📈 ETF Transfer Voting Engine")
st.caption(
    f"Temporal Fusion Transfer Learning · 7 Fixed-Income ETFs · "
    f"MA({best_ma}) target · Transfer Voting (DTW-weighted)"
)

# ── NEXT ALLOCATION ───────────────────────────────────────────────────────────
st.markdown("---")
col_sig, col_exp = st.columns([1, 2])

with col_sig:
    st.subheader("📡 Next Allocation")

    # Compute expected returns for the latest date using FULL transfer voting
    data_dict_latest = prepare_all_features(
        df_raw[df_raw.index.year >= year_start], ma_window=best_ma
    )
    X_latest_dict = {
        etf: data_dict_latest["features"][etf].iloc[-1:]
        for etf in ETF_LIST
        if etf in data_dict_latest["features"]
    }

    exp_returns = {}
    for etf in ETF_LIST:
        if etf not in X_latest_dict:
            continue
        try:
            # Full transfer voting prediction for latest bar
            pred = model.predict_single_etf(
                X_latest_dict[etf], etf, source_feature_dict=X_latest_dict
            )[0]
            price = float(df_raw.iloc[-1][etf])
            exp_returns[etf] = pred / price if price > 0 else 0.0
        except Exception:
            exp_returns[etf] = 0.0

    predicted_etf = max(exp_returns, key=exp_returns.get) if exp_returns else "N/A"
    next_date     = get_next_trading_day(df_raw.index[-1])
    etf_color     = ETF_COLORS.get(predicted_etf, "#333333")

    st.markdown(
        f"<div style='padding:16px; border-radius:10px; "
        f"background:{etf_color}22; border-left:5px solid {etf_color};'>"
        f"<span style='font-size:2rem; font-weight:700; color:{etf_color};'>"
        f"{predicted_etf}</span><br>"
        f"<span style='color:#888;'>{next_date}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.caption(f"Model: Transfer Voting · MA({best_ma}) · "
               f"Trained through {model_info.get('training_end_date', 'N/A')}")

with col_exp:
    st.subheader("📊 Expected Returns (all ETFs)")
    if exp_returns:
        exp_df = (
            pd.DataFrame.from_dict(exp_returns, orient="index", columns=["Expected Return"])
            .sort_values("Expected Return", ascending=True)
        )
        bar_colors = [ETF_COLORS.get(e, "#888") for e in exp_df.index]
        fig_bar = go.Figure(go.Bar(
            x=exp_df["Expected Return"] * 100,
            y=exp_df.index,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{v*100:.4f}%" for v in exp_df["Expected Return"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_title="Expected MA Diff / Price (%)",
            height=260, margin=dict(l=0, r=20, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── METRICS ROW ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 OOS Performance Metrics")

m1, m2, m3, m4, m5, m6 = st.columns(6)

def _fmt_pct(v, decimals=1):
    if v is None:
        return "—"
    return f"{v*100:.{decimals}f}%"

def _fmt_float(v, decimals=2):
    if v is None:
        return "—"
    return f"{v:.{decimals}f}"

m1.metric("Ann. Return",  _fmt_pct(metrics.get("ann_return")))
m2.metric("Sharpe Ratio", _fmt_float(metrics.get("sharpe")))
m3.metric("Max Drawdown", _fmt_pct(metrics.get("max_dd")))
m4.metric("Win Rate",     _fmt_pct(metrics.get("win_rate")))
m5.metric("# Trades",     str(metrics.get("n_trades", "—")))
m6.metric("MA Window",    f"MA({best_ma})")

# MA window comparison
if "ma3_ann_return" in model_info and "ma5_ann_return" in model_info:
    st.caption(
        f"MA(3): {model_info['ma3_ann_return']*100:.2f}% ann return  |  "
        f"MA(5): {model_info['ma5_ann_return']*100:.2f}% ann return  →  "
        f"**MA({best_ma}) selected**"
    )

# ── EQUITY CURVE ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Equity Curve (OOS Only)")

# Build buy-and-hold SPY for comparison
spy_available = "SPY" in df_raw.columns
if spy_available:
    spy_oos = df_raw["SPY"].loc[oos_mask] if oos_mask.any() else pd.Series(dtype=float)
    spy_oos_ret = spy_oos.pct_change().fillna(0)
    spy_equity  = (1 + spy_oos_ret).cumprod()
    # Normalise to same start as strategy
    if not equity_oos.empty and len(spy_equity) > 0:
        strat_start = float(equity_oos["strategy"].iloc[0])
        spy_equity  = spy_equity / spy_equity.iloc[0] * strat_start

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=equity_oos.index,
    y=equity_oos["strategy"],
    mode="lines",
    name="Strategy",
    line=dict(color="#1f77b4", width=2),
))
if spy_available and not spy_equity.empty:
    fig_eq.add_trace(go.Scatter(
        x=spy_equity.index,
        y=spy_equity.values,
        mode="lines",
        name="SPY B&H",
        line=dict(color="#aaaaaa", width=1.5, dash="dot"),
    ))
fig_eq.update_layout(
    height=380,
    xaxis_title="Date",
    yaxis_title="Portfolio Value",
    legend=dict(orientation="h", y=1.02),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=30, b=0),
)
fig_eq.update_xaxes(showgrid=True, gridcolor="#eeeeee")
fig_eq.update_yaxes(showgrid=True, gridcolor="#eeeeee")
st.plotly_chart(fig_eq, use_container_width=True)

# ── ALLOCATION HISTORY ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Allocation Over Time (OOS)")

audit = results.get("audit_trail", pd.DataFrame())
if not audit.empty and oos_start is not None:
    audit_oos = audit.loc[audit.index >= oos_start] if not audit.empty else audit
    if not audit_oos.empty and "selected_etf" in audit_oos.columns:
        alloc_counts = audit_oos["selected_etf"].value_counts()
        fig_pie = go.Figure(go.Pie(
            labels=alloc_counts.index,
            values=alloc_counts.values,
            marker_colors=[ETF_COLORS.get(e, "#888") for e in alloc_counts.index],
            hole=0.4,
        ))
        fig_pie.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        col_pie, col_tbl = st.columns([1, 2])
        with col_pie:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_tbl:
            pct = (alloc_counts / alloc_counts.sum() * 100).round(1)
            st.dataframe(
                pct.rename("% Days Held").reset_index().rename(columns={"index": "ETF"}),
                hide_index=True, height=260,
            )

# ── TRADE LOG ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗒️ Last 20 OOS Trades")

if not audit.empty:
    audit_display = audit.loc[audit.index >= oos_start] if oos_start else audit
    switches = audit_display[audit_display.get("switch_flag", False) == True] if "switch_flag" in audit_display.columns else audit_display
    if switches.empty:
        switches = audit_display.tail(20)
    st.dataframe(
        switches.tail(20)[[
            c for c in ["selected_etf", "expected_return", "signal_z",
                        "switch_reason", "in_cash"]
            if c in switches.columns
        ]].style.format({
            "expected_return": "{:.5f}",
            "signal_z":        lambda v: f"{v:.2f}" if v is not None else "—",
        }),
        use_container_width=True,
        height=400,
    )
else:
    st.info("No trade data available.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "P2-ETF-ENTROPY · Transfer Voting Ensemble · "
    "Based on: *Flexible Target Prediction for Quantitative Trading* (Entropy 2026) · "
    f"OOS: {oos_start.date() if oos_start else '?'} → {oos_end.date() if oos_end else '?'}"
)
