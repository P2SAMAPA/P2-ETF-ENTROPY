"""
ETF Transfer Voting Engine — Streamlit App (Option B)
======================================================
Year slider triggers full re-prediction + backtest in the UI.
Model is loaded from HF (pre-trained by GitHub Actions).
When user changes year_start, predictions are regenerated
on the fly using the loaded model on the new feature window.

80/10/10 split is recomputed for each year_start so OOS dates
shift accordingly and are always shown clearly.

Training time is displayed so user can see how long each run took.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

from data_loader import load_dataset, load_metadata
from update_data import incremental_update
from feature_engineering import prepare_all_features, get_oos_dates, TARGET_ETFS
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

HF_REPO   = "P2SAMAPA/etf-entropy-dataset"
ETF_COLORS = {
    "TLT": "#1f77b4", "VNQ": "#ff7f0e", "GLD": "#ffd700",
    "SLV": "#aec7e8", "VCIT": "#2ca02c", "HYG": "#d62728",
    "LQD": "#9467bd", "CASH": "#7f7f7f",
}


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 ETF Entropy Engine")

st.sidebar.header("🔄 Data Management")
metadata = load_metadata()
if metadata:
    st.sidebar.info(f"Data updated: **{metadata.get('last_data_update','?')}**")
    if metadata.get("last_training_date"):
        st.sidebar.caption(f"Last trained: {metadata['last_training_date']}")
else:
    st.sidebar.warning("Metadata not found")

if st.sidebar.button("🔄 Refresh Dataset"):
    with st.spinner("Updating dataset..."):
        incremental_update()
        st.cache_data.clear()
    st.sidebar.success("Dataset refreshed ✅")
    st.rerun()

st.sidebar.header("⚙️ Strategy Controls")
st.sidebar.caption(
    "Changing **Start Year** re-splits 80/10/10 and re-runs predictions "
    "with the loaded model on the new window."
)

year_start  = st.sidebar.slider("Start Year", 2008, 2022, 2012)
tsl_pct     = st.sidebar.slider("Trailing Stop Loss (%)",  10,  25,  15)
tx_cost     = st.sidebar.slider("Transaction Cost (bps)",  10,  75,  25)
z_threshold = st.sidebar.slider("Z-Score Re-entry", 0.50, 2.00, 1.00, step=0.05)


# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset from HuggingFace...")
def _load_raw():
    return load_dataset()


@st.cache_resource(show_spinner="Loading model from HuggingFace...")
def _load_model():
    os.makedirs("artifacts", exist_ok=True)
    meta_path = hf_hub_download(
        repo_id=HF_REPO, filename="models/best_model.json",
        repo_type="dataset", local_dir="artifacts",
    )
    with open(meta_path) as f:
        model_info = json.load(f)

    best_ma    = model_info["best_ma_window"]
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"models/transfer_voting_MA{best_ma}.pkl",
        repo_type="dataset", local_dir="artifacts",
    )
    model = TransferVotingModel(TARGET_ETFS, best_ma, "artifacts")
    model.load(model_path)
    return model, model_info


# ── Compute predictions + backtest for given year_start ───────────────────────
# NOT cached — intentionally reruns when slider changes (Option B)

def run_for_year(df_raw, model, model_info, year_start, tsl_pct, tx_cost, z_threshold):
    """
    Full pipeline for a given year_start:
      1. Recompute 80/10/10 split from year_start
      2. Regenerate predictions using loaded model
      3. Run backtest on OOS (last 10%)
      4. Return results + timing breakdown
    """
    timings = {}
    best_ma = model_info["best_ma_window"]

    # Step 1 — features + split
    t = time.time()
    data_dict = prepare_all_features(df_raw, ma_window=best_ma, year_start=year_start)
    timings["Feature engineering"] = time.time() - t

    oos_start, oos_end = get_oos_dates(data_dict)

    if not data_dict["features"]:
        return None, None, None, timings, oos_start, oos_end

    price_df     = df_raw[TARGET_ETFS]
    tbill_series = df_raw["3MTBILL"]

    # Step 2 — predictions on FULL window (for equity curve continuity)
    t = time.time()
    X_dict   = data_dict["features"]
    pred_raw = model.predict_all_etfs(X_dict)
    predictions = {
        etf: pd.Series(preds, index=X_dict[etf].index)
        for etf, preds in pred_raw.items()
    }
    timings["Predictions (transfer voting)"] = time.time() - t

    # Step 3 — backtest on full window
    t = time.time()
    common_dates = sorted(set.intersection(*[set(s.index) for s in predictions.values()]))
    price_aligned = price_df.loc[price_df.index.isin(common_dates)]
    tbill_aligned = tbill_series.loc[tbill_series.index.isin(common_dates)]

    engine = StrategyEngine(
        TARGET_ETFS,
        tsl_pct=tsl_pct,
        transaction_cost_bps=tx_cost,
        z_score_threshold=z_threshold,
    )
    results = run_backtest(predictions, price_aligned, tbill_aligned, engine, common_dates)
    timings["Backtest"] = time.time() - t

    # Step 4 — metrics on OOS only
    equity = results["equity_curve"]
    if oos_start and oos_end:
        oos_mask    = (equity.index >= oos_start) & (equity.index <= oos_end)
        equity_oos  = equity.loc[oos_mask]
        returns_oos = results["returns"].loc[oos_mask]
        rf_oos      = results["risk_free"].loc[oos_mask]
    else:
        equity_oos  = equity
        returns_oos = results["returns"]
        rf_oos      = results["risk_free"]

    metrics = calculate_metrics(
        equity_oos["strategy"], returns_oos, rf_oos, results["audit_trail"]
    )

    timings["Total"] = sum(timings.values())
    return results, metrics, equity_oos, timings, oos_start, oos_end


# ── Load base data + model ────────────────────────────────────────────────────
df_raw            = _load_raw()
model, model_info = _load_model()
best_ma           = model_info["best_ma_window"]

# ── Run pipeline ─────────────────────────────────────────────────────────────
with st.spinner(f"Running predictions for year_start={year_start}..."):
    results, metrics, equity_oos, timings, oos_start, oos_end = run_for_year(
        df_raw, model, model_info,
        year_start, tsl_pct, tx_cost, z_threshold
    )

if results is None:
    st.error("Not enough data for the selected start year. Try an earlier year.")
    st.stop()


# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("📈 ETF Transfer Voting Engine")
st.caption(
    f"Transfer Voting Ensemble · 7 ETFs · MA({best_ma}) target · "
    f"80/10/10 split from {year_start} · "
    f"OOS: {oos_start.date() if oos_start else '?'} → {oos_end.date() if oos_end else '?'}"
)


# ── TIMING EXPANDER ───────────────────────────────────────────────────────────
with st.expander("⏱ Computation timing", expanded=False):
    cols = st.columns(len(timings))
    for col, (label, secs) in zip(cols, timings.items()):
        col.metric(label, f"{secs:.1f}s")

    # Also show training info from model_info
    st.caption(
        f"GitHub Actions training time is shown in the Actions log. "
        f"Model last trained: **{model_info.get('last_trained', 'N/A')[:10]}**"
    )
    if "ma3_val_ann_return" in model_info:
        st.caption(
            f"MA selection (val set): "
            f"MA(3)={model_info['ma3_val_ann_return']*100:.2f}% ann  "
            f"MA(5)={model_info['ma5_val_ann_return']*100:.2f}% ann  "
            f"→ MA({best_ma}) selected"
        )


# ── SPLIT INFO BAR ────────────────────────────────────────────────────────────
with st.expander("📅 80/10/10 Split Details", expanded=False):
    ref = TARGET_ETFS[0]
    data_dict_disp = prepare_all_features(df_raw, ma_window=best_ma, year_start=year_start)
    sd = data_dict_disp["split_dates"].get(ref, {})
    if sd:
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Train (80%)**\n{sd['train_start']} → {sd['train_end']}\n{sd['n_train']} days")
        c2.warning(f"**Val (10%)**\n{sd['val_start']} → {sd['val_end']}\n{sd['n_val']} days")
        c3.success(f"**OOS (10%)**\n{sd['oos_start']} → {sd['oos_end']}\n{sd['n_test']} days")
    st.caption("OOS window used for all metrics and equity curve below.")


# ── NEXT ALLOCATION ───────────────────────────────────────────────────────────
st.markdown("---")
col_sig, col_exp = st.columns([1, 2])

with col_sig:
    st.subheader("📡 Next Allocation")

    # Predict latest bar with full transfer voting
    data_latest = prepare_all_features(df_raw, ma_window=best_ma, year_start=year_start)
    X_latest    = {etf: data_latest["features"][etf].iloc[-1:]
                   for etf in TARGET_ETFS if etf in data_latest["features"]}

    exp_returns = {}
    for etf in TARGET_ETFS:
        if etf not in X_latest:
            continue
        try:
            pred  = model.predict_single_etf(X_latest[etf], etf,
                                              source_feature_dict=X_latest)[0]
            price = float(df_raw.iloc[-1][etf])
            exp_returns[etf] = pred / price if price > 0 else 0.0
        except Exception:
            exp_returns[etf] = 0.0

    predicted_etf = max(exp_returns, key=exp_returns.get) if exp_returns else "N/A"
    next_date     = get_next_trading_day(df_raw.index[-1])
    etf_color     = ETF_COLORS.get(predicted_etf, "#333")

    st.markdown(
        f"<div style='padding:16px;border-radius:10px;"
        f"background:{etf_color}22;border-left:5px solid {etf_color};'>"
        f"<span style='font-size:2.2rem;font-weight:700;color:{etf_color}'>"
        f"{predicted_etf}</span><br>"
        f"<span style='color:#888;font-size:0.9rem'>{next_date}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Model: Transfer Voting · MA({best_ma})")

with col_exp:
    st.subheader("Expected Return — All ETFs")
    if exp_returns:
        exp_df = (
            pd.DataFrame.from_dict(exp_returns, orient="index", columns=["er"])
            .sort_values("er", ascending=True)
        )
        fig_bar = go.Figure(go.Bar(
            x=exp_df["er"] * 100,
            y=exp_df.index,
            orientation="h",
            marker_color=[ETF_COLORS.get(e, "#888") for e in exp_df.index],
            text=[f"{v*100:.4f}%" for v in exp_df["er"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            xaxis_title="Expected MA Diff / Price (%)",
            height=250, margin=dict(l=0, r=40, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── OOS METRICS ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📊 OOS Performance  ({oos_start.date() if oos_start else '?'} → {oos_end.date() if oos_end else '?'})")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Ann. Return",  f"{metrics.get('ann_return',0)*100:.2f}%")
m2.metric("Sharpe Ratio", f"{metrics.get('sharpe',0):.3f}")
m3.metric("Max Drawdown", f"{metrics.get('max_dd',0)*100:.2f}%")
m4.metric("Win Rate",     f"{metrics.get('win_rate',0)*100:.1f}%")
m5.metric("# Trades",     str(metrics.get("n_trades", "—")))
m6.metric("MA Window",    f"MA({best_ma})")


# ── EQUITY CURVE ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Equity Curve (OOS Only)")

spy_ret    = df_raw["SPY"].pct_change().fillna(0) if "SPY" in df_raw.columns else None
oos_mask   = (equity_oos.index >= oos_start) & (equity_oos.index <= oos_end) if oos_start else slice(None)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=equity_oos.index, y=equity_oos["strategy"],
    name="Strategy", line=dict(color="#1f77b4", width=2),
))
if spy_ret is not None and oos_start:
    spy_oos = spy_ret.loc[(spy_ret.index >= oos_start) & (spy_ret.index <= oos_end)]
    spy_eq  = (1 + spy_oos).cumprod()
    if len(spy_eq) > 0:
        spy_eq = spy_eq / spy_eq.iloc[0] * float(equity_oos["strategy"].iloc[0])
        fig_eq.add_trace(go.Scatter(
            x=spy_eq.index, y=spy_eq.values,
            name="SPY B&H", line=dict(color="#aaa", width=1.5, dash="dot"),
        ))
fig_eq.update_layout(
    height=380, xaxis_title="Date", yaxis_title="Portfolio Value",
    legend=dict(orientation="h", y=1.02),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=30, b=0),
)
fig_eq.update_xaxes(showgrid=True, gridcolor="#eee")
fig_eq.update_yaxes(showgrid=True, gridcolor="#eee")
st.plotly_chart(fig_eq, use_container_width=True)


# ── ALLOCATION PIE ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Allocation Breakdown (OOS)")

audit = results.get("audit_trail", pd.DataFrame())
if not audit.empty and oos_start:
    audit_oos = audit.loc[audit.index >= oos_start]
    if not audit_oos.empty and "selected_etf" in audit_oos.columns:
        counts = audit_oos["selected_etf"].value_counts()
        col_pie, col_tbl = st.columns([1, 2])
        with col_pie:
            fig_pie = go.Figure(go.Pie(
                labels=counts.index, values=counts.values,
                marker_colors=[ETF_COLORS.get(e, "#888") for e in counts.index],
                hole=0.4,
            ))
            fig_pie.update_layout(
                height=280, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_tbl:
            pct = (counts / counts.sum() * 100).round(1)
            st.dataframe(
                pct.rename("% Days").reset_index().rename(columns={"index": "ETF"}),
                hide_index=True, height=260,
            )


# ── TRADE LOG ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗒️ Last 20 OOS Trades")

if not audit.empty:
    audit_oos = audit.loc[audit.index >= oos_start] if oos_start else audit
    switches  = audit_oos[audit_oos.get("switch_flag", pd.Series(False, index=audit_oos.index))]
    if switches.empty:
        switches = audit_oos.tail(20)
    display_cols = [c for c in ["selected_etf", "expected_return", "signal_z",
                                 "switch_reason", "in_cash"] if c in switches.columns]
    st.dataframe(switches.tail(20)[display_cols], use_container_width=True, height=380)
else:
    st.info("No trade data available.")


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"P2-ETF-ENTROPY · Transfer Voting · MA({best_ma}) · "
    f"80/10/10 split from {year_start} · "
    f"OOS: {oos_start.date() if oos_start else '?'} → {oos_end.date() if oos_end else '?'} · "
    "Based on: Entropy 2026, 28, 84"
)
