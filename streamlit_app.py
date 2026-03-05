"""
ETF Transfer Voting Engine — Streamlit App (Option B)
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
from utils import get_hero_next_date, get_oos_index, max_drawdown

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ETF Transfer Voting Engine",
    page_icon="📈",
    layout="wide",
)

HF_REPO    = "P2SAMAPA/etf-entropy-dataset"
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
    st.sidebar.info(f"Data updated: **{metadata.get('last_data_update', '?')}**")
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
    "Changing **Start Year** re-splits 80/10/10 and re-runs "
    "predictions with the loaded model on the new window."
)
year_start  = st.sidebar.slider("Start Year",              2008, 2022, 2012)
tsl_pct     = st.sidebar.slider("Trailing Stop Loss (%)",    10,   25,   15)
tx_cost     = st.sidebar.slider("Transaction Cost (bps)",     10,   75,   25)
z_threshold = st.sidebar.slider("Z-Score Re-entry",         0.50, 2.00, 1.00, step=0.05)


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


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_for_year(df_raw, model, model_info, year_start, tsl_pct, tx_cost, z_threshold):
    """
    Returns: results, metrics_dict, equity_oos, timings,
             oos_start, oos_end, data_dict, predictions
    """
    timings = {}
    best_ma = model_info["best_ma_window"]

    # Step 1 — features + 80/10/10 split
    t = time.time()
    data_dict = prepare_all_features(df_raw, ma_window=best_ma, year_start=year_start)
    timings["Features"] = round(time.time() - t, 1)

    oos_start, oos_end = get_oos_dates(data_dict)

    if not data_dict.get("features"):
        return None, None, None, timings, oos_start, oos_end, data_dict, {}

    price_df     = df_raw[[e for e in TARGET_ETFS if e in df_raw.columns]]
    tbill_series = df_raw["3MTBILL"]

    # Step 2 — predictions (full transfer voting, all dates)
    t = time.time()
    X_dict   = data_dict["features"]
    pred_raw = model.predict_all_etfs(X_dict)
    predictions = {
        etf: pd.Series(preds, index=X_dict[etf].index)
        for etf, preds in pred_raw.items()
        if etf in X_dict
    }
    timings["Predictions"] = round(time.time() - t, 1)

    if not predictions:
        return None, None, None, timings, oos_start, oos_end, data_dict, {}

    # Step 3 — backtest on full date range
    t = time.time()
    common_dates  = sorted(set.intersection(*[set(s.index) for s in predictions.values()]))
    price_aligned = price_df.loc[price_df.index.isin(common_dates)]
    tbill_aligned = tbill_series.loc[tbill_series.index.isin(common_dates)]

    engine = StrategyEngine(
        TARGET_ETFS,
        tsl_pct=tsl_pct,
        transaction_cost_bps=tx_cost,
        z_score_threshold=z_threshold,
    )
    results = run_backtest(predictions, price_aligned, tbill_aligned, engine, common_dates)
    timings["Backtest"] = round(time.time() - t, 1)

    # Step 4 — slice to OOS only for metrics
    equity = results["equity_curve"]

    if oos_start and oos_end:
        equity_oos, returns_oos, rf_oos = get_oos_index(
            equity, results["returns"], results["risk_free"], oos_start, oos_end
        )
    else:
        equity_oos  = equity
        returns_oos = results["returns"]
        rf_oos      = results["risk_free"]

    if equity_oos.empty:
        return None, None, None, timings, oos_start, oos_end, data_dict, predictions

    # Step 5 — compute metrics from OOS equity Series (not DataFrame)
    equity_series = (
        equity_oos["strategy"]
        if isinstance(equity_oos, pd.DataFrame) and "strategy" in equity_oos.columns
        else equity_oos
    )

    metrics = calculate_metrics(
        equity_series, returns_oos, rf_oos, results["audit_trail"]
    )

    # Compute MaxDD with date separately (calculate_metrics may not return it)
    dd_val, dd_date = max_drawdown(equity_series)
    metrics["max_dd"]      = dd_val
    metrics["max_dd_date"] = dd_date

    # Count switches in OOS audit
    audit = results.get("audit_trail", pd.DataFrame())
    if not audit.empty and oos_start:
        audit_oos = audit.loc[audit.index >= oos_start]
        n_switches = int(audit_oos.get("switch_flag", pd.Series(dtype=bool)).sum()) if "switch_flag" in audit_oos.columns else 0
        # Avg hold: OOS days / number of switches
        n_oos_days = len(audit_oos)
        metrics["n_switches"]   = n_switches
        metrics["avg_hold_days"] = round(n_oos_days / max(n_switches, 1), 1)
    else:
        metrics["n_switches"]    = 0
        metrics["avg_hold_days"] = 0

    timings["Total"] = round(sum(timings.values()), 1)
    return results, metrics, equity_oos, timings, oos_start, oos_end, data_dict, predictions


# ── Load ──────────────────────────────────────────────────────────────────────
df_raw            = _load_raw()
model, model_info = _load_model()
best_ma           = model_info["best_ma_window"]

# ── Run ───────────────────────────────────────────────────────────────────────
with st.spinner(f"Computing — year_start={year_start}, MA({best_ma})..."):
    (results, metrics, equity_oos,
     timings, oos_start, oos_end,
     data_dict, predictions) = run_for_year(
        df_raw, model, model_info,
        year_start, tsl_pct, tx_cost, z_threshold
    )

if results is None:
    st.error("Not enough data for the selected start year. Try an earlier year.")
    st.stop()

oos_label = (f"{oos_start.date()} → {oos_end.date()}"
             if oos_start and oos_end else "N/A")

# ── PAGE TITLE ────────────────────────────────────────────────────────────────
st.title("📈 ETF Transfer Voting Engine")
st.caption(
    f"Transfer Voting Ensemble · 7 ETFs · MA({best_ma}) · "
    f"80/10/10 from {year_start} · OOS: {oos_label}"
)

# ── TIMING ────────────────────────────────────────────────────────────────────
with st.expander("⏱ Computation timing", expanded=False):
    t_cols = st.columns(len(timings))
    for col, (lbl, sec) in zip(t_cols, timings.items()):
        col.metric(lbl, f"{sec}s")
    trained = model_info.get("last_trained", "N/A")[:10]
    st.caption(f"Model last trained (GitHub Actions): **{trained}**")
    if "ma3_val_ann_return" in model_info:
        st.caption(
            f"MA selection (val): "
            f"MA(3) = {model_info['ma3_val_ann_return']*100:.2f}%  |  "
            f"MA(5) = {model_info['ma5_val_ann_return']*100:.2f}%  →  "
            f"MA({best_ma}) selected"
        )

# ── SPLIT DETAILS ─────────────────────────────────────────────────────────────
with st.expander("📅 80/10/10 Split Details", expanded=False):
    ref = next((e for e in TARGET_ETFS if e in data_dict.get("split_dates", {})), None)
    if ref:
        sd = data_dict["split_dates"][ref]
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Train (80%)**\n\n{sd['train_start']} → {sd['train_end']}\n\n{sd['n_train']} days")
        c2.warning(f"**Val (10%)**\n\n{sd['val_start']} → {sd['val_end']}\n\n{sd['n_val']} days")
        c3.success(f"**OOS (10%)**\n\n{sd['oos_start']} → {sd['oos_end']}\n\n{sd['n_test']} days")
        st.caption("All metrics below use the OOS window only.")

# ── NEXT ALLOCATION ───────────────────────────────────────────────────────────
st.markdown("---")
col_sig, col_exp = st.columns([1, 2])

with col_sig:
    st.subheader("📡 Next Allocation")

    # Reuse data_dict already computed — no second prepare_all_features call
    X_latest = {
        etf: data_dict["features"][etf].iloc[-1:]
        for etf in TARGET_ETFS
        if etf in data_dict.get("features", {})
    }

    exp_returns = {}
    for etf in TARGET_ETFS:
        if etf not in X_latest:
            continue
        try:
            pred  = model.predict_single_etf(
                X_latest[etf], etf, source_feature_dict=X_latest
            )[0]
            price = float(df_raw[etf].iloc[-1])
            exp_returns[etf] = pred / price if price > 0 else 0.0
        except Exception:
            exp_returns[etf] = 0.0

    predicted_etf = max(exp_returns, key=exp_returns.get) if exp_returns else "N/A"
    next_date     = get_hero_next_date()        # anchored to today
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
            height=260, margin=dict(l=0, r=80, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ── OOS METRICS ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📊 OOS Performance  ({oos_label})")

dd_val  = metrics.get("max_dd", 0)
dd_date = metrics.get("max_dd_date", None)
dd_str  = f"{dd_val*100:.2f}%"
if dd_date is not None:
    try:
        dd_str += f"\n_{pd.Timestamp(dd_date).date()}_"
    except Exception:
        pass

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Ann. Return",    f"{metrics.get('ann_return', 0)*100:.2f}%")
m2.metric("Sharpe Ratio",   f"{metrics.get('sharpe', 0):.3f}")
m3.metric("Max Drawdown",   f"{dd_val*100:.2f}%",
          delta=f"trough: {pd.Timestamp(dd_date).date()}" if dd_date is not None else None,
          delta_color="off")
m4.metric("Switches (OOS)", str(metrics.get("n_switches", "—")))
m5.metric("Avg Hold (days)",str(metrics.get("avg_hold_days", "—")))

# Win rate below metrics
wr = metrics.get("win_rate", None)
if wr is not None:
    st.caption(f"Win rate (daily positive returns in OOS): **{wr*100:.1f}%**")

# ── EQUITY CURVE ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Equity Curve (OOS Only)")

equity_series = (
    equity_oos["strategy"]
    if isinstance(equity_oos, pd.DataFrame) and "strategy" in equity_oos.columns
    else equity_oos
)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(
    x=equity_series.index, y=equity_series.values,
    name="Strategy", line=dict(color="#1f77b4", width=2),
))

if "SPY" in df_raw.columns and oos_start and oos_end:
    spy_ret = df_raw["SPY"].pct_change().fillna(0)
    spy_oos = spy_ret.loc[(spy_ret.index >= oos_start) & (spy_ret.index <= oos_end)]
    if len(spy_oos) > 0:
        spy_eq = (1 + spy_oos).cumprod()
        spy_eq = spy_eq / spy_eq.iloc[0] * float(equity_series.iloc[0])
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
    else:
        st.info("No allocation data in OOS window.")
else:
    st.info("Audit trail not available.")

# ── TRADE LOG ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗒️ Last 20 OOS Trades")

if not audit.empty and oos_start:
    audit_oos = audit.loc[audit.index >= oos_start].copy()

    # Fix timestamps — show date only
    audit_oos.index = pd.to_datetime(audit_oos.index).date

    # Join actual daily returns for each held ETF
    if "selected_etf" in audit_oos.columns:
        def _get_daily_return(row):
            etf  = row.get("selected_etf", "CASH")
            date = row.name
            if etf == "CASH" or etf not in df_raw.columns:
                return None
            try:
                ts = pd.Timestamp(date)
                if ts in df_raw.index:
                    idx = df_raw.index.get_loc(ts)
                    if idx > 0:
                        p1 = df_raw[etf].iloc[idx]
                        p0 = df_raw[etf].iloc[idx - 1]
                        return round((p1 / p0 - 1) * 100, 3) if p0 > 0 else None
            except Exception:
                pass
            return None

        audit_oos["daily_ret_%"] = audit_oos.apply(_get_daily_return, axis=1)

    # Show switches + all rows, last 20
    display_cols = [c for c in
                    ["selected_etf", "daily_ret_%", "expected_return",
                     "signal_z", "switch_reason", "in_cash"]
                    if c in audit_oos.columns]

    # Filter to switches only, fallback to all rows
    if "switch_flag" in audit_oos.columns:
        switches = audit_oos[audit_oos["switch_flag"] == True]
        if switches.empty:
            switches = audit_oos
    else:
        switches = audit_oos

    st.dataframe(
        switches.tail(20)[display_cols],
        use_container_width=True,
        height=380,
    )
else:
    st.info("No trade data available.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"P2-ETF-ENTROPY · Transfer Voting · MA({best_ma}) · "
    f"80/10/10 from {year_start} · OOS: {oos_label} · "
    "Entropy 2026, 28, 84"
)
