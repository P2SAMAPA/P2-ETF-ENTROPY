"""
ETF Transfer Voting Engine — Streamlit App
Per-year model loading: each start year has its own trained model in HF.
If no model exists for selected year today, triggers GitHub Actions retraining.
"""

import os
import json
import time
import datetime
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download, list_repo_files

from data_loader import load_dataset, load_metadata
from update_data import main as incremental_update
from feature_engineering import prepare_all_features, get_oos_dates, TARGET_ETFS
from strategy_engine import StrategyEngine
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from metrics import calculate_metrics
from utils import get_hero_next_date, get_oos_index

st.set_page_config(page_title="ETF Transfer Voting Engine", page_icon="📈", layout="wide")

HF_REPO      = "P2SAMAPA/etf-entropy-dataset"
GITHUB_REPO  = "P2SAMAPA/P2-ETF-ENTROPY"
ETF_COLORS   = {
    "TLT": "#1f77b4", "VNQ": "#ff7f0e", "GLD": "#ffd700",
    "SLV": "#aec7e8", "VCIT": "#2ca02c", "HYG": "#d62728",
    "LQD": "#9467bd", "CASH": "#7f7f7f",
}


# ── GitHub Actions trigger ────────────────────────────────────────────────────

def trigger_github_training(start_year: int) -> bool:
    """Trigger GitHub Actions workflow for the given start year."""
    token = st.secrets.get("GH_PAT", "")
    if not token:
        st.error("GH_PAT not found in Streamlit secrets. Cannot trigger retraining.")
        return False
    url  = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/train.yml/dispatches"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept":        "application/vnd.github+json",
        },
        json={
            "ref": "main",
            "inputs": {
                "run_training": "true",
                "start_year":   str(start_year),
            },
        },
        timeout=15,
    )
    return resp.status_code == 204


# ── HF model availability check ───────────────────────────────────────────────

@st.cache_data(ttl=120, show_spinner=False)
def check_model_available(start_year: int) -> bool:
    """Returns True if models/year_XXXX/best_model.json exists in HF dataset."""
    try:
        hf_hub_download(
            repo_id    = HF_REPO,
            filename   = f"models/year_{start_year}/best_model.json",
            repo_type  = "dataset",
            local_dir  = f"artifacts/year_{start_year}",
        )
        return True
    except Exception:
        return False


@st.cache_data(ttl=60, show_spinner=False)
def list_trained_years() -> list:
    """Return list of years that have a trained model in HF."""
    try:
        files   = list(list_repo_files(repo_id=HF_REPO, repo_type="dataset"))
        years   = set()
        for f in files:
            # e.g. models/year_2012/best_model.json
            parts = f.split("/")
            if len(parts) >= 2 and parts[0] == "models" and parts[1].startswith("year_"):
                try:
                    years.add(int(parts[1].replace("year_", "")))
                except ValueError:
                    pass
        return sorted(years)
    except Exception:
        return []


# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model...")
def _load_model_for_year(start_year: int):
    local_dir  = f"artifacts/year_{start_year}"
    os.makedirs(local_dir, exist_ok=True)
    meta_path  = hf_hub_download(
        repo_id   = HF_REPO,
        filename  = f"models/year_{start_year}/best_model.json",
        repo_type = "dataset",
        local_dir = local_dir,
    )
    with open(meta_path) as f:
        model_info = json.load(f)
    best_ma    = model_info["best_ma_window"]
    model_path = hf_hub_download(
        repo_id   = HF_REPO,
        filename  = f"models/year_{start_year}/transfer_voting_MA{best_ma}.pkl",
        repo_type = "dataset",
        local_dir = local_dir,
    )
    model = TransferVotingModel(TARGET_ETFS, best_ma, local_dir)
    model.load(model_path)
    return model, model_info


# ── Dataset loader ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def _load_raw():
    return load_dataset()


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_for_year(df_raw, model, model_info, year_start, tsl_pct, tx_cost, z_threshold):
    timings = {}
    best_ma = model_info["best_ma_window"]

    t = time.time()
    data_dict = prepare_all_features(df_raw, ma_window=best_ma, year_start=year_start)
    timings["Features"] = round(time.time() - t, 1)

    oos_start, oos_end = get_oos_dates(data_dict)
    if not data_dict.get("features"):
        return None, None, None, timings, oos_start, oos_end, data_dict, {}

    price_df     = df_raw[[e for e in TARGET_ETFS if e in df_raw.columns]]
    tbill_series = df_raw["3MTBILL"]

    t = time.time()
    X_dict = data_dict["features"]
    try:
        pred_raw = model.predict_all_etfs(X_dict)
    except ValueError as e:
        if "feature names" in str(e).lower() or "feature_names" in str(e).lower():
            raise RuntimeError("MODEL_STALE") from e
        raise
    predictions = {
        etf: pd.Series(preds, index=X_dict[etf].index)
        for etf, preds in pred_raw.items() if etf in X_dict
    }
    timings["Predictions"] = round(time.time() - t, 1)

    if not predictions:
        return None, None, None, timings, oos_start, oos_end, data_dict, {}

    t = time.time()
    common_dates  = sorted(set.intersection(*[set(s.index) for s in predictions.values()]))
    price_aligned = price_df.loc[price_df.index.isin(common_dates)]
    tbill_aligned = tbill_series.loc[tbill_series.index.isin(common_dates)]

    engine  = StrategyEngine(TARGET_ETFS, tsl_pct=tsl_pct,
                              transaction_cost_bps=tx_cost, z_score_threshold=z_threshold)
    results = run_backtest(predictions, price_aligned, tbill_aligned, engine, common_dates)
    timings["Backtest"] = round(time.time() - t, 1)

    equity = results["equity_curve"]
    if oos_start and oos_end:
        equity_oos, returns_oos, rf_oos = get_oos_index(
            equity, results["returns"], results["risk_free"], oos_start, oos_end)
    else:
        equity_oos, returns_oos, rf_oos = equity, results["returns"], results["risk_free"]

    if equity_oos.empty:
        return None, None, None, timings, oos_start, oos_end, data_dict, predictions

    equity_series = (equity_oos["strategy"]
                     if isinstance(equity_oos, pd.DataFrame) else equity_oos)

    raw_metrics = calculate_metrics(equity_series, returns_oos, rf_oos, results["audit_trail"])

    metrics = {
        "ann_return":  raw_metrics.get("annualized_return",  raw_metrics.get("ann_return",  0)),
        "sharpe":      raw_metrics.get("sharpe_ratio",       raw_metrics.get("sharpe",       0)),
        "max_dd":      raw_metrics.get("max_drawdown",       raw_metrics.get("max_dd",       0)),
        "max_dd_date": raw_metrics.get("max_drawdown_date",  raw_metrics.get("max_dd_date",  None)),
        "max_dd_peak": raw_metrics.get("max_drawdown_peak_date", None),
        "win_rate":    raw_metrics.get("win_rate",           0),
        "volatility":  raw_metrics.get("volatility",        0),
        "calmar":      raw_metrics.get("calmar_ratio",       0),
        "sortino":     raw_metrics.get("sortino_ratio",      0),
        "hit_ratio":   raw_metrics.get("hit_ratio_15d",      None),
    }

    audit = results.get("audit_trail", pd.DataFrame())
    if not audit.empty and oos_start:
        oos_audit = audit.loc[audit.index >= oos_start]
        if "switch_flag" in oos_audit.columns:
            metrics["n_switches"]    = int(oos_audit["switch_flag"].sum())
            n_days                   = len(oos_audit)
            metrics["avg_hold_days"] = round(n_days / max(metrics["n_switches"], 1), 1)
        else:
            etf_series = oos_audit.get("selected_etf", pd.Series(dtype=str))
            switches   = int((etf_series != etf_series.shift(1)).sum()) - 1
            metrics["n_switches"]    = max(switches, 0)
            metrics["avg_hold_days"] = round(len(oos_audit) / max(switches, 1), 1)

    timings["Total"] = round(sum(timings.values()), 1)
    return results, metrics, equity_oos, timings, oos_start, oos_end, data_dict, predictions


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

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
year_start  = st.sidebar.slider("Start Year",             2008, 2022, 2015)
tsl_pct     = st.sidebar.slider("Trailing Stop Loss (%)",   10,   25,   12)
tx_cost     = st.sidebar.slider("Transaction Cost (bps)",    10,   25,   12)
z_threshold = st.sidebar.slider("Z-Score Re-entry",        0.50, 2.00, 0.70, step=0.05)

# Show which years already have trained models
trained_years = list_trained_years()
if trained_years:
    st.sidebar.markdown("---")
    st.sidebar.caption(f"✅ Trained years: {', '.join(str(y) for y in trained_years)}")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL AVAILABILITY CHECK — trigger Actions if needed
# ══════════════════════════════════════════════════════════════════════════════

df_raw = _load_raw()

model_ready = check_model_available(year_start)

if not model_ready:
    st.title("📈 ETF Transfer Voting Engine")
    st.warning(
        f"⏳ **No trained model found for start year {year_start}.**\n\n"
        f"Click below to trigger a ~25 minute training run. "
        f"This page will auto-refresh every 30 seconds until the model is ready."
    )

    col_btn, col_status = st.columns([1, 3])
    with col_btn:
        if st.button(f"🚀 Train model for {year_start}", type="primary"):
            triggered = trigger_github_training(year_start)
            if triggered:
                st.success("✅ Training job triggered! Refreshing every 30s...")
                st.session_state["training_triggered"] = True
            else:
                st.error("❌ Failed to trigger training. Check GH_PAT secret.")

    with col_status:
        if st.session_state.get("training_triggered"):
            st.info("🔄 Training in progress (~25 mins). Page auto-refreshes every 30s.")

    # Show already-trained years as quick links
    if trained_years:
        st.markdown("---")
        st.subheader("📋 Available trained years — select one to view results immediately")
        cols = st.columns(len(trained_years))
        for col, yr in zip(cols, trained_years):
            col.metric(f"Year {yr}", "✅ Ready")
        st.info(f"Move the **Start Year** slider to one of: {trained_years} to see results now.")

    # Auto-refresh every 30 seconds while waiting
    st.markdown(
        """
        <script>
        setTimeout(function() { window.location.reload(); }, 30000);
        </script>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADED — run pipeline
# ══════════════════════════════════════════════════════════════════════════════

try:
    model, model_info = _load_model_for_year(year_start)
    best_ma           = model_info["best_ma_window"]
except Exception as e:
    st.error(f"❌ Failed to load model for year {year_start}: {e}")
    st.stop()

try:
    with st.spinner(f"Computing — year_start={year_start}, MA({best_ma})..."):
        (results, metrics, equity_oos,
         timings, oos_start, oos_end,
         data_dict, predictions) = run_for_year(
            df_raw, model, model_info, year_start, tsl_pct, tx_cost, z_threshold)
except RuntimeError as e:
    if "MODEL_STALE" in str(e):
        st.error(
            "⚠️ **Model is stale** — feature set mismatch.\n\n"
            "Trigger a retrain via GitHub Actions for this year."
        )
        st.stop()
    raise

if results is None:
    st.error("Not enough data for the selected start year. Try an earlier year.")
    st.stop()

oos_label = (f"{oos_start.date()} → {oos_end.date()}" if oos_start and oos_end else "N/A")

# ── PAGE TITLE ────────────────────────────────────────────────────────────────
st.title("📈 ETF Transfer Voting Engine")
run_date_display = model_info.get("run_date", model_info.get("last_trained", "N/A"))[:10]
st.caption(
    f"Transfer Voting · 7 ETFs · MA({best_ma}) · "
    f"Trained from {year_start} · Run: {run_date_display} · OOS: {oos_label}"
)

# ── TIMING ────────────────────────────────────────────────────────────────────
with st.expander("⏱ Computation timing", expanded=False):
    cols = st.columns(len(timings))
    for col, (lbl, sec) in zip(cols, timings.items()):
        col.metric(lbl, f"{sec}s")

# ── SPLIT DETAILS ─────────────────────────────────────────────────────────────
with st.expander("📅 80/10/10 Split Details", expanded=False):
    ref = next((e for e in TARGET_ETFS if e in data_dict.get("split_dates", {})), None)
    if ref:
        sd = data_dict["split_dates"][ref]
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Train (80%)**\n\n{sd['train_start']} → {sd['train_end']}\n\n{sd['n_train']} days")
        c2.warning(f"**Val (10%)**\n\n{sd['val_start']} → {sd['val_end']}\n\n{sd['n_val']} days")
        c3.success(f"**OOS (10%)**\n\n{sd['oos_start']} → {sd['oos_end']}\n\n{sd['n_test']} days")
        st.caption("Metrics, equity curve and trade log all use the OOS window.")

# ── NEXT ALLOCATION ───────────────────────────────────────────────────────────
st.markdown("---")
col_sig, col_exp = st.columns([1, 2])

with col_sig:
    st.subheader("📡 Next Allocation")
    X_latest = {etf: data_dict["features"][etf].iloc[-1:]
                for etf in TARGET_ETFS if etf in data_dict.get("features", {})}
    exp_returns = {}
    for etf in TARGET_ETFS:
        if etf not in X_latest:
            continue
        try:
            pred  = model.predict_single_etf(X_latest[etf], etf, source_feature_dict=X_latest)[0]
            price = float(df_raw[etf].iloc[-1])
            exp_returns[etf] = pred / price * 100.0 if price > 0 else 0.0
        except Exception:
            exp_returns[etf] = 0.0

    all_negative  = all(v < 0 for v in exp_returns.values()) if exp_returns else False
    predicted_etf = "CASH" if all_negative else (
        max(exp_returns, key=exp_returns.get) if exp_returns else "N/A"
    )
    next_date  = get_hero_next_date()
    etf_color  = ETF_COLORS.get(predicted_etf, "#333")

    st.markdown(
        f"<div style='padding:16px;border-radius:10px;"
        f"background:{etf_color}22;border-left:5px solid {etf_color};'>"
        f"<span style='font-size:2.2rem;font-weight:700;color:{etf_color}'>"
        f"{predicted_etf}</span><br>"
        f"<span style='color:#888;font-size:0.9rem'>{next_date}</span>"
        f"</div>", unsafe_allow_html=True)
    st.caption(f"Model trained from {year_start} · MA({best_ma})")

with col_exp:
    st.subheader("Expected Return — All ETFs")
    if exp_returns:
        exp_df = (pd.DataFrame.from_dict(exp_returns, orient="index", columns=["er"])
                  .sort_values("er", ascending=True))
        fig_bar = go.Figure(go.Bar(
            x=exp_df["er"], y=exp_df.index, orientation="h",
            marker_color=[ETF_COLORS.get(e, "#888") for e in exp_df.index],
            text=[f"{v:.4f}%" for v in exp_df["er"]], textposition="outside",
        ))
        fig_bar.update_layout(xaxis_title="Expected MA Diff / Price (%)",
                               height=260, margin=dict(l=0, r=80, t=10, b=30),
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_bar, use_container_width=True)

# ── OOS METRICS ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📊 OOS Performance  ({oos_label})")

dd_val      = metrics.get("max_dd", 0)
dd_date     = metrics.get("max_dd_date", None)
dd_peak     = metrics.get("max_dd_peak", None)
dd_date_str = str(pd.Timestamp(dd_date).date()) if dd_date is not None else "N/A"
dd_peak_str = str(pd.Timestamp(dd_peak).date()) if dd_peak is not None else "N/A"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ann. Return",  f"{metrics.get('ann_return', 0)*100:.2f}%")
m2.metric("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}")
m3.metric("Max Drawdown", f"{dd_val*100:.2f}%",
          delta=f"peak {dd_peak_str} → trough {dd_date_str}", delta_color="off")
m4.metric("Calmar Ratio", f"{metrics.get('calmar', 0):.2f}")

d1, d2, d3, d4 = st.columns(4)
d1.metric("Win Rate",    f"{metrics.get('win_rate', 0)*100:.1f}%")
d2.metric("Volatility",  f"{metrics.get('volatility', 0)*100:.2f}%")
d3.metric("Sortino",     f"{metrics.get('sortino', 0):.3f}")
hit = metrics.get("hit_ratio", None)
d4.metric("Hit Ratio (15d)", f"{hit*100:.1f}%" if hit is not None else "N/A")

# ── EQUITY CURVE ──────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📉 Equity Curve (OOS Only)")

equity_series = (equity_oos["strategy"]
                 if isinstance(equity_oos, pd.DataFrame) and "strategy" in equity_oos.columns
                 else equity_oos)

fig_eq = go.Figure()
fig_eq.add_trace(go.Scatter(x=equity_series.index, y=equity_series.values,
                             name="Strategy", line=dict(color="#1f77b4", width=2)))
if "SPY" in df_raw.columns and oos_start and oos_end:
    spy_ret = df_raw["SPY"].pct_change().fillna(0)
    spy_oos = spy_ret.loc[(spy_ret.index >= oos_start) & (spy_ret.index <= oos_end)]
    if len(spy_oos) > 0:
        spy_eq = (1 + spy_oos).cumprod()
        spy_eq = spy_eq / spy_eq.iloc[0] * float(equity_series.iloc[0])
        fig_eq.add_trace(go.Scatter(x=spy_eq.index, y=spy_eq.values, name="SPY B&H",
                                     line=dict(color="#aaa", width=1.5, dash="dot")))
fig_eq.update_layout(height=380, xaxis_title="Date", yaxis_title="Portfolio Value",
                      legend=dict(orientation="h", y=1.02),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=30, b=0))
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
                marker_colors=[ETF_COLORS.get(e, "#888") for e in counts.index], hole=0.4))
            fig_pie.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                   paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col_tbl:
            pct = (counts / counts.sum() * 100).round(1)
            st.dataframe(pct.rename("% Days").reset_index().rename(columns={"index": "ETF"}),
                          hide_index=True, height=260)

# ── AUDIT TABLE ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🗒️ OOS Trade Log  (last 20 entries)")

if not audit.empty and oos_start:
    audit_oos = audit.loc[audit.index >= oos_start].copy()
    audit_oos.index = pd.to_datetime(audit_oos.index).normalize().date
    disp = audit_oos.tail(20).copy()

    if "actual_return" in disp.columns:
        disp["return_%"] = disp["actual_return"].apply(
            lambda x: f"{x*100:.4f}%" if pd.notna(x) else "—"
        )
    if "expected_return" in disp.columns:
        disp["exp_ret_%"] = disp["expected_return"].apply(
            lambda x: f"{x:.4f}%" if pd.notna(x) and x != 0 else "—"
        )
    if "signal_z" in disp.columns:
        disp["z_score"] = disp["signal_z"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "—"
        )
    if "switch_reason" in disp.columns:
        disp["reason"] = disp["switch_reason"].apply(
            lambda x: x if x and x != "" else "—"
        )
    if "switch_flag" in disp.columns:
        disp["switch"] = disp["switch_flag"].apply(lambda x: "✅" if x else "")
    if "in_cash" in disp.columns:
        disp["cash"] = disp["in_cash"].apply(lambda x: "CASH" if x else "")

    display_cols = [c for c in
                    ["selected_etf", "return_%", "exp_ret_%",
                     "z_score", "switch", "reason", "cash"]
                    if c in disp.columns]
    st.dataframe(disp[display_cols], use_container_width=True, height=400)
else:
    st.info("No trade data available.")

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    f"P2-ETF-ENTROPY · Transfer Voting · MA({best_ma}) · "
    f"Trained from {year_start} · Run: {run_date_display} · "
    f"OOS: {oos_label} · Entropy 2026, 28, 84"
)
