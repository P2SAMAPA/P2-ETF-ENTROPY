"""
ETF Transfer Voting Engine — Streamlit App
Tab 1: Single year deep-dive (per-year trained model)
Tab 2: Consensus Sweep (compare all trained years side by side)
Multi-year parallel training: select 1-5 years, trigger all at once.
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
from utils import get_hero_next_date

st.set_page_config(page_title="ETF Transfer Voting Engine", page_icon="📈", layout="wide")

HF_REPO     = "P2SAMAPA/etf-entropy-dataset"
GITHUB_REPO = "P2SAMAPA/P2-ETF-ENTROPY"
ALL_YEARS   = list(range(2008, 2023))
ETF_COLORS  = {
    "TLT": "#1f77b4", "VNQ": "#ff7f0e", "GLD": "#ffd700",
    "SLV": "#aec7e8", "VCIT": "#2ca02c", "HYG": "#d62728",
    "LQD": "#9467bd", "CASH": "#7f7f7f",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def trigger_github_training(start_year: int) -> bool:
    token = st.secrets.get("GH_PAT", "")
    if not token:
        return False
    url  = (f"https://api.github.com/repos/{GITHUB_REPO}"
            f"/actions/workflows/train.yml/dispatches")
    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json"},
        json={"ref": "main",
              "inputs": {"run_training": "true",
                         "start_year": str(start_year)}},
        timeout=15,
    )
    return resp.status_code == 204


@st.cache_data(ttl=120, show_spinner=False)
def check_model_available(start_year: int) -> bool:
    try:
        hf_hub_download(
            repo_id=HF_REPO,
            filename=f"models/year_{start_year}/best_model.json",
            repo_type="dataset",
            local_dir=f"artifacts/year_{start_year}",
        )
        return True
    except Exception:
        return False


def _today_utc() -> str:
    """Return today's date string in UTC — matches GitHub Actions run_date."""
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


@st.cache_data(ttl=60, show_spinner=False)
def list_trained_years() -> list:
    """Return sorted list of years that have a trained model in HF."""
    return sorted(_year_run_dates().keys())


@st.cache_data(ttl=60, show_spinner=False)
def _year_run_dates() -> dict:
    """
    Return {year: run_date_str} for all trained years.
    Fetches best_model.json for each year to get the run_date field.
    """
    try:
        files = list(list_repo_files(repo_id=HF_REPO, repo_type="dataset"))
        years = set()
        for f in files:
            parts = f.split("/")
            if (len(parts) >= 3 and parts[0] == "models"
                    and parts[1].startswith("year_")
                    and parts[2] == "best_model.json"):
                try:
                    years.add(int(parts[1].replace("year_", "")))
                except ValueError:
                    pass
    except Exception:
        return {}

    run_dates = {}
    for yr in sorted(years):
        try:
            local_dir = f"artifacts/year_{yr}"
            os.makedirs(local_dir, exist_ok=True)
            meta_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=f"models/year_{yr}/best_model.json",
                repo_type="dataset",
                local_dir=local_dir,
                force_download=True,
            )
            with open(meta_path) as f:
                meta = json.load(f)
            run_dates[yr] = meta.get("run_date", "unknown")
        except Exception:
            run_dates[yr] = "unknown"
    return run_dates


@st.cache_resource(show_spinner="Loading model...")
def _load_model_for_year(start_year: int):
    local_dir = f"artifacts/year_{start_year}"
    os.makedirs(local_dir, exist_ok=True)
    meta_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"models/year_{start_year}/best_model.json",
        repo_type="dataset", local_dir=local_dir,
    )
    with open(meta_path) as f:
        model_info = json.load(f)
    best_ma    = model_info["best_ma_window"]
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename=f"models/year_{start_year}/transfer_voting_MA{best_ma}.pkl",
        repo_type="dataset", local_dir=local_dir,
    )
    model = TransferVotingModel(TARGET_ETFS, best_ma, local_dir)
    model.load(model_path)
    return model, model_info


@st.cache_data(show_spinner="Loading dataset...")
def _load_raw():
    return load_dataset()


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

    # Run backtest on OOS dates only — prevents engine CASH state from
    # carrying over from val period and locking out z-score re-entry.
    if oos_start and oos_end:
        oos_dates = [d for d in common_dates
                     if pd.Timestamp(d) >= oos_start and pd.Timestamp(d) <= oos_end]
    else:
        oos_dates = common_dates

    engine  = StrategyEngine(TARGET_ETFS, tsl_pct=tsl_pct,
                              transaction_cost_bps=tx_cost, z_score_threshold=z_threshold)
    results = run_backtest(predictions, price_aligned, tbill_aligned, engine, oos_dates)
    timings["Backtest"] = round(time.time() - t, 1)

    equity     = results["equity_curve"]
    equity_oos = equity
    returns_oos = results["returns"]
    rf_oos      = results["risk_free"]

    if equity_oos is None or (hasattr(equity_oos, 'empty') and equity_oos.empty):
        return None, None, None, timings, oos_start, oos_end, data_dict, predictions

    equity_series = (equity_oos["strategy"]
                     if isinstance(equity_oos, pd.DataFrame)
                     and "strategy" in equity_oos.columns else equity_oos)

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
            etf_s    = oos_audit.get("selected_etf", pd.Series(dtype=str))
            switches = int((etf_s != etf_s.shift(1)).sum()) - 1
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
tsl_pct     = st.sidebar.slider("Trailing Stop Loss (%)",  10,  25,  12)
tx_cost     = st.sidebar.slider("Transaction Cost (bps)",  10,  25,  12)
z_threshold = st.sidebar.slider("Z-Score Re-entry",      0.50, 2.00, 0.70, step=0.05)

trained_years = list_trained_years()
year_run_dates = _year_run_dates()
today_utc      = _today_utc()

st.sidebar.markdown("---")
st.sidebar.header("📅 Year Model Status")
if trained_years:
    for yr in trained_years:
        rd = year_run_dates.get(yr, "unknown")
        tag = "✅ today" if rd == today_utc else f"🔄 {rd}"
        st.sidebar.caption(f"{yr}: {tag}")
else:
    st.sidebar.warning("No trained years yet.")

# Years available to train:
# - never trained, OR trained on a previous day (stale → eligible for refresh)
years_trained_today = [yr for yr in trained_years
                       if year_run_dates.get(yr) == today_utc]
years_needing_train = [yr for yr in ALL_YEARS
                       if yr not in years_trained_today]

st.sidebar.markdown("---")
st.sidebar.header("🚀 Train / Refresh Years")
st.sidebar.caption(
    "Untrained years and yesterday's models are available. "
    "Select 1–5 — all train in parallel (~30 mins)."
)
years_to_train = st.sidebar.multiselect(
    "Select years to train",
    options=years_needing_train,
    max_selections=5,
    format_func=lambda y: (
        f"{y}  🔄 retrain (last: {year_run_dates.get(y, '?')})"
        if y in trained_years else f"{y}  ⬜ new"
    ),
    placeholder="Choose up to 5 years...",
)
if st.sidebar.button("🚀 Trigger Training", type="primary",
                     disabled=len(years_to_train) == 0):
    ok_years, fail_years = [], []
    for yr in years_to_train:
        (ok_years if trigger_github_training(yr) else fail_years).append(yr)
    if ok_years:
        st.sidebar.success(f"✅ Triggered: {ok_years}")
        st.session_state["training_triggered_years"] = ok_years
    if fail_years:
        st.sidebar.error(f"❌ Failed: {fail_years} — check GH_PAT secret")

if st.session_state.get("training_triggered_years"):
    st.sidebar.info(
        f"🔄 Training in progress for "
        f"{st.session_state['training_triggered_years']}. "
        f"Page refreshes every 30s."
    )


# ══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ══════════════════════════════════════════════════════════════════════════════

df_raw = _load_raw()

# Auto-refresh while training
if st.session_state.get("training_triggered_years"):
    st.markdown(
        "<script>setTimeout(function(){window.location.reload();},30000);</script>",
        unsafe_allow_html=True)

if not trained_years:
    st.title("📈 ETF Transfer Voting Engine")
    st.info(
        "No trained models yet. Use the **Train New Years** section "
        "in the sidebar to get started."
    )
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["📊 Single Year Deep-Dive", "🔍 Consensus Sweep"])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1
# ─────────────────────────────────────────────────────────────────────────────

with tab1:
    view_year = st.selectbox(
        "Select trained year to view:",
        options=trained_years,
        index=len(trained_years) - 1,
        format_func=lambda y: f"{y}  ✅",
    )
    untrained_reminder = [y for y in ALL_YEARS if y not in trained_years]
    if untrained_reminder:
        st.caption(f"Years not yet trained: {untrained_reminder} — use sidebar.")

    try:
        model, model_info = _load_model_for_year(view_year)
        best_ma           = model_info["best_ma_window"]
    except Exception as e:
        st.error(f"❌ Failed to load model for {view_year}: {e}")
        st.stop()

    try:
        with st.spinner(f"Computing — year={view_year}, MA({best_ma})..."):
            (results, metrics, equity_oos,
             timings, oos_start, oos_end,
             data_dict, predictions) = run_for_year(
                df_raw, model, model_info, view_year,
                tsl_pct, tx_cost, z_threshold)
    except RuntimeError as e:
        if "MODEL_STALE" in str(e):
            st.error("⚠️ Model is stale — retrain this year via sidebar.")
            st.stop()
        raise

    if results is None:
        st.error("Not enough data for this year.")
        st.stop()

    oos_label        = (f"{oos_start.date()} → {oos_end.date()}"
                        if oos_start and oos_end else "N/A")
    run_date_display = model_info.get("run_date",
                       model_info.get("last_trained", "N/A"))[:10]

    st.title("📈 ETF Transfer Voting Engine")
    st.caption(
        f"Transfer Voting · 7 ETFs · MA({best_ma}) · "
        f"Trained from {view_year} · Run: {run_date_display} · OOS: {oos_label}"
    )

    with st.expander("⏱ Computation timing", expanded=False):
        tcols = st.columns(len(timings))
        for col, (lbl, sec) in zip(tcols, timings.items()):
            col.metric(lbl, f"{sec}s")

    with st.expander("📅 80/10/10 Split Details", expanded=False):
        ref = next((e for e in TARGET_ETFS if e in data_dict.get("split_dates", {})), None)
        if ref:
            sd = data_dict["split_dates"][ref]
            c1, c2, c3 = st.columns(3)
            c1.info(f"**Train (80%)**\n\n{sd['train_start']} → {sd['train_end']}\n\n{sd['n_train']} days")
            c2.warning(f"**Val (10%)**\n\n{sd['val_start']} → {sd['val_end']}\n\n{sd['n_val']} days")
            c3.success(f"**OOS (10%)**\n\n{sd['oos_start']} → {sd['oos_end']}\n\n{sd['n_test']} days")
            st.caption("Metrics, equity curve and trade log all use the OOS window.")

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
                pred  = model.predict_single_etf(
                    X_latest[etf], etf, source_feature_dict=X_latest)[0]
                price = float(df_raw[etf].iloc[-1])
                exp_returns[etf] = pred / price * 100.0 if price > 0 else 0.0
            except Exception:
                exp_returns[etf] = 0.0
        predicted_etf = (max(exp_returns, key=exp_returns.get)
                         if exp_returns else "N/A")
        next_date  = get_hero_next_date()
        etf_color  = ETF_COLORS.get(predicted_etf, "#333")
        st.markdown(
            f"<div style='padding:16px;border-radius:10px;"
            f"background:{etf_color}22;border-left:5px solid {etf_color};'>"
            f"<span style='font-size:2.2rem;font-weight:700;color:{etf_color}'>"
            f"{predicted_etf}</span><br>"
            f"<span style='color:#888;font-size:0.9rem'>{next_date}</span>"
            f"</div>", unsafe_allow_html=True)
        st.caption(f"Model trained from {view_year} · MA({best_ma})")

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
            fig_bar.update_layout(
                xaxis_title="Expected MA Diff / Price (%)",
                height=260, margin=dict(l=0, r=80, t=10, b=30),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_bar, use_container_width=True)

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
    d1.metric("Win Rate",   f"{metrics.get('win_rate', 0)*100:.1f}%")
    d2.metric("Volatility", f"{metrics.get('volatility', 0)*100:.2f}%")
    d3.metric("Sortino",    f"{metrics.get('sortino', 0):.3f}")
    hit = metrics.get("hit_ratio", None)
    d4.metric("Hit Ratio (15d)", f"{hit*100:.1f}%" if hit is not None else "N/A")

    st.markdown("---")
    st.subheader("📉 Equity Curve (OOS Only)")
    equity_series = (equity_oos["strategy"]
                     if isinstance(equity_oos, pd.DataFrame)
                     and "strategy" in equity_oos.columns else equity_oos)
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
                    hole=0.4))
                fig_pie.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                                       paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_pie, use_container_width=True)
            with col_tbl:
                pct = (counts / counts.sum() * 100).round(1)
                st.dataframe(
                    pct.rename("% Days").reset_index().rename(columns={"index": "ETF"}),
                    hide_index=True, height=260)

    st.markdown("---")
    st.subheader("🗒️ OOS Trade Log  (last 20 entries)")
    if not audit.empty and oos_start:
        audit_oos = audit.loc[audit.index >= oos_start].copy()
        audit_oos.index = pd.to_datetime(audit_oos.index).normalize().date
        disp = audit_oos.tail(20).copy()
        if "actual_return" in disp.columns:
            disp["return_%"] = disp["actual_return"].apply(
                lambda x: f"{x*100:.4f}%" if pd.notna(x) else "—")
        if "expected_return" in disp.columns:
            disp["exp_ret_%"] = disp["expected_return"].apply(
                lambda x: f"{x:.4f}%" if pd.notna(x) and x != 0 else "—")
        if "signal_z" in disp.columns:
            disp["z_score"] = disp["signal_z"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        if "switch_reason" in disp.columns:
            disp["reason"] = disp["switch_reason"].apply(
                lambda x: x if x and x != "" else "—")
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

    st.markdown("---")
    st.caption(
        f"P2-ETF-ENTROPY · Transfer Voting · MA({best_ma}) · "
        f"Trained from {view_year} · Run: {run_date_display} · "
        f"OOS: {oos_label} · Entropy 2026, 28, 84"
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — CONSENSUS SWEEP
# ─────────────────────────────────────────────────────────────────────────────

with tab2:
    st.title("🔍 Consensus Sweep — All Trained Years")

    if len(trained_years) < 2:
        st.info(
            "Need at least 2 trained years for consensus sweep. "
            "Use the sidebar to trigger training for more years."
        )
        st.stop()

    st.caption(
        f"Comparing {len(trained_years)} trained year-models: {trained_years}. "
        f"Strategy controls from sidebar apply to all."
    )

    sweep_rows    = []
    equity_traces = {}

    progress = st.progress(0, text="Loading sweep data...")
    for i, yr in enumerate(trained_years):
        progress.progress(i / len(trained_years), text=f"Running year {yr}...")
        try:
            m, mi = _load_model_for_year(yr)
            (res, mets, eq_oos, _,
             oos_s, oos_e, dd, _) = run_for_year(
                df_raw, m, mi, yr, tsl_pct, tx_cost, z_threshold)

            if res is None or mets is None:
                continue

            # Next allocation for this year's model
            X_lat = {etf: dd["features"][etf].iloc[-1:]
                     for etf in TARGET_ETFS if etf in dd.get("features", {})}
            er = {}
            for etf in TARGET_ETFS:
                if etf not in X_lat:
                    continue
                try:
                    pred  = m.predict_single_etf(
                        X_lat[etf], etf, source_feature_dict=X_lat)[0]
                    price = float(df_raw[etf].iloc[-1])
                    er[etf] = pred / price * 100.0 if price > 0 else 0.0
                except Exception:
                    er[etf] = 0.0
            next_pick = max(er, key=er.get) if er else "N/A"

            oos_lbl = (f"{oos_s.date()} → {oos_e.date()}"
                       if oos_s and oos_e else "N/A")

            sweep_rows.append({
                "Start Year":      yr,
                "OOS Period":      oos_lbl,
                "Next Pick":       next_pick,
                "Ann. Return":     f"{mets.get('ann_return', 0)*100:.2f}%",
                "Sharpe":          f"{mets.get('sharpe', 0):.3f}",
                "Max DD":          f"{mets.get('max_dd', 0)*100:.2f}%",
                "Calmar":          f"{mets.get('calmar', 0):.2f}",
                "Win Rate":        f"{mets.get('win_rate', 0)*100:.1f}%",
                "Sortino":         f"{mets.get('sortino', 0):.3f}",
                "Volatility":      f"{mets.get('volatility', 0)*100:.2f}%",
                "Avg Hold Days":   mets.get("avg_hold_days", "N/A"),
                "# Switches":      mets.get("n_switches", "N/A"),
                "_ann_return_raw": mets.get("ann_return", 0),
                "_sharpe_raw":     mets.get("sharpe", 0),
            })

            if eq_oos is not None:
                eq_s = (eq_oos["strategy"]
                        if isinstance(eq_oos, pd.DataFrame)
                        and "strategy" in eq_oos.columns else eq_oos)
                equity_traces[yr] = eq_s

        except Exception as e:
            st.warning(f"Year {yr} skipped: {e}")

    progress.progress(1.0, text="Done.")

    if not sweep_rows:
        st.error("Could not load any year models.")
        st.stop()

    # Consensus signal
    st.markdown("---")
    all_picks     = [r["Next Pick"] for r in sweep_rows]
    pick_counts   = pd.Series(all_picks).value_counts()
    top_pick      = pick_counts.index[0]
    top_count     = int(pick_counts.iloc[0])
    consensus_pct = top_count / len(all_picks) * 100

    c1, c2, c3 = st.columns(3)
    etf_color = ETF_COLORS.get(top_pick, "#333")
    c1.markdown(
        f"<div style='padding:16px;border-radius:10px;"
        f"background:{etf_color}22;border-left:5px solid {etf_color};'>"
        f"<span style='font-size:1.8rem;font-weight:700;color:{etf_color}'>"
        f"{top_pick}</span><br>"
        f"<span style='color:#888;font-size:0.85rem'>Consensus Next Allocation</span>"
        f"</div>", unsafe_allow_html=True)
    c2.metric("Consensus Strength",
              f"{consensus_pct:.0f}%",
              f"{top_count}/{len(all_picks)} year-models agree")
    c3.metric("Models Compared", len(sweep_rows))

    st.caption("Vote breakdown: " + "  |  ".join(
        f"{etf}: {cnt}" for etf, cnt in pick_counts.items()))

    # Metrics table
    st.markdown("---")
    st.subheader("📊 OOS Metrics by Start Year")
    display_cols = ["Start Year", "OOS Period", "Next Pick",
                    "Ann. Return", "Sharpe", "Max DD", "Calmar",
                    "Win Rate", "Sortino", "Volatility",
                    "Avg Hold Days", "# Switches"]
    st.dataframe(
        pd.DataFrame(sweep_rows)[display_cols],
        hide_index=True, use_container_width=True, height=400)

    # Equity overlay
    st.markdown("---")
    st.subheader("📉 OOS Equity Curves — All Years (normalised to 1.0)")
    colors_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
                    "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
                    "#98df8a", "#ff9896", "#c5b0d5"]
    fig_sweep = go.Figure()
    for idx, (yr, eq_s) in enumerate(sorted(equity_traces.items())):
        eq_norm = eq_s / eq_s.iloc[0]
        fig_sweep.add_trace(go.Scatter(
            x=eq_norm.index, y=eq_norm.values,
            name=f"From {yr}",
            line=dict(color=colors_cycle[idx % len(colors_cycle)], width=1.8),
        ))
    fig_sweep.update_layout(
        height=420, xaxis_title="Date", yaxis_title="Normalised Value (start=1.0)",
        legend=dict(orientation="h", y=1.02),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=30, b=0))
    fig_sweep.update_xaxes(showgrid=True, gridcolor="#eee")
    fig_sweep.update_yaxes(showgrid=True, gridcolor="#eee")
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Sharpe bar chart
    st.markdown("---")
    st.subheader("📊 Sharpe Ratio by Start Year")
    sharpe_vals = [r["_sharpe_raw"] for r in sweep_rows]
    sharpe_yrs  = [r["Start Year"] for r in sweep_rows]
    fig_sharpe  = go.Figure(go.Bar(
        x=[str(y) for y in sharpe_yrs], y=sharpe_vals,
        marker_color=["#2ca02c" if v > 0 else "#d62728" for v in sharpe_vals],
        text=[f"{v:.2f}" for v in sharpe_vals], textposition="outside",
    ))
    fig_sharpe.update_layout(
        height=300, xaxis_title="Start Year", yaxis_title="Sharpe Ratio",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig_sharpe, use_container_width=True)

    st.markdown("---")
    st.caption(
        f"P2-ETF-ENTROPY · Consensus Sweep · {len(sweep_rows)} year-models · "
        f"Entropy 2026, 28, 84"
    )
