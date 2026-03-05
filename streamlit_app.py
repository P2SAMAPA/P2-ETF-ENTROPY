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
# ─── UPDATED IMPORTS ──────────────────────────────
from utils import get_next_trading_day, get_hero_next_date, get_oos_index

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
        # ─── UPDATED ────────────────────────────────
        equity_oos, returns_oos, rf_oos = get_oos_index(
            equity, results["returns"], results["risk_free"], oos_start, oos_end
        )
    else:
        equity_oos = equity
        returns_oos = results["returns"]
        rf_oos = results["risk_free"]

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
    # ─── UPDATED LINE ───────────────────────────────
    next_date     = get_hero_next_date(predictions, TARGET_ETFS)
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
