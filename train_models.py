#!/usr/bin/env python3
"""
train_models.py — P2-ETF-ENTROPY
==================================
Full training pipeline for GitHub Actions.

80/10/10 split (train/val/OOS) computed dynamically from year_start.
Val set used for MA window selection. OOS never touched during training.

Steps
-----
1. Load dataset from HuggingFace
2. Build features for MA(3) and MA(5) with 80/10/10 split
3. Compute DTW weights on 80% train prices only
4. Train TransferVotingModel for each MA window
5. Evaluate on VAL set (10%) to pick best MA window
6. Report OOS metrics (10%) — informational only, not used for selection
7. Save best model + metadata + split info to HuggingFace

Timing is logged at each step so you can see where time is spent.
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import load_dataset, load_metadata, save_to_hf
from feature_engineering import (
    prepare_all_features, get_oos_dates, TARGET_ETFS
)
from transfer_voting import TransferVotingModel
from strategy_engine import StrategyEngine
from backtest import run_backtest
from metrics import calculate_metrics

ARTIFACT_DIR = "artifacts"
MA_WINDOWS   = [3, 5]


def _timer(label, start):
    elapsed = time.time() - start
    print(f"  ⏱  {label}: {elapsed:.1f}s")
    return elapsed


def _run_backtest_on_split(model, data_dict, split_key, price_df, tbill_series,
                            tsl_pct=15, tx_cost_bps=25, z_threshold=1.0):
    """
    Run backtest on a given split ('features_val' or 'features_test').
    Returns (metrics_dict, results_dict).
    """
    X_dict = data_dict[split_key]

    pred_raw = model.predict_all_etfs(X_dict)
    predictions = {
        etf: pd.Series(preds, index=X_dict[etf].index)
        for etf, preds in pred_raw.items()
        if etf in X_dict
    }

    if not predictions:
        return {}, {}

    all_dates   = sorted(set().union(*[set(s.index) for s in predictions.values()]))
    price_slice = price_df.loc[price_df.index.isin(all_dates)]
    tbill_slice = tbill_series.loc[tbill_series.index.isin(all_dates)]

    engine = StrategyEngine(
        TARGET_ETFS,
        tsl_pct=tsl_pct,
        transaction_cost_bps=tx_cost_bps,
        z_score_threshold=z_threshold,
    )
    results = run_backtest(predictions, price_slice, tbill_slice, engine, all_dates)
    metrics = calculate_metrics(
        results["equity_curve"]["strategy"],
        results["returns"],
        results["risk_free"],
        results["audit_trail"],
    )
    return metrics, results


def main():
    t_total = time.time()

    print("=" * 60)
    print("P2-ETF-ENTROPY — Training Pipeline")
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────
    t = time.time()
    print("\nStep 1: Loading dataset...")
    df = load_dataset()
    print(f"  Shape: {df.shape}  |  {df.index[0].date()} → {df.index[-1].date()}")
    _timer("load dataset", t)

    missing = [c for c in TARGET_ETFS + ["3MTBILL"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    price_df     = df[TARGET_ETFS].copy()
    tbill_series = df["3MTBILL"].copy()

    # Default training uses full history from 2008
    year_start = 2008

    # ── 2. Build features for both MA windows ─────────────────────────
    t = time.time()
    print(f"\nStep 2: Building features (year_start={year_start})...")
    data_by_ma = {}
    for ma in MA_WINDOWS:
        data_by_ma[ma] = prepare_all_features(df, ma_window=ma, year_start=year_start)
    _timer("feature engineering (both MA windows)", t)

    # Print split summary for first ETF
    ref_etf = TARGET_ETFS[0]
    for ma in MA_WINDOWS:
        sd = data_by_ma[ma]["split_dates"].get(ref_etf, {})
        if sd:
            print(f"\n  MA({ma}) split ({ref_etf}):")
            print(f"    Train : {sd['train_start']} → {sd['train_end']}  ({sd['n_train']} rows)")
            print(f"    Val   : {sd['val_start']}   → {sd['val_end']}    ({sd['n_val']} rows)")
            print(f"    OOS   : {sd['oos_start']}   → {sd['oos_end']}    ({sd['n_test']} rows)")

    # ── 3. Train models ───────────────────────────────────────────────
    models      = {}
    val_metrics = {}
    oos_metrics = {}

    for ma in MA_WINDOWS:
        print(f"\n{'='*60}")
        print(f"Step 3: Training TransferVotingModel — MA({ma})")
        print(f"{'='*60}")
        t = time.time()

        data_dict = data_by_ma[ma]

        # DTW on training prices only (80% window)
        # Determine train end date from split_dates
        train_end_date = data_dict["split_dates"][ref_etf]["train_end"]
        train_price_df = price_df.loc[price_df.index <= pd.Timestamp(train_end_date)]

        model = TransferVotingModel(TARGET_ETFS, ma, ARTIFACT_DIR)
        model.fit(
            X_train_dict   = data_dict["features_train"],
            y_train_dict   = data_dict["targets_train"],
            train_price_df = train_price_df,
        )
        models[ma] = model
        t_train = _timer(f"MA({ma}) training", t)

        # Save locally
        local_pkl = os.path.join(ARTIFACT_DIR, f"transfer_voting_MA{ma}.pkl")
        model.save(local_pkl)

        # ── 4. Val backtest (used for MA selection) ───────────────────
        t = time.time()
        print(f"\nStep 4a: Validation backtest — MA({ma})...")
        v_metrics, _ = _run_backtest_on_split(
            model, data_dict, "features_val", price_df, tbill_series
        )
        val_metrics[ma] = v_metrics
        _timer(f"MA({ma}) val backtest", t)
        print(f"  VAL  → Ann Return: {v_metrics.get('ann_return', 0)*100:.2f}%  "
              f"Sharpe: {v_metrics.get('sharpe', 0):.3f}  "
              f"MaxDD: {v_metrics.get('max_dd', 0)*100:.2f}%")

        # ── 5. OOS backtest (informational only) ──────────────────────
        t = time.time()
        print(f"\nStep 4b: OOS backtest — MA({ma}) (informational)...")
        o_metrics, _ = _run_backtest_on_split(
            model, data_dict, "features_test", price_df, tbill_series
        )
        oos_metrics[ma] = o_metrics
        _timer(f"MA({ma}) OOS backtest", t)
        print(f"  OOS  → Ann Return: {o_metrics.get('ann_return', 0)*100:.2f}%  "
              f"Sharpe: {o_metrics.get('sharpe', 0):.3f}  "
              f"MaxDD: {o_metrics.get('max_dd', 0)*100:.2f}%")

    # ── 6. Select best MA using VAL performance ───────────────────────
    print(f"\n{'='*60}")
    print("Step 5: MA window selection (by Val Ann Return)")
    print(f"{'='*60}")
    best_ma = max(MA_WINDOWS, key=lambda m: val_metrics[m].get("ann_return", -np.inf))
    print(f"  MA(3) val Ann Return: {val_metrics[3].get('ann_return',0)*100:.2f}%")
    print(f"  MA(5) val Ann Return: {val_metrics[5].get('ann_return',0)*100:.2f}%")
    print(f"  ✅ Selected: MA({best_ma})")

    # ── 7. Gather OOS split dates for the best MA ─────────────────────
    best_data   = data_by_ma[best_ma]
    oos_start, oos_end = get_oos_dates(best_data)

    # ── 8. Upload to HuggingFace ──────────────────────────────────────
    t = time.time()
    print(f"\nStep 6: Uploading to HuggingFace...")

    for ma in MA_WINDOWS:
        local_pkl = os.path.join(ARTIFACT_DIR, f"transfer_voting_MA{ma}.pkl")
        save_to_hf(local_pkl, f"models/transfer_voting_MA{ma}.pkl")
        print(f"  Uploaded transfer_voting_MA{ma}.pkl")

        dtw_path = os.path.join(ARTIFACT_DIR, f"dtw_matrix_MA{ma}.npy")
        if os.path.exists(dtw_path):
            save_to_hf(dtw_path, f"models/dtw_matrix_MA{ma}.npy")
            print(f"  Uploaded dtw_matrix_MA{ma}.npy")

    # best_model.json — Streamlit reads this on startup
    best_model_info = {
        "best_ma_window":      best_ma,
        "year_start":          year_start,
        "split_pct":           "80/10/10",
        "oos_start_date":      str(oos_start.date()) if oos_start else "",
        "oos_end_date":        str(oos_end.date())   if oos_end   else "",
        "ma3_val_ann_return":  round(val_metrics[3].get("ann_return", 0), 6),
        "ma5_val_ann_return":  round(val_metrics[5].get("ann_return", 0), 6),
        "ma3_val_sharpe":      round(val_metrics[3].get("sharpe", 0), 4),
        "ma5_val_sharpe":      round(val_metrics[5].get("sharpe", 0), 4),
        "ma3_oos_ann_return":  round(oos_metrics[3].get("ann_return", 0), 6),
        "ma5_oos_ann_return":  round(oos_metrics[5].get("ann_return", 0), 6),
        "ma3_oos_sharpe":      round(oos_metrics[3].get("sharpe", 0), 4),
        "ma5_oos_sharpe":      round(oos_metrics[5].get("sharpe", 0), 4),
        "split_dates":         best_data["split_dates"],
        "last_trained":        datetime.utcnow().isoformat() + "Z",
        "etf_list":            TARGET_ETFS,
    }

    bm_path = os.path.join(ARTIFACT_DIR, "best_model.json")
    with open(bm_path, "w") as f:
        json.dump(best_model_info, f, indent=2)
    save_to_hf(bm_path, "models/best_model.json")
    print("  Uploaded best_model.json")

    # Update metadata.json
    meta = load_metadata() or {}
    meta["last_training_date"] = datetime.utcnow().strftime("%Y-%m-%d")
    meta["best_ma_window"]     = best_ma
    meta_path = os.path.join(ARTIFACT_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    save_to_hf(meta_path, "metadata.json")
    print("  Updated metadata.json")

    _timer("HF upload", t)

    # ── Final summary ─────────────────────────────────────────────────
    total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print("Training pipeline complete.")
    print(f"  Total time : {total_elapsed/60:.1f} minutes")
    print(f"  Best MA    : {best_ma}")
    print(f"  OOS window : {oos_start.date() if oos_start else '?'} → "
          f"{oos_end.date() if oos_end else '?'}")
    print(f"\n  {'MA':>4}  {'Val Ret':>9}  {'Val Shrp':>9}  {'OOS Ret':>9}  {'OOS Shrp':>9}")
    for ma in MA_WINDOWS:
        marker = " ✅" if ma == best_ma else "   "
        print(f"  {ma:>4}{marker}  "
              f"{val_metrics[ma].get('ann_return',0)*100:>8.2f}%  "
              f"{val_metrics[ma].get('sharpe',0):>9.3f}  "
              f"{oos_metrics[ma].get('ann_return',0)*100:>8.2f}%  "
              f"{oos_metrics[ma].get('sharpe',0):>9.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
