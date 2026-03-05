#!/usr/bin/env python3
"""
train_models.py — P2-ETF-ENTROPY
==================================
Full training pipeline for GitHub Actions.

Steps
-----
1. Load dataset from HuggingFace
2. Build features for MA(3) and MA(5) — train/test split enforced
3. Compute DTW weights on training prices only (2008–2021)
4. Train TransferVotingModel for each MA window
5. Run backtest on OOS (test) set to pick best MA window
6. Save best model + metadata to HuggingFace

Outputs saved to HF dataset (P2SAMAPA/etf-entropy-dataset)
  models/transfer_voting_MA{3|5}.pkl
  models/best_model.json
  models/dtw_matrix_MA{3|5}.npy
  metadata.json  (updated last_training_date + best_ma_window)
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime

from data_loader import load_dataset, load_metadata, save_to_hf, HF_DATASET_REPO
from feature_engineering import (
    prepare_all_features, TARGET_ETFS, TRAIN_END
)
from transfer_voting import TransferVotingModel
from strategy_engine import StrategyEngine
from backtest import run_backtest
from metrics import calculate_metrics

ARTIFACT_DIR = "artifacts"
MA_WINDOWS   = [3, 5]


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_and_upload(obj_bytes_or_path, repo_path: str):
    """Write to a temp file then upload to HF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(repo_path)[1]) as f:
        if isinstance(obj_bytes_or_path, (bytes, bytearray)):
            f.write(obj_bytes_or_path)
            tmp = f.name
        else:
            tmp = obj_bytes_or_path   # already a file path
    save_to_hf(tmp, repo_path)
    if tmp != obj_bytes_or_path:
        os.unlink(tmp)


def _run_oos_backtest(
    model: TransferVotingModel,
    data_dict: dict,
    price_df: pd.DataFrame,
    tbill_series: pd.Series,
    tsl_pct: float = 15,
    tx_cost_bps: float = 25,
    z_threshold: float = 1.0,
) -> dict:
    """Run backtest on OOS (test) window, return metrics dict."""

    # Get test features (already normalised with train stats)
    X_test_dict = data_dict["features_test"]

    # Predict — full transfer voting on test features
    predictions_raw = model.predict_all_etfs(X_test_dict)

    # Wrap as pd.Series indexed by date
    predictions = {}
    for etf, preds in predictions_raw.items():
        idx = X_test_dict[etf].index
        predictions[etf] = pd.Series(preds, index=idx)

    # Align price_df to test window
    test_dates = sorted(set().union(*[s.index for s in predictions.values()]))
    price_test = price_df.loc[price_df.index.isin(test_dates)]
    tbill_test = tbill_series.loc[tbill_series.index.isin(test_dates)]

    engine = StrategyEngine(
        TARGET_ETFS,
        tsl_pct=tsl_pct,
        transaction_cost_bps=tx_cost_bps,
        z_score_threshold=z_threshold,
    )

    results = run_backtest(predictions, price_test, tbill_test, engine, test_dates)
    metrics = calculate_metrics(
        results["equity_curve"]["strategy"],
        results["returns"],
        results["risk_free"],
        results["audit_trail"],
    )
    return metrics, results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("P2-ETF-ENTROPY — Training Pipeline")
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    print("=" * 60)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ── 1. Load dataset ───────────────────────────────────────────────
    print("\nStep 1: Loading dataset from HuggingFace...")
    df = load_dataset()
    print(f"  Shape: {df.shape}  |  {df.index[0].date()} → {df.index[-1].date()}")

    # Verify required columns
    missing = [c for c in TARGET_ETFS + ["3MTBILL"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    price_df     = df[TARGET_ETFS].copy()
    tbill_series = df["3MTBILL"].copy()

    # Training-window prices only (for DTW — no look-ahead)
    train_price_df = price_df.loc[price_df.index <= pd.Timestamp(TRAIN_END)]
    print(f"  Training window for DTW: {train_price_df.index[0].date()} → "
          f"{train_price_df.index[-1].date()} ({len(train_price_df)} rows)")

    # ── 2. Build features for both MA windows ─────────────────────────
    print("\nStep 2: Building features...")
    data_by_ma = {}
    for ma in MA_WINDOWS:
        data_by_ma[ma] = prepare_all_features(df, ma_window=ma, train_end=TRAIN_END)

    # ── 3. Train TransferVotingModel for each MA window ───────────────
    print("\nStep 3: Training models...")
    models    = {}
    oos_metrics = {}

    for ma in MA_WINDOWS:
        print(f"\n--- MA({ma}) ---")
        data_dict = data_by_ma[ma]

        model = TransferVotingModel(TARGET_ETFS, ma, ARTIFACT_DIR)
        model.fit(
            X_train_dict   = data_dict["features_train"],
            y_train_dict   = data_dict["targets_train"],
            train_price_df = train_price_df,
        )
        models[ma] = model

        # Save model locally
        local_path = os.path.join(ARTIFACT_DIR, f"transfer_voting_MA{ma}.pkl")
        model.save(local_path)

        # ── 4. OOS backtest to evaluate this MA window ────────────────
        print(f"\nStep 4a: OOS backtest for MA({ma})...")
        metrics, results = _run_oos_backtest(
            model, data_dict, price_df, tbill_series
        )
        oos_metrics[ma] = metrics
        print(f"  MA({ma}) OOS → Ann Return: {metrics.get('ann_return', 0)*100:.2f}%  "
              f"Sharpe: {metrics.get('sharpe', 0):.3f}  "
              f"MaxDD: {metrics.get('max_dd', 0)*100:.2f}%")

    # ── 5. Select best MA window ──────────────────────────────────────
    print("\nStep 5: Selecting best MA window...")
    best_ma = max(MA_WINDOWS, key=lambda m: oos_metrics[m].get("ann_return", -np.inf))
    print(f"  Best MA window: {best_ma}  "
          f"(Ann Return = {oos_metrics[best_ma].get('ann_return',0)*100:.2f}%)")

    # ── 6. Determine OOS date range ───────────────────────────────────
    best_X_test   = data_by_ma[best_ma]["features_test"]
    oos_dates     = sorted(set().union(*[X.index for X in best_X_test.values()]))
    oos_start_str = str(oos_dates[0].date())  if oos_dates else ""
    oos_end_str   = str(oos_dates[-1].date()) if oos_dates else ""

    # ── 7. Save artifacts to HuggingFace ─────────────────────────────
    print("\nStep 6: Uploading artifacts to HuggingFace...")

    for ma in MA_WINDOWS:
        local_pkl = os.path.join(ARTIFACT_DIR, f"transfer_voting_MA{ma}.pkl")
        save_to_hf(local_pkl, f"models/transfer_voting_MA{ma}.pkl")
        print(f"  Uploaded transfer_voting_MA{ma}.pkl")

        dtw_path = os.path.join(ARTIFACT_DIR, f"dtw_matrix_MA{ma}.npy")
        if os.path.exists(dtw_path):
            save_to_hf(dtw_path, f"models/dtw_matrix_MA{ma}.npy")
            print(f"  Uploaded dtw_matrix_MA{ma}.npy")

    # best_model.json — read by Streamlit app
    best_model_info = {
        "best_ma_window":    best_ma,
        "oos_start_date":    oos_start_str,
        "oos_end_date":      oos_end_str,
        "training_end_date": TRAIN_END,
        "ma3_ann_return":    round(oos_metrics[3].get("ann_return", 0), 6),
        "ma5_ann_return":    round(oos_metrics[5].get("ann_return", 0), 6),
        "ma3_sharpe":        round(oos_metrics[3].get("sharpe", 0), 4),
        "ma5_sharpe":        round(oos_metrics[5].get("sharpe", 0), 4),
        "last_trained":      datetime.utcnow().isoformat() + "Z",
        "etf_list":          TARGET_ETFS,
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

    print("\n" + "=" * 60)
    print("Training pipeline complete.")
    print(f"  Best MA:    {best_ma}")
    print(f"  OOS window: {oos_start_str} → {oos_end_str}")
    for ma in MA_WINDOWS:
        m = oos_metrics[ma]
        print(f"  MA({ma}): Return={m.get('ann_return',0)*100:.2f}%  "
              f"Sharpe={m.get('sharpe',0):.3f}  "
              f"MaxDD={m.get('max_dd',0)*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
