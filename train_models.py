#!/usr/bin/env python3
"""
Training script for GitHub Actions.
Reads TRAINING_START_YEAR env var (set by train.yml).
Saves artifacts to models/year_XXXX/ in HF dataset, stamped with run date.
Skips upload if a model for this year was already run today.
"""
import os
import sys
import time
import json
import glob
import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd, list_repo_files
from data_loader import load_dataset
from feature_engineering import prepare_all_features
from ma_optimizer import optimize_ma_window, MA_WINDOWS

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"


def already_trained_today(token: str, start_year: int) -> bool:
    """Check if a model for this year was already uploaded today."""
    today     = datetime.date.today().isoformat()
    hf_prefix = f"models/year_{start_year}/"
    try:
        files = list(list_repo_files(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token,
        ))
        for f in files:
            if f.startswith(hf_prefix) and today in f:
                print(f"  ⏭️  Model for year {start_year} already trained today ({today}). Skipping.")
                return True
    except Exception as e:
        print(f"  ⚠️  Could not check existing runs: {e}")
    return False


def upload_artifacts_to_hf(artifact_path: str, token: str,
                            start_year: int, run_date: str):
    """Upload all artifact files to HuggingFace under models/year_XXXX/."""
    print("\n📤 Uploading artifacts to HuggingFace...")
    api = HfApi(token=token)

    patterns = [
        f"{artifact_path}/*.pkl",
        f"{artifact_path}/*.json",
        f"{artifact_path}/*.npy",
    ]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern))

    if not files:
        print("  ⚠️ No artifact files found to upload.")
        return

    hf_folder = f"models/year_{start_year}"
    operations = []
    for local_path in files:
        filename  = os.path.basename(local_path)
        # Stamp filename with run date so multiple runs don't overwrite
        # e.g. transfer_voting_MA3_2026-03-07.pkl
        name, ext = os.path.splitext(filename)
        stamped   = f"{name}_{run_date}{ext}"
        repo_path = f"{hf_folder}/{stamped}"

        with open(local_path, "rb") as f:
            content = f.read()
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=content,
            )
        )
        print(f"  + {repo_path} ({len(content)//1024}KB)")

    # Also upload an un-stamped latest pointer for each file
    # so Streamlit can always find models/year_2012/best_model.json
    for local_path in files:
        filename  = os.path.basename(local_path)
        repo_path = f"{hf_folder}/{filename}"
        with open(local_path, "rb") as f:
            content = f.read()
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=content,
            )
        )

    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        commit_message=(
            f"Model year_{start_year} update [{run_date}]"
        ),
        operations=operations,
    )
    print(f"  ✅ Uploaded {len(operations)} files to {HF_DATASET_REPO}/{hf_folder}")


def main():
    t0 = time.time()

    # ── Read start year from environment (set by train.yml) ───────────
    start_year_str = os.getenv("TRAINING_START_YEAR", "2012")
    try:
        start_year = int(start_year_str)
    except ValueError:
        raise RuntimeError(f"Invalid TRAINING_START_YEAR: '{start_year_str}'")

    run_date = datetime.date.today().isoformat()
    print(f"Starting retraining — start_year={start_year}, run_date={run_date}")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    # Skip if already trained today for this year
    if already_trained_today(token, start_year):
        sys.exit(0)

    df = load_dataset()
    print(f"Dataset shape: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")

    etf_list = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]
    required = etf_list + ["3MTBILL"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    price_df = df[etf_list].copy()
    tbill_df = df["3MTBILL"].copy()

    # Prepare features for every MA window using the requested start year
    data_dict = {}
    for ma in MA_WINDOWS:
        print(f"\nPreparing features MA({ma}) …")
        t = time.time()
        data_dict[ma] = prepare_all_features(df, ma_window=ma, year_start=start_year)
        print(f"  MA({ma}) features ready in {round(time.time()-t,1)}s")

    # Optimise + train
    best_ma, results = optimize_ma_window(
        etf_list  = etf_list,
        data_dict = data_dict,
        price_df  = price_df,
        tbill_df  = tbill_df,
    )

    total = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"Training complete in {total}s  |  Best MA: MA({best_ma})  |  Year: {start_year}")
    for w, r in sorted(results.items()):
        print(f"  MA({w}): val ann_return = {r*100:.2f}%")
    print("="*60)

    # Enrich best_model.json with year and run date
    best_model_path = "artifacts/best_model.json"
    if os.path.exists(best_model_path):
        with open(best_model_path) as f:
            bm = json.load(f)
        bm["start_year"]   = start_year
        bm["run_date"]     = run_date
        bm["last_trained"] = run_date
        with open(best_model_path, "w") as f:
            json.dump(bm, f, indent=2)

    # Upload all artifacts to HuggingFace under models/year_XXXX/
    upload_artifacts_to_hf("artifacts", token, start_year, run_date)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
