#!/usr/bin/env python3
"""
Training script for GitHub Actions.
Reads TRAINING_START_YEAR env var (set by train.yml).
Saves artifacts to models/year_XXXX/ in HF dataset, stamped with run date.
Skips upload if a model for this year was already run today.
Cleans up old date-stamped files after upload — keeps only latest per year.
"""
import os
import sys
import re
import time
import json
import glob
import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete, list_repo_files
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
            repo_id=HF_DATASET_REPO, repo_type="dataset", token=token))
        for f in files:
            if f.startswith(hf_prefix) and today in f:
                print(f"  ⏭️  Model for year {start_year} already trained today ({today}). Skipping.")
                return True
    except Exception as e:
        print(f"  ⚠️  Could not check existing runs: {e}")
    return False


def delete_old_stamped_files(api, token: str, start_year: int, run_date: str):
    """
    Delete date-stamped files from models/year_XXXX/ that are NOT from today.
    Keeps: un-stamped latest pointers + today's stamped copy.
    Deletes: any *_YYYY-MM-DD.* where date != run_date.
    """
    hf_folder    = f"models/year_{start_year}"
    date_pattern = re.compile(r'_(\d{4}-\d{2}-\d{2})\.')
    try:
        all_files = list(list_repo_files(
            repo_id=HF_DATASET_REPO, repo_type="dataset", token=token))
    except Exception as e:
        print(f"  ⚠️  Could not list files for cleanup: {e}")
        return

    to_delete = []
    for f in all_files:
        if not f.startswith(hf_folder + "/"):
            continue
        m = date_pattern.search(f)
        if m and m.group(1) != run_date:
            to_delete.append(f)

    if not to_delete:
        print("  🧹 No old stamped files to clean up.")
        return

    print(f"  🧹 Deleting {len(to_delete)} old stamped files from {hf_folder}...")
    try:
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Cleanup old artifacts year_{start_year} (keeping {run_date})",
            operations=[CommitOperationDelete(path_in_repo=f) for f in to_delete],
        )
        for f in to_delete:
            print(f"    - {f}")
        print(f"  ✅ Cleanup complete.")
    except Exception as e:
        print(f"  ⚠️  Cleanup failed (non-fatal): {e}")


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

    hf_folder  = f"models/year_{start_year}"
    operations = []

    for local_path in files:
        filename = os.path.basename(local_path)
        name, ext = os.path.splitext(filename)

        # Date-stamped archive copy
        stamped   = f"{name}_{run_date}{ext}"
        repo_path = f"{hf_folder}/{stamped}"
        with open(local_path, "rb") as f:
            content = f.read()
        operations.append(CommitOperationAdd(
            path_in_repo=repo_path, path_or_fileobj=content))
        print(f"  + {repo_path} ({len(content)//1024}KB)")

        # Un-stamped latest pointer (always overwritten)
        operations.append(CommitOperationAdd(
            path_in_repo=f"{hf_folder}/{filename}",
            path_or_fileobj=content))

    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"Model year_{start_year} update [{run_date}]",
        operations=operations,
    )
    print(f"  ✅ Uploaded {len(operations)} files to {HF_DATASET_REPO}/{hf_folder}")

    # Clean up old stamped files — keep only today's
    delete_old_stamped_files(api, token, start_year, run_date)


def main():
    t0 = time.time()

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

    data_dict = {}
    for ma in MA_WINDOWS:
        print(f"\nPreparing features MA({ma}) …")
        t = time.time()
        data_dict[ma] = prepare_all_features(df, ma_window=ma, year_start=start_year)
        print(f"  MA({ma}) features ready in {round(time.time()-t,1)}s")

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

    upload_artifacts_to_hf("artifacts", token, start_year, run_date)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
