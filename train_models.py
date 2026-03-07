#!/usr/bin/env python3
"""
Training script for GitHub Actions — TUNED
Prepares features for MA windows, runs optimize_ma_window,
then uploads all artifacts to HuggingFace dataset repo.
"""
import os
import sys
import time
import json
import glob
import numpy as np
import pandas as pd

from huggingface_hub import HfApi, CommitOperationAdd
from data_loader import load_dataset
from feature_engineering import prepare_all_features
from ma_optimizer import optimize_ma_window, MA_WINDOWS

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"


def upload_artifacts_to_hf(artifact_path: str, token: str):
    """Upload all artifact files to HuggingFace dataset repo."""
    print("\n📤 Uploading artifacts to HuggingFace...")
    api = HfApi(token=token)

    # Collect all files to upload
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

    operations = []
    for local_path in files:
        repo_path = local_path  # mirrors local path in HF repo
        with open(local_path, "rb") as f:
            content = f.read()
        operations.append(
            CommitOperationAdd(
                path_in_repo=repo_path,
                path_or_fileobj=content,
            )
        )
        print(f"  + {repo_path} ({len(content)//1024}KB)")

    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"Model artifacts update [{time.strftime('%Y-%m-%d')}]",
        operations=operations,
    )
    print(f"  ✅ Uploaded {len(operations)} files to {HF_DATASET_REPO}")


def main():
    t0 = time.time()
    print("Starting weekly retraining …")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")

    df = load_dataset()
    print(f"Dataset shape: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")

    etf_list = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD"]
    required = etf_list + ["3MTBILL"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    price_df = df[etf_list].copy()
    tbill_df = df["3MTBILL"].copy()

    # Prepare features for every MA window
    data_dict = {}
    for ma in MA_WINDOWS:
        print(f"\nPreparing features MA({ma}) …")
        t = time.time()
        data_dict[ma] = prepare_all_features(df, ma_window=ma)
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
    print(f"Training complete in {total}s  |  Best MA: MA({best_ma})")
    for w, r in sorted(results.items()):
        print(f"  MA({w}): val ann_return = {r*100:.2f}%")
    print("="*60)

    # Upload all artifacts to HuggingFace
    upload_artifacts_to_hf("artifacts", token)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
