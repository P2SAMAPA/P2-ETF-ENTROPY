"""
One-time reseed script — run manually to rebuild raw_data.parquet from 2008 to today.
Usage: python reseed.py
"""
import os
import json
import pandas as pd
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, CommitOperationAdd
from datetime import datetime

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD", "SPY", "AGG"]

def reseed():
    print("=" * 60)
    print("FULL RESEED FROM 2008-01-01")
    print("=" * 60)

    start_date = "2008-01-01"
    end_date   = datetime.today().strftime("%Y-%m-%d")

    # ── ETF data ──────────────────────────────────────────────────────
    print(f"\nDownloading ETF data {start_date} → {end_date}...")
    from datetime import timedelta, datetime as dt
    end_exclusive = (dt.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
    
    etf = yf.download(
        ETF_LIST, start=start_date, end=end_exclusive,
        auto_adjust=True, progress=True, threads=False
    )
    if isinstance(etf.columns, pd.MultiIndex):
        etf = etf["Close"]
    print(f"ETF data: {etf.shape}  ({etf.index[0].date()} → {etf.index[-1].date()})")

    # ── T-Bill data ───────────────────────────────────────────────────
    print(f"\nDownloading T-Bill data {start_date} → {end_date}...")
    fred  = Fred(api_key=os.getenv("FRED_API_KEY"))
    tbill = fred.get_series("DGS3MO", observation_start=start_date, observation_end=end_date)
    tbill = tbill.to_frame("3MTBILL").ffill()
    print(f"T-Bill data: {tbill.shape}  ({tbill.index[0].date()} → {tbill.index[-1].date()})")

    # ── Merge ─────────────────────────────────────────────────────────
    df = etf.join(tbill, how="left").ffill().bfill()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    print(f"\nMerged dataset: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")

    # Verify all columns present
    for col in ETF_LIST + ["3MTBILL"]:
        status = "✅" if col in df.columns else "❌ MISSING"
        print(f"  {col}: {status}")

    # ── Save parquet ──────────────────────────────────────────────────
    df.to_parquet("raw_data.parquet")
    print(f"\nSaved raw_data.parquet ({os.path.getsize('raw_data.parquet'):,} bytes)")

    # ── Update metadata ───────────────────────────────────────────────
    last_date = str(df.index[-1].date())
    metadata  = {
        "last_data_update":   last_date,
        "last_training_date": None,
        "best_ma_window":     None,
        "dataset_version":    1,
        "seed_date":          str(datetime.today().date()),
    }
    # Preserve existing training metadata if present
    try:
        from huggingface_hub import hf_hub_download
        meta_path = hf_hub_download(HF_DATASET_REPO, "metadata.json", repo_type="dataset")
        with open(meta_path) as f:
            existing = json.load(f)
        metadata["last_training_date"] = existing.get("last_training_date")
        metadata["best_ma_window"]     = existing.get("best_ma_window")
        metadata["dataset_version"]    = existing.get("dataset_version", 1) + 1
    except Exception:
        pass

    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata: last_data_update = {last_date}")

    # ── Upload both files to HF ───────────────────────────────────────
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")

    api = HfApi(token=token)

    for local_file, repo_file in [("raw_data.parquet", "raw_data.parquet"),
                                   ("metadata.json",    "metadata.json")]:
        with open(local_file, "rb") as f:
            file_bytes = f.read()
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Reseed {repo_file} — {last_date} ({len(file_bytes):,} bytes)",
            operations=[CommitOperationAdd(
                path_in_repo=repo_file,
                path_or_fileobj=file_bytes,
            )],
        )
        print(f"✅ Uploaded {repo_file} ({len(file_bytes):,} bytes)")

    print("\n" + "=" * 60)
    print(f"RESEED COMPLETE — data covers {df.index[0].date()} → {last_date}")
    print("=" * 60)

if __name__ == "__main__":
    reseed()
