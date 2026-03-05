"""
Data Loader Module — fixed
  - download_tbill_data: guard start_date > end_date → return empty frame
  - download_etf_data: same guard
  - Both accept explicit end_date (no silent today default causing stale end)
"""

import os
import json
import time
from datetime import timedelta, datetime

import pandas as pd
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download

from utils import get_latest_trading_day

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD", "SPY", "AGG"]


def download_etf_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Guard: nothing to download
    if start_date > end_date:
        print(f"ETF: start {start_date} > end {end_date} — returning empty.")
        return pd.DataFrame()

    # yfinance end is EXCLUSIVE — add 1 day so the end_date itself is included
    from datetime import datetime, timedelta as _td
    end_exclusive = (datetime.strptime(end_date, "%Y-%m-%d") + _td(days=1)).strftime("%Y-%m-%d")

    print(f"Downloading ETF data from {start_date} to {end_date} (yf end={end_exclusive})...")
    data = yf.download(
        ETF_LIST, start=start_date, end=end_exclusive,
        auto_adjust=True, progress=False, threads=False
    )

    if data.empty:
        print("  yfinance returned empty DataFrame.")
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data.to_frame("Close")

    return data


def download_tbill_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    # Guard: FRED raises 400 if start > end
    if start_date > end_date:
        print(f"T-Bill: start {start_date} > end {end_date} — returning empty.")
        return pd.DataFrame(columns=["3MTBILL"])

    print(f"Downloading T-Bill data from {start_date} to {end_date}...")
    fred  = Fred(api_key=os.getenv("FRED_API_KEY"))
    tbill = fred.get_series(
        "DGS3MO",
        observation_start=start_date,
        observation_end=end_date,
    )
    tbill = tbill.to_frame("3MTBILL").ffill()
    return tbill


def load_metadata():
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="metadata.json",
            repo_type="dataset",
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load metadata: {e}")
        return None


def save_to_hf(file_path, repo_path):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=repo_path,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
    )
    print(f"Uploaded {repo_path} to HF")


def load_dataset():
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="raw_data.parquet",
            repo_type="dataset",
        )
        df = pd.read_parquet(path)
        print(f"Loaded from HF: {df.shape}  "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df
    except Exception as e:
        print(f"HF load failed: {e}")
        if os.path.exists("raw_data.parquet"):
            df = pd.read_parquet("raw_data.parquet")
            print(f"Loaded local fallback: {df.shape}")
            return df
        raise FileNotFoundError("No dataset found. Run seed_dataset() first.")


def seed_dataset(end_date=None):
    print("=" * 60)
    print("SEEDING DATASET FROM 2008")
    print("=" * 60)

    start_date = "2008-01-01"
    if end_date is None:
        end_date = str(get_latest_trading_day())

    etf   = download_etf_data(start_date, end_date)
    tbill = download_tbill_data(start_date, end_date)

    df = etf.join(tbill, how="left").ffill().bfill()

    for col in ETF_LIST + ["3MTBILL"]:
        if col not in df.columns:
            print(f"Warning: {col} missing from dataset")

    df.to_parquet("raw_data.parquet")
    print(f"Saved: {df.shape}")

    metadata = {
        "last_data_update":   str(end_date),
        "last_training_date": None,
        "best_ma_window":     None,
        "dataset_version":    1,
        "seed_date":          str(datetime.today().date()),
    }
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    save_to_hf("raw_data.parquet", "raw_data.parquet")
    save_to_hf("metadata.json",    "metadata.json")

    print(f"SEED COMPLETE → {end_date}")
    return df


def get_last_update_date():
    meta = load_metadata()
    return pd.to_datetime(meta["last_data_update"]) if meta else None
