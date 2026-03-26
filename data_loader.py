"""
Data Loader Module — fixed
  - download_tbill_data: guard start_date > end_date → return empty frame
  - download_etf_data: same guard
  - Both accept explicit end_date (no silent today default causing stale end)
  - Now fetches all ETFs (both Option A and Option B) for data pipeline.
  - Option‑aware artifact loading added.
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
from config import ALL_TICKERS   # <-- new import

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
# Original ETF list is kept for backward compatibility (used only for Option A artifacts)
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
    # Use ALL_TICKERS to fetch all 18 ETFs
    data = yf.download(
        ALL_TICKERS, start=start_date, end=end_exclusive,
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
    """
    Upload file to HuggingFace using create_commit + bytes.
    Reads file into memory first — works regardless of working directory.
    Raises on failure so caller can surface the error.
    """
    from huggingface_hub import CommitOperationAdd
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — cannot upload to HuggingFace")

    with open(file_path, "rb") as f:
        file_bytes = f.read()

    api = HfApi(token=token)
    api.create_commit(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
        commit_message=f"Update {repo_path} — {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M UTC')}",
        operations=[CommitOperationAdd(
            path_in_repo=repo_path,
            path_or_fileobj=file_bytes,
        )],
    )
    print(f"✅ Uploaded {repo_path} to HF ({len(file_bytes):,} bytes)")


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

    # Verify that all expected columns exist (original + new equity)
    for col in ALL_TICKERS + ["3MTBILL"]:
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


# =============================================================================
# NEW: Option‑aware artifact loading/saving
# =============================================================================

def _option_artifact_path(option: str, filename: str) -> str:
    """
    Return the local path for an option‑specific artifact.
    Option 'a' -> artifacts/option_a/<filename>
    Option 'b' -> artifacts/option_b/<filename>
    """
    return os.path.join("artifacts", f"option_{option}", filename)


def load_artifacts(option: str, filename: str):
    """
    Load an artifact (e.g., model weights, predictions, etc.) for a given option.
    Tries HF first, then local fallback.
    """
    repo_path = _option_artifact_path(option, filename)
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=repo_path,
            repo_type="dataset",
        )
        # Determine if it's a parquet or a pickle/other based on extension
        if filename.endswith(".parquet"):
            df = pd.read_parquet(path)
            print(f"Loaded {repo_path} from HF ({len(df)} rows)")
            return df
        else:
            # For pickle or other binary files, return raw bytes (caller will interpret)
            with open(path, "rb") as f:
                return f.read()
    except Exception as e:
        print(f"Could not load {repo_path} from HF: {e}")
        # Try local fallback
        local_path = _option_artifact_path(option, filename)
        if os.path.exists(local_path):
            if filename.endswith(".parquet"):
                df = pd.read_parquet(local_path)
                print(f"Loaded {repo_path} from local fallback ({len(df)} rows)")
                return df
            else:
                with open(local_path, "rb") as f:
                    return f.read()
        raise FileNotFoundError(f"Artifact {repo_path} not found.")


def save_artifacts(option: str, filename: str, data) -> bool:
    """
    Save an artifact (DataFrame or bytes) for a given option to local disk and upload to HF.
    """
    local_dir = _option_artifact_path(option, "")
    os.makedirs(local_dir, exist_ok=True)
    local_path = _option_artifact_path(option, filename)
    repo_path = _option_artifact_path(option, filename)

    if isinstance(data, pd.DataFrame):
        data.to_parquet(local_path)
    else:
        with open(local_path, "wb") as f:
            f.write(data)

    try:
        save_to_hf(local_path, repo_path)
        return True
    except Exception as e:
        print(f"Failed to upload {repo_path}: {e}")
        return False
