"""
Data Loader Module
Handles downloading, seeding, and loading ETF data from yfinance and FRED
Integrates with HuggingFace Dataset repository for storage
"""

import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
import json
from datetime import timedelta, datetime
from utils import get_latest_trading_day

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "VCIT", "HYG", "LQD", "SPY", "AGG"]


def download_etf_data(start_date, end_date=None):
    """
    Download ETF price data from yfinance
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str, optional
        End date in 'YYYY-MM-DD' format (default: today)
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Downloading ETF data from {start_date} to {end_date}...")
    data = yf.download(ETF_LIST, start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    # Handle multi-index columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    else:
        data = data.to_frame("Close")
    
    return data


def download_tbill_data(start_date, end_date=None):
    """
    Download 3-month T-Bill data from FRED
    
    Parameters:
    -----------
    start_date : str
        Start date
    end_date : str, optional
        End date (default: today)
    """
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print(f"Downloading T-Bill data from {start_date} to {end_date}...")
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    tbill = fred.get_series("DGS3MO", observation_start=start_date, observation_end=end_date)
    tbill = tbill.to_frame("3MTBILL")
    tbill = tbill.ffill()  # Forward fill missing values
    
    return tbill


def load_metadata():
    """Load metadata from HF dataset repo"""
    try:
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="metadata.json",
            repo_type="dataset"
        )
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Could not load metadata: {e}")
        return None


def save_to_hf(file_path, repo_path):
    """Upload file to HuggingFace dataset repository"""
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=repo_path,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )
    print(f"Uploaded {repo_path} to HF")


def load_dataset():
    """
    Load dataset from HuggingFace or local fallback
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ETF prices and T-Bill data
    """
    try:
        # Try to download from HF
        path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="raw_data.parquet",
            repo_type="dataset"
        )
        df = pd.read_parquet(path)
        print(f"Loaded dataset from HF: {df.shape}, dates: {df.index[0]} to {df.index[-1]}")
        return df
    except Exception as e:
        print(f"HF load failed: {e}")
        # Try local fallback
        if os.path.exists("raw_data.parquet"):
            df = pd.read_parquet("raw_data.parquet")
            print(f"Loaded dataset from local: {df.shape}")
            return df
        else:
            raise FileNotFoundError("No dataset found. Run seed_dataset() first.")


def seed_dataset(end_date=None):
    """
    Initial seed of dataset from 2008 to present (or specified end_date)
    One-time operation to populate HF dataset repository
    
    Parameters:
    -----------
    end_date : str, optional
        End date for seeding (default: latest trading day)
    """
    print("="*60)
    print("SEEDING DATASET FROM 2008")
    print("="*60)
    
    start_date = "2008-01-01"
    
    if end_date is None:
        end_date = get_latest_trading_day()
    
    print(f"Date range: {start_date} to {end_date}")
    
    # Download data
    etf = download_etf_data(start_date, end_date)
    tbill = download_tbill_data(start_date, end_date)
    
    # Join and clean
    df = etf.join(tbill, how="left")
    df = df.ffill().bfill()  # Forward and backward fill
    
    # Ensure all expected columns present
    expected_cols = ETF_LIST + ["3MTBILL"]
    for col in expected_cols:
        if col not in df.columns:
            print(f"Warning: {col} not in dataset")
    
    # Save locally first
    df.to_parquet("raw_data.parquet")
    print(f"Saved raw data: {df.shape}")
    
    # Create metadata
    metadata = {
        "last_data_update": str(end_date),
        "last_training_date": None,
        "best_ma_window": None,
        "dataset_version": 1,
        "seed_date": str(datetime.today().date())
    }
    
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Upload to HF
    print("Uploading to HuggingFace...")
    save_to_hf("raw_data.parquet", "raw_data.parquet")
    save_to_hf("metadata.json", "metadata.json")
    
    print("="*60)
    print(f"SEED COMPLETE! Dataset available till: {end_date}")
    print("="*60)
    
    return df


def get_last_update_date():
    """Helper to get last update date from metadata"""
    meta = load_metadata()
    if meta:
        return pd.to_datetime(meta["last_data_update"])
    return None
