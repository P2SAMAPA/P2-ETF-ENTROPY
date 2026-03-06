"""
update_data.py - Daily incremental update script.
Runs via GitHub Actions to add new data since last update.
"""
import os
import json
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

# --- Configuration ---
HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "HYG", "VCIT", "LQD", "AGG", "SPY"]
LOCAL_DATA_FILE = "raw_data.parquet"
LOCAL_META_FILE = "metadata.json"

def download_current_dataset(token):
    """Download current dataset from HF Hub."""
    print("📥 Downloading current dataset from Hugging Face...")
    
    # Download parquet file
    parquet_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="raw_data.parquet",
        repo_type="dataset",
        token=token
    )
    df = pd.read_parquet(parquet_path)
    
    # Download metadata
    try:
        meta_path = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename="metadata.json",
            repo_type="dataset",
            token=token
        )
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
    except Exception:
        # Create basic metadata if not found
        metadata = {
            "last_data_update": str(df.index[-1].date()),
            "dataset_version": 1
        }
    
    return df, metadata

def fetch_new_etf_data(ticker, start_date, end_date):
    """Fetch data for one ticker from start_date to end_date."""
    try:
        df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
            threads=False
        )
        
        if df.empty:
            return None
        
        # Extract Close prices (same logic as reseed)
        if isinstance(df.columns, pd.MultiIndex):
            df = df['Close']
            if isinstance(df, pd.DataFrame):
                df = df.iloc[:, 0]
        else:
            close_cols = [c for c in df.columns if 'Close' in str(c)]
            if close_cols:
                df = df[close_cols[0]]
            else:
                df = df.iloc[:, 0]
        
        if isinstance(df, pd.DataFrame):
            df = df.squeeze()
        df.name = ticker
        
        return df
    except Exception as e:
        print(f"    ⚠️ Error fetching {ticker}: {e}")
        return None

def main():
    print("=" * 60)
    print("DAILY DATA UPDATE")
    print("=" * 60)
    
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set")
    
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        raise RuntimeError("FRED_API_KEY not set")
    
    # 1. Load current dataset
    current_df, metadata = download_current_dataset(token)
    last_update = pd.to_datetime(metadata["last_data_update"])
    today = datetime.now()
    
    print(f"Current data: up to {last_update.date()}")
    print(f"Today: {today.date()}")
    
    # If already up to date, exit
    if last_update.date() >= today.date():
        print("✅ Data is already up to date.")
        return
    
    # 2. Define date range for new data
    start_new = (last_update + timedelta(days=1)).strftime("%Y-%m-%d")
    end_new = today.strftime("%Y-%m-%d")
    end_exclusive = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"\n📥 Fetching new data: {start_new} to {end_new}")
    
    # 3. Fetch new ETF data
    print("\nFetching ETF data...")
    new_etf_data = {}
    
    for ticker in ETF_LIST:
        print(f"  Processing {ticker}...")
        series = fetch_new_etf_data(ticker, start_new, end_exclusive)
        if series is not None and not series.empty:
            new_etf_data[ticker] = series
            print(f"    ✅ {len(series)} new rows")
    
    if not new_etf_data:
        print("⚠️ No new ETF data found.")
    else:
        new_etf_df = pd.DataFrame(new_etf_data)
        print(f"\n📊 New ETF data shape: {new_etf_df.shape}")
    
    # 4. Fetch new T-Bill data
    print("\nFetching new T-Bill data...")
    fred = Fred(api_key=fred_api_key)
    new_tbill = fred.get_series("DGS3MO", observation_start=start_new, observation_end=end_new)
    
    if not new_tbill.empty:
        new_tbill_df = new_tbill.to_frame("3MTBILL")
        print(f"  ✅ {len(new_tbill_df)} new rows")
    else:
        new_tbill_df = None
        print("  ⚠️ No new T-Bill data.")
    
    # 5. Merge new data with existing
    if new_etf_data or (new_tbill_df is not None and not new_tbill_df.empty):
        # Start with existing data
        updated_df = current_df.copy()
        
        # Append new ETF data
        if new_etf_data:
            for ticker in ETF_LIST:
                if ticker in new_etf_data:
                    new_series = new_etf_data[ticker]
                    updated_df.loc[new_series.index, ticker] = new_series
        
        # Append new T-Bill data
        if new_tbill_df is not None and not new_tbill_df.empty:
            updated_df.loc[new_tbill_df.index, "3MTBILL"] = new_tbill_df["3MTBILL"]
        
        # Forward fill any missing values (should not be needed, but safe)
        updated_df = updated_df.ffill()
        
        # Update metadata
        new_last_date = str(updated_df.index[-1].date())
        metadata["last_data_update"] = new_last_date
        metadata["dataset_version"] = metadata.get("dataset_version", 1) + 1
        
        print(f"\n✅ Updated dataset shape: {updated_df.shape}")
        print(f"   New last date: {new_last_date}")
        
        # 6. Save locally
        updated_df.to_parquet(LOCAL_DATA_FILE)
        with open(LOCAL_META_FILE, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # 7. Upload to Hugging Face
        print("\n📤 Uploading updated dataset to Hugging Face...")
        api = HfApi(token=token)
        
        for local_file, repo_file in [
            (LOCAL_DATA_FILE, "raw_data.parquet"),
            (LOCAL_META_FILE, "metadata.json")
        ]:
            with open(local_file, "rb") as f:
                content = f.read()
            
            api.create_commit(
                repo_id=HF_DATASET_REPO,
                repo_type="dataset",
                token=token,
                commit_message=f"Daily update: {new_last_date} (v{metadata['dataset_version']})",
                operations=[CommitOperationAdd(
                    path_in_repo=repo_file,
                    path_or_fileobj=content
                )],
            )
            print(f"  ✅ Uploaded {repo_file}")
        
        print("\n" + "=" * 60)
        print(f"🎉 UPDATE COMPLETE - Now up to {new_last_date}")
        print("=" * 60)
    else:
        print("\n⚠️ No new data found. Dataset unchanged.")

if __name__ == "__main__":
    main()
