"""
update_data.py - Daily incremental update script with Stooq fallback.
Fetches new data since last update, using Yahoo Finance first, then Stooq if needed.
"""
import os
import json
import time
import random
import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from datetime import datetime, timedelta
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd

# --- Configuration ---
HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "HYG", "VCIT", "LQD", "AGG", "SPY"]
LOCAL_DATA_FILE = "raw_data.parquet"
LOCAL_META_FILE = "metadata.json"

# Create a session with a browser-like user-agent to reduce rate limiting
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

def download_current_dataset(token):
    """Download current dataset from HF Hub."""
    print("📥 Downloading current dataset from Hugging Face...")
    
    parquet_path = hf_hub_download(
        repo_id=HF_DATASET_REPO,
        filename="raw_data.parquet",
        repo_type="dataset",
        token=token
    )
    df = pd.read_parquet(parquet_path)
    
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
        metadata = {
            "last_data_update": str(df.index[-1].date()),
            "dataset_version": 1
        }
    
    return df, metadata

def fetch_new_etf_data_yf(ticker, start, end):
    """Fetch ETF data from Yahoo Finance for the given date range."""
    for attempt in range(4):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
                session=session
            )
            
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            
            # Handle MultiIndex columns
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
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["rate limit", "too many requests", "429", "ratelimit"])
            
            if is_rate_limit and attempt < 3:
                wait = 15 * (2 ** attempt) + random.randint(5, 10)
                print(f"    ⚠️ YF rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ❌ YF failed for {ticker}: {e}")
                return None
    return None

def fetch_new_etf_data_stooq(ticker, start, end):
    """Fetch ETF data from Stooq (fallback) for the given date range."""
    stooq_symbol = ticker.lower() + '.us'
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    
    for attempt in range(3):
        try:
            df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
            if df.empty:
                raise ValueError(f"No data from Stooq for {ticker}")
            
            # Filter by date range
            df = df.sort_index()
            mask = (df.index >= start) & (df.index <= end)
            df = df.loc[mask]
            
            if df.empty:
                raise ValueError(f"No data in date range for {ticker} from Stooq")
            
            series = df['Close']
            series.name = ticker
            series.index = pd.to_datetime(series.index).tz_localize(None)
            return series
            
        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                print(f"    ⚠️ Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    ❌ Stooq failed for {ticker} after 3 attempts.")
                return None
    return None

def main():
    print("=" * 60)
    print("DAILY DATA UPDATE (with Stooq fallback)")
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
        # Try Yahoo Finance first
        series = fetch_new_etf_data_yf(ticker, start_new, end_exclusive)
        
        # If YF fails, try Stooq
        if series is None:
            print(f"    🔄 Trying Stooq fallback for {ticker}...")
            series = fetch_new_etf_data_stooq(ticker, start_new, end_new)  # Stooq uses inclusive end
        
        if series is not None and not series.empty:
            new_etf_data[ticker] = series
            source = "Stooq" if series is not None else "YF"
            print(f"    ✅ {len(series)} new rows from {source}")
        else:
            print(f"    ⚠️ No new data for {ticker} from any source.")
    
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
        # Build a DataFrame from all new ETF data
        if new_etf_data:
            new_etf_df = pd.DataFrame(new_etf_data)
            # Ensure index is DatetimeIndex
            new_etf_df.index = pd.to_datetime(new_etf_df.index)
        else:
            new_etf_df = pd.DataFrame()
        
        # Add T-Bill data if available
        if new_tbill_df is not None and not new_tbill_df.empty:
            new_tbill_df.index = pd.to_datetime(new_tbill_df.index)
            # Combine ETF and T-Bill data
            if not new_etf_df.empty:
                new_combined = new_etf_df.join(new_tbill_df, how='outer')
            else:
                new_combined = new_tbill_df
        else:
            new_combined = new_etf_df
        
        # Concatenate with existing data (this properly handles new dates)
        updated_df = pd.concat([current_df, new_combined])
        
        # Remove any duplicate indices (keep last/newest)
        updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
        
        # Sort by date to ensure chronological order
        updated_df = updated_df.sort_index()
        
        # Forward fill any missing values (in case some ETFs have data on days others don't)
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
