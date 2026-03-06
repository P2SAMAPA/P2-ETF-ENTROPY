"""
reseed.py - ONE-TIME script to build complete dataset from 2008.
Run manually: python reseed.py
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
from huggingface_hub import HfApi, CommitOperationAdd

# --- Configuration ---
HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ["TLT", "VNQ", "GLD", "SLV", "HYG", "VCIT", "LQD", "AGG", "SPY"]
START_DATE = "2008-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

# Create a session with a browser-like user-agent to reduce rate limiting
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

def fetch_etf_data(ticker, start, end):
    """Fetch ETF data with robust retry logic and session reuse."""
    for attempt in range(6):  # up to 6 attempts
        try:
            # Use the shared session
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
                session=session  # reuse session with custom headers
            )
            
            if df.empty:
                raise ValueError(f"No data for {ticker}")
            
            # Handle MultiIndex columns (same as before)
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
            
            print(f"  ✅ {ticker}: {len(df)} rows")
            return df
            
        except Exception as e:
            # Check if it's a rate limit error
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["rate limit", "too many requests", "429", "ratelimit"])
            
            if is_rate_limit and attempt < 5:
                # Exponential backoff with jitter: 30s, 60s, 120s, 240s, 480s
                wait = 30 * (2 ** attempt) + random.randint(5, 15)
                print(f"  ⚠️ Rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Failed to fetch {ticker} after {attempt+1} attempts: {e}")
                return None  # return None instead of raising to continue with other tickers
    
    return None

def main():
    print("=" * 60)
    print("FULL RESEED FROM 2008-01-01")
    print("=" * 60)
    
    # 1. Fetch ETF data
    print(f"\n📥 Downloading ETFs ({START_DATE} to {END_DATE})...")
    etf_data = {}
    failed_tickers = []
    
    for ticker in ETF_LIST:
        print(f"\n--- {ticker} ---")
        series = fetch_etf_data(ticker, START_DATE, END_DATE)
        if series is not None:
            etf_data[ticker] = series
        else:
            failed_tickers.append(ticker)
    
    if not etf_data:
        raise RuntimeError("No ETF data could be fetched. Aborting.")
    
    if failed_tickers:
        print(f"\n⚠️ Failed tickers: {failed_tickers} — continuing with {len(etf_data)} tickers.")
    
    # Combine into DataFrame
    etf_df = pd.DataFrame(etf_data)
    print(f"\n📊 ETF DataFrame shape: {etf_df.shape}")
    print(f"   Date range: {etf_df.index[0].date()} to {etf_df.index[-1].date()}")
    
    # 2. Fetch T-Bill data
    print(f"\n📥 Downloading 3-Month T-Bill from FRED...")
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    tbill = fred.get_series("DGS3MO", observation_start=START_DATE, observation_end=END_DATE)
    tbill_df = tbill.to_frame("3MTBILL").ffill()
    print(f"   T-Bill rows: {len(tbill_df)}")
    
    # 3. Merge datasets
    full_df = etf_df.join(tbill_df, how='left').ffill().bfill()
    full_df.index = pd.to_datetime(full_df.index).tz_localize(None)
    
    print(f"\n✅ Merged dataset shape: {full_df.shape}")
    print(f"   Date range: {full_df.index[0].date()} to {full_df.index[-1].date()}")
    
    # Verify all columns present
    all_cols = ETF_LIST + ["3MTBILL"]
    missing_cols = [c for c in all_cols if c not in full_df.columns]
    if missing_cols:
        print(f"⚠️ Warning: Missing columns: {missing_cols}")
    
    # 4. Save locally
    full_df.to_parquet("raw_data.parquet")
    file_size = os.path.getsize("raw_data.parquet")
    print(f"\n💾 Saved raw_data.parquet ({file_size:,} bytes)")
    
    # 5. Create metadata
    metadata = {
        "last_data_update": str(full_df.index[-1].date()),
        "last_training_date": None,
        "best_ma_window": None,
        "dataset_version": 1,
        "seed_date": str(datetime.today().date()),
        "rows": len(full_df),
        "columns": list(full_df.columns)
    }
    
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"📝 Created metadata.json")
    
    # 6. Upload to Hugging Face
    print(f"\n📤 Uploading to Hugging Face: {HF_DATASET_REPO}")
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN environment variable not set")
    
    api = HfApi(token=token)
    
    for local_file, repo_file in [
        ("raw_data.parquet", "raw_data.parquet"),
        ("metadata.json", "metadata.json")
    ]:
        with open(local_file, "rb") as f:
            content = f.read()
        
        api.create_commit(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=token,
            commit_message=f"Reseed: {repo_file} - {metadata['last_data_update']}",
            operations=[CommitOperationAdd(
                path_in_repo=repo_file,
                path_or_fileobj=content
            )],
        )
        print(f"  ✅ Uploaded {repo_file}")
    
    print("\n" + "=" * 60)
    print(f"🎉 RESEED COMPLETE - {len(full_df)} rows, {len(full_df.columns)} columns")
    print("=" * 60)

if __name__ == "__main__":
    main()
