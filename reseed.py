"""
reseed.py - ONE-TIME script to build complete dataset from 2008.
Uses Yahoo Finance first (bulk batching + rate_limit=True), falls back to Stooq if YF fails.
Now fetches all ETFs (both Option A and Option B) using ALL_TICKERS from config.py.
Run manually: python reseed.py
"""
import os
import sys
import json
import time
import random
import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from datetime import datetime, timedelta
from huggingface_hub import HfApi, CommitOperationAdd

# Import config to get all tickers
try:
    from config import ALL_TICKERS, OPTION_A_ETFS, OPTION_B_ETFS
except ImportError:
    print("ERROR: config.py not found. Please create it with ALL_TICKERS defined.")
    sys.exit(1)

# --- Configuration ---
HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ALL_TICKERS
START_DATE = "2008-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
BATCH_SIZE = 10  # Number of tickers per batch (adjust based on total count)
BATCH_DELAY = 5  # Seconds between batches
TICKER_DELAY = 1  # Seconds between individual fallback requests

# Create a session with a browser-like user-agent to reduce rate limiting
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
})

def fetch_etf_data_stooq(ticker, start, end):
    """
    Fetch ETF data from Stooq as fallback.
    Stooq symbol: ticker.lower() + '.us' (e.g., 'spy.us')
    """
    stooq_symbol = ticker.lower() + '.us'
    url = f"https://stooq.com/q/d/l/?s={stooq_symbol}&i=d"
    
    for attempt in range(3):
        try:
            df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
            if df.empty:
                raise ValueError(f"No data from Stooq for {ticker}")
            
            df = df.sort_index()
            mask = (df.index >= start) & (df.index <= end)
            df = df.loc[mask]
            
            if df.empty:
                raise ValueError(f"No data in date range for {ticker} from Stooq")
            
            series = df['Close']
            series.name = ticker
            series.index = pd.to_datetime(series.index).tz_localize(None)
            
            print(f"  ✅ {ticker} (Stooq): {len(series)} rows")
            return series
            
        except Exception as e:
            if attempt < 2:
                wait = 5 * (2 ** attempt) + random.randint(1, 5)
                print(f"  ⚠️ Stooq attempt {attempt+1} failed for {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Stooq failed for {ticker} after 3 attempts.")
                return None
    return None

def main():
    print("=" * 60)
    print("FULL RESEED FROM 2008-01-01 (with Stooq fallback & batching)")
    print(f"Tickers: {ETF_LIST}")
    print(f"Batch size: {BATCH_SIZE}, delay between batches: {BATCH_DELAY}s")
    print("=" * 60)
    
    # 1. Fetch ETF data in batches
    print(f"\n📥 Downloading ETFs in batches ({START_DATE} to {END_DATE})...")
    etf_data = {}
    failed_tickers = []
    
    for batch_start_idx in range(0, len(ETF_LIST), BATCH_SIZE):
        batch = ETF_LIST[batch_start_idx:batch_start_idx + BATCH_SIZE]
        batch_num = batch_start_idx // BATCH_SIZE + 1
        print(f"\n--- Batch {batch_num}: {batch} ---")
        
        # Attempt bulk download for the batch
        df_batch = None
        for attempt in range(3):
            try:
                df_batch = yf.download(
                    batch,
                    start=START_DATE,
                    end=END_DATE,
                    progress=False,
                    auto_adjust=True,
                    group_by='ticker',
                    rate_limit=True,      # Built-in rate limiting
                    threads=False,
                    session=session
                )
                break
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(k in err_str for k in ["rate limit", "too many requests", "429", "ratelimit"])
                if is_rate_limit and attempt < 2:
                    wait = 30 * (2 ** attempt) + random.randint(5, 15)
                    print(f"  ⚠️ Rate limited on batch {batch_num} (attempt {attempt+1}). Waiting {wait}s...")
                    time.sleep(wait)
                else:
                    print(f"  ❌ Batch {batch_num} failed after {attempt+1} attempts: {e}")
                    df_batch = None
                    break
        
        # If batch download failed, fall back to individual Stooq for each ticker in batch
        if df_batch is None:
            print(f"  🔄 Bulk download failed. Falling back to Stooq individually for each ticker in batch...")
            for ticker in batch:
                series = fetch_etf_data_stooq(ticker, START_DATE, END_DATE)
                if series is not None:
                    etf_data[ticker] = series
                else:
                    failed_tickers.append(ticker)
                time.sleep(TICKER_DELAY)  # Delay between individual fallback requests
        else:
            # Success: extract each ticker's Close price
            for ticker in batch:
                if ticker in df_batch.columns:
                    # For multi-ticker download, structure is df_batch[ticker]['Close']
                    if 'Close' in df_batch[ticker].columns:
                        series = df_batch[ticker]['Close']
                    else:
                        series = df_batch[ticker]  # fallback
                    series.name = ticker
                    series = series.dropna()
                    if len(series) > 0:
                        etf_data[ticker] = series
                        print(f"  ✅ {ticker}: {len(series)} rows")
                    else:
                        print(f"  ⚠️ {ticker} returned empty data, trying Stooq...")
                        series = fetch_etf_data_stooq(ticker, START_DATE, END_DATE)
                        if series is not None:
                            etf_data[ticker] = series
                        else:
                            failed_tickers.append(ticker)
                else:
                    print(f"  ⚠️ {ticker} missing in batch response, trying Stooq...")
                    series = fetch_etf_data_stooq(ticker, START_DATE, END_DATE)
                    if series is not None:
                        etf_data[ticker] = series
                    else:
                        failed_tickers.append(ticker)
        
        # Delay between batches to avoid rate limits
        if batch_start_idx + BATCH_SIZE < len(ETF_LIST):
            print(f"  ⏳ Waiting {BATCH_DELAY}s before next batch...")
            time.sleep(BATCH_DELAY)
    
    if not etf_data:
        raise RuntimeError("No ETF data could be fetched from any source. Aborting.")
    
    if failed_tickers:
        print(f"\n⚠️ Failed tickers: {failed_tickers} — continuing with {len(etf_data)} tickers.")
    
    # Combine into DataFrame
    etf_df = pd.DataFrame(etf_data)
    print(f"\n📊 ETF DataFrame shape: {etf_df.shape}")
    print(f"   Date range: {etf_df.index[0].date()} to {etf_df.index[-1].date()}")
    
    # 2. Fetch T-Bill data from FRED
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
