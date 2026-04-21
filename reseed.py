"""
reseed.py - ONE-TIME script to build complete dataset from 2008.
Uses Yahoo Finance (primary) with Stooq fallback.
**RUN THIS LOCALLY ON YOUR OWN MACHINE, NOT ON GITHUB ACTIONS.**
"""

import os
import sys
import json
import time
import random
import pandas as pd
import yfinance as yf
from fredapi import Fred
from datetime import datetime
from huggingface_hub import HfApi, CommitOperationAdd

# Attempt to import pandas_datareader for Stooq
try:
    import pandas_datareader.data as web
    STOOQ_AVAILABLE = True
except ImportError:
    STOOQ_AVAILABLE = False
    print("WARNING: pandas_datareader not installed. Stooq fallback will be disabled.")
    print("Install it with: pip install pandas-datareader")

# Import config to get all tickers
try:
    from config import ALL_TICKERS
except ImportError:
    print("ERROR: config.py not found. Please create it with ALL_TICKERS defined.")
    sys.exit(1)

# --- Configuration ---
HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"
ETF_LIST = ALL_TICKERS
START_DATE = "2008-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
BASE_DELAY = 5.0                     # Seconds between tickers (increase if still rate limited)
MAX_RETRIES = 3                      # Retry attempts per ticker


def fetch_etf_stooq(ticker, start, end):
    """Fetch from Stooq (no rate limits, but sometimes missing tickers)."""
    if not STOOQ_AVAILABLE:
        return None
    try:
        # Stooq format for US ETFs: 'spy.us'
        stooq_ticker = f"{ticker.lower()}.us"
        df = web.DataReader(stooq_ticker, 'stooq', start, end)
        if df.empty:
            return None
        df = df.sort_index()
        series = df['Close']
        series.name = ticker
        print(f"  ✅ {ticker} via Stooq: {len(series)} rows")
        return series
    except Exception as e:
        print(f"  ⚠️ Stooq fallback failed for {ticker}: {e}")
        return None


def fetch_etf_data_yf(ticker, start, end):
    """Fetch Close price from Yahoo Finance with exponential backoff."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                threads=False,
            )
            if df.empty:
                raise ValueError(f"No data for {ticker}")

            # Extract Close price
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
            df.index = pd.to_datetime(df.index).tz_localize(None)

            print(f"  ✅ {ticker} via Yahoo: {len(df)} rows")
            return df

        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(k in err_str for k in ["rate limit", "too many requests", "429", "ratelimit"])

            if is_rate_limit and attempt < MAX_RETRIES:
                wait = 120 * (attempt + 1) + random.randint(30, 60)
                print(f"  ⚠️ Rate limited on {ticker} (attempt {attempt+1}). Waiting {wait}s...")
                time.sleep(wait)
            elif attempt < MAX_RETRIES:
                # Other errors: shorter wait then retry
                wait = 10 * (attempt + 1)
                print(f"  ⚠️ Error on {ticker}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  ❌ Yahoo failed for {ticker} after {MAX_RETRIES} attempts: {e}")
                return None
    return None


def fetch_etf_with_fallback(ticker, start, end):
    """Try Yahoo first, then Stooq."""
    series = fetch_etf_data_yf(ticker, start, end)
    if series is not None:
        return series

    if STOOQ_AVAILABLE:
        print(f"  🔄 Falling back to Stooq for {ticker}...")
        series = fetch_etf_stooq(ticker, start, end)
        if series is not None:
            return series

    print(f"  ❌ All sources failed for {ticker}")
    return None


def main():
    print("=" * 60)
    print("FULL RESEED FROM 2008-01-01")
    print("⚠️  WARNING: This script MUST be run on your LOCAL machine.")
    print("   GitHub Actions IPs are aggressively rate‑limited by Yahoo Finance.")
    print("=" * 60)
    print(f"Tickers: {ETF_LIST}")
    print(f"Base delay: {BASE_DELAY}s | Stooq fallback: {STOOQ_AVAILABLE}")
    print("=" * 60)

    # 1. Fetch ETF data
    print(f"\n📥 Downloading ETFs ({START_DATE} to {END_DATE})...")
    etf_data = {}
    failed_tickers = []

    for idx, ticker in enumerate(ETF_LIST):
        print(f"\n--- {ticker} ({idx+1}/{len(ETF_LIST)}) ---")
        series = fetch_etf_with_fallback(ticker, START_DATE, END_DATE)
        if series is not None:
            etf_data[ticker] = series
        else:
            failed_tickers.append(ticker)

        # Polite delay
        delay = BASE_DELAY + random.uniform(-1.0, 2.0)
        print(f"  ⏳ Waiting {delay:.1f}s before next ticker...")
        time.sleep(delay)

    if not etf_data:
        raise RuntimeError("No ETF data could be fetched. Aborting.")

    if failed_tickers:
        print(f"\n⚠️ Failed tickers after all attempts: {failed_tickers}")
        print(f"   Continuing with {len(etf_data)} tickers.")

    # Combine into DataFrame
    etf_df = pd.DataFrame(etf_data)
    print(f"\n📊 ETF DataFrame shape: {etf_df.shape}")
    print(f"   Date range: {etf_df.index[0].date()} to {etf_df.index[-1].date()}")

    # 2. Fetch T-Bill data from FRED
    print(f"\n📥 Downloading 3-Month T-Bill from FRED...")
    fred_api_key = os.getenv("FRED_API_KEY")
    if not fred_api_key:
        print("⚠️ FRED_API_KEY not set. Skipping T-Bill data.")
        tbill_df = pd.DataFrame(index=etf_df.index)
        tbill_df["3MTBILL"] = 0.0
    else:
        fred = Fred(api_key=fred_api_key)
        tbill = fred.get_series("DGS3MO", observation_start=START_DATE, observation_end=END_DATE)
        tbill_df = tbill.to_frame("3MTBILL").ffill()
        print(f"   T-Bill rows: {len(tbill_df)}")

    # 3. Merge datasets
    full_df = etf_df.join(tbill_df, how='left').ffill().bfill()
    full_df.index = pd.to_datetime(full_df.index).tz_localize(None)

    print(f"\n✅ Merged dataset shape: {full_df.shape}")

    # 4. Save locally
    full_df.to_parquet("raw_data.parquet")
    file_size = os.path.getsize("raw_data.parquet")
    print(f"\n💾 Saved raw_data.parquet ({file_size:,} bytes)")

    # 5. Metadata
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

    # 6. Upload to Hugging Face (optional)
    print(f"\n📤 Uploading to Hugging Face: {HF_DATASET_REPO}")
    token = os.getenv("HF_TOKEN")
    if token:
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
                operations=[CommitOperationAdd(path_in_repo=repo_file, path_or_fileobj=content)],
            )
            print(f"  ✅ Uploaded {repo_file}")
    else:
        print("  ⚠️ HF_TOKEN not set. Files saved locally only.")

    print("\n" + "=" * 60)
    print(f"🎉 RESEED COMPLETE - {len(full_df)} rows, {len(full_df.columns)} columns")
    print("=" * 60)


if __name__ == "__main__":
    main()
