import time
import pandas as pd
from datetime import timedelta
from data_loader import (
    load_metadata,
    download_etf_data,
    download_tbill_data,
    save_to_hf
)
from huggingface_hub import hf_hub_download
import json
import os
from utils import get_latest_trading_day


def incremental_update():
    metadata = load_metadata()
    if metadata is None:
        print("No dataset found. Run seed first.")
        return

    last_update = pd.to_datetime(metadata["last_data_update"])

    # Use today as the target — let yfinance return whatever is available
    # (markets may be mid-session; yfinance returns completed bars only)
    import pytz
    eastern  = pytz.timezone("US/Eastern")
    today    = pd.Timestamp(pd.Timestamp.now(tz=eastern).date())
    start_date = (last_update + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date   = today.strftime("%Y-%m-%d")

    # ── Guard: start is after today — nothing to fetch ────────────────────────
    if start_date > end_date:
        print(f"Dataset already up to date ({last_update.date()}). Nothing to do.")
        return

    # ── Download with retry for yfinance rate limits ──────────────────────────
    etf_new   = _download_with_retry(download_etf_data,   start_date, end_date)
    tbill_new = _download_with_retry(download_tbill_data, start_date, end_date)

    if etf_new is None or etf_new.empty:
        print("No new ETF data returned — skipping update.")
        return

    # ── Merge with existing dataset ───────────────────────────────────────────
    raw_path = hf_hub_download(
        "P2SAMAPA/etf-entropy-dataset", "raw_data.parquet", repo_type="dataset"
    )
    df_old = pd.read_parquet(raw_path)

    df_new = etf_new.join(tbill_new, how="left").ffill()
    df     = pd.concat([df_old, df_new])
    df     = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_parquet("raw_data.parquet")

    # Use the last date actually present in the merged df
    metadata["last_data_update"] = str(df.index[-1].date())
    metadata["dataset_version"]  = metadata.get("dataset_version", 1) + 1

    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    save_to_hf("raw_data.parquet", "raw_data.parquet")
    save_to_hf("metadata.json",    "metadata.json")

    print(f"Data refreshed: {last_update.date()} → {end_date}")


def _download_with_retry(fn, start_date, end_date, retries=4, base_wait=15):
    """
    Call fn(start_date, end_date) with exponential backoff on rate-limit errors.
    Covers both yfinance YFRateLimitError and generic HTTP 429.
    """
    for attempt in range(retries):
        try:
            return fn(start_date, end_date)
        except Exception as e:
            err = str(e).lower()
            is_rate_limit = any(k in err for k in
                                ["rate limit", "too many requests", "429", "ratelimit"])
            if is_rate_limit and attempt < retries - 1:
                wait = base_wait * (2 ** attempt)   # 15s, 30s, 60s
                print(f"  Rate limited — waiting {wait}s before retry "
                      f"({attempt+1}/{retries-1})…")
                time.sleep(wait)
            else:
                print(f"  Download failed after {attempt+1} attempt(s): {e}")
                raise
    return None


if __name__ == "__main__":
    incremental_update()
