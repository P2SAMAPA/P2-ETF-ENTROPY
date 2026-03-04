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
    latest_trading = pd.to_datetime(get_latest_trading_day())

    if last_update >= latest_trading:
        print(f"Dataset already updated till {last_update.date()}")
        return

    start_date = (last_update + timedelta(days=1)).strftime("%Y-%m-%d")

    etf_new = download_etf_data(start_date)
    tbill_new = download_tbill_data(start_date)

    raw_path = hf_hub_download("P2SAMAPA/etf-entropy-dataset", "raw_data.parquet", repo_type="dataset")
    df_old = pd.read_parquet(raw_path)

    df_new = etf_new.join(tbill_new, how="left")
    df_new = df_new.ffill()

    df = pd.concat([df_old, df_new])
    df = df[~df.index.duplicated(keep="last")]

    df.to_parquet("raw_data.parquet")

    metadata["last_data_update"] = str(latest_trading.date())
    metadata["dataset_version"] += 1

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    save_to_hf("raw_data.parquet", "raw_data.parquet")
    save_to_hf("metadata.json", "metadata.json")

    print(f"Data refreshed till {latest_trading.date()}")

if __name__ == "__main__":
    incremental_update()
