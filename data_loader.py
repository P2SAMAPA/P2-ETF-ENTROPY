import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from huggingface_hub import HfApi, hf_hub_download
import json
from datetime import timedelta
from utils import get_latest_trading_day

HF_DATASET_REPO = "P2SAMAPA/etf-entropy-dataset"

ETF_LIST = ["TLT","VNQ","GLD","SLV","VCIT","HYG","LQD","SPY","AGG"]

def download_etf_data(start_date):
    data = yf.download(ETF_LIST, start=start_date, auto_adjust=True)
    data = data["Close"]
    return data

def download_tbill_data(start_date):
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    tbill = fred.get_series("DGS3MO")
    tbill = tbill.to_frame("3MTBILL")
    tbill = tbill.loc[start_date:]
    tbill = tbill.ffill()
    return tbill

def load_metadata():
    try:
        path = hf_hub_download(HF_DATASET_REPO, "metadata.json", repo_type="dataset")
        with open(path) as f:
            return json.load(f)
    except:
        return None

def save_to_hf(file_path, repo_path):
    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=repo_path,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset"
    )

def seed_dataset():
    print("Seeding dataset from 2008...")
    start_date = "2008-01-01"

    etf = download_etf_data(start_date)
    tbill = download_tbill_data(start_date)

    df = etf.join(tbill, how="left")
    df = df.ffill()

    df.to_parquet("raw_data.parquet")

    metadata = {
        "last_data_update": str(get_latest_trading_day()),
        "last_training_date": None,
        "best_ma_window": None,
        "dataset_version": 1
    }

    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

    save_to_hf("raw_data.parquet", "raw_data.parquet")
    save_to_hf("metadata.json", "metadata.json")

    print("Seeding complete.")
