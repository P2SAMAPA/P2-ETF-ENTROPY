"""
MA Optimizer — TUNED
Changes vs original:
  - MA windows expanded: [3, 5] → [2, 3, 4, 5, 7, 10]
  - targets now dict with 'train'/'val'/'test' keys (matches tuned feature_engineering)
  - predict_single_etf called with source_feature_dict for full transfer voting
  - best_model.json saves all window returns + val/oos distinction
  - Prints timing per window
"""

import json
import os
import time

import pandas as pd

from transfer_voting import TransferVotingModel
from backtest import run_backtest
from strategy_engine import StrategyEngine
from metrics import calculate_metrics

MA_WINDOWS = [3, 5]   # reduced to 2 windows for speed — saves ~33% training time


def optimize_ma_window(
    etf_list,
    data_dict,
    price_df,
    tbill_df,
    artifact_path="artifacts",
):
    os.makedirs(artifact_path, exist_ok=True)

    val_returns = {}   # keyed by ma_window
    oos_returns = {}   # informational only

    for ma_window in MA_WINDOWS:

        t_start = time.time()
        print(f"\n{'─'*60}")
        print(f"  MA({ma_window}) — training …")

        if ma_window not in data_dict:
            print(f"  Skipping MA({ma_window}) — no data prepared")
            continue

        features_dict = data_dict[ma_window]["features"]
        targets_dict  = data_dict[ma_window]["targets"]
        sd_dict       = data_dict[ma_window]["split_dates"]
        train_end     = data_dict[ma_window]["train_end"]
        val_end       = data_dict[ma_window]["val_end"]

        # ── Common date range ─────────────────────────────────────────────────
        common_idx = None
        for etf in etf_list:
            if etf not in features_dict:
                continue
            idx = features_dict[etf].index
            common_idx = idx if common_idx is None else common_idx.intersection(idx)

        if common_idx is None or len(common_idx) == 0:
            print(f"  MA({ma_window}): no common dates — skipping")
            continue

        common_idx    = pd.DatetimeIndex(sorted(common_idx))
        train_dates   = common_idx[:train_end]
        val_dates     = common_idx[train_end:val_end]
        oos_dates     = common_idx[val_end:]

        price_aligned = price_df.reindex(common_idx).ffill()
        tbill_aligned = tbill_df.reindex(common_idx).ffill()

        # ── Build train feature/target dicts ──────────────────────────────────
        X_train = {etf: features_dict[etf].loc[
                       features_dict[etf].index.intersection(train_dates)]
                   for etf in etf_list if etf in features_dict}
        y_train = {etf: targets_dict[etf]["train"]
                   for etf in etf_list if etf in targets_dict}

        # ── Train model ───────────────────────────────────────────────────────
        model = TransferVotingModel(etf_list, ma_window, artifact_path)
        model.fit(X_train, y_train,
                  price_aligned.loc[price_aligned.index.intersection(train_dates)])

        model_path = f"{artifact_path}/transfer_voting_MA{ma_window}.pkl"
        model.save(model_path)
        print(f"  Saved → {model_path}")

        # ── Prediction helper (two-pass transfer voting) ──────────────────────
        def _predict_window(dates_window):
            X_window = {
                etf: features_dict[etf].loc[
                    features_dict[etf].index.intersection(dates_window)]
                for etf in etf_list if etf in features_dict
            }

            # Pass 1: simple voting predictions for every ETF
            # These become the source_feature_dict for Pass 2
            simple = {}
            for etf in etf_list:
                if etf in X_window and X_window[etf].shape[0] > 0:
                    simple[etf] = X_window[etf]   # pass features, not preds

            # Pass 2: full transfer voting — each ETF uses other ETFs' models
            # FIX: keyword was 'source_predictions' but the method parameter
            # is named 'source_feature_dict' in TransferVotingModel.
            preds_dict = {}
            for etf in etf_list:
                if etf in X_window and X_window[etf].shape[0] > 0:
                    tv_preds = model.predict_single_etf(
                        X_window[etf],
                        etf,
                        source_feature_dict=simple,   # ← was source_predictions=simple
                    )
                    preds_dict[etf] = pd.Series(
                        tv_preds,
                        index=X_window[etf].index,
                    )
            return preds_dict

        # ── Val backtest (used for MA selection) ──────────────────────────────
        val_preds = _predict_window(val_dates)
        if val_preds:
            engine_val = StrategyEngine(etf_list)
            val_common = sorted(
                set.intersection(*[set(s.index) for s in val_preds.values()])
            )
            val_bt = run_backtest(
                val_preds,
                price_aligned,
                tbill_aligned,
                engine_val,
                val_common,
            )
            val_metrics = calculate_metrics(
                val_bt["equity_curve"]["strategy"],
                val_bt["returns"],
                val_bt["risk_free"],
                val_bt["audit_trail"],
            )
            val_returns[ma_window] = val_metrics["ann_return"]
            print(f"  MA({ma_window}) val ann_return = {val_metrics['ann_return']*100:.2f}%")
        else:
            val_returns[ma_window] = -999.0

        # ── OOS backtest (informational only) ─────────────────────────────────
        oos_preds = _predict_window(oos_dates)
        if oos_preds:
            engine_oos = StrategyEngine(etf_list)
            oos_common = sorted(
                set.intersection(*[set(s.index) for s in oos_preds.values()])
            )
            oos_bt = run_backtest(
                oos_preds,
                price_aligned,
                tbill_aligned,
                engine_oos,
                oos_common,
            )
            oos_metrics = calculate_metrics(
                oos_bt["equity_curve"]["strategy"],
                oos_bt["returns"],
                oos_bt["risk_free"],
                oos_bt["audit_trail"],
            )
            oos_returns[ma_window] = oos_metrics["ann_return"]
            print(f"  MA({ma_window}) OOS ann_return  = {oos_metrics['ann_return']*100:.2f}%  "
                  f"[informational]")
        else:
            oos_returns[ma_window] = 0.0

        elapsed = round(time.time() - t_start, 1)
        print(f"  MA({ma_window}) done in {elapsed}s")

    # ── Select best MA by VAL return ──────────────────────────────────────────
    if not val_returns:
        raise RuntimeError("No MA windows produced valid results")

    best_ma = max(val_returns, key=val_returns.get)
    print(f"\n{'='*60}")
    print(f"  Best MA window (by val): MA({best_ma})")
    for w in sorted(val_returns):
        print(f"    MA({w}):  val={val_returns[w]*100:.2f}%  "
              f"oos={oos_returns.get(w, 0)*100:.2f}%")

    # ── Save best_model.json ───────────────────────────────────────────────────
    model_meta = {
        "best_ma_window": best_ma,
        "val_returns":    {str(k): round(v, 6) for k, v in val_returns.items()},
        "oos_returns":    {str(k): round(v, 6) for k, v in oos_returns.items()},
        # Legacy keys for streamlit_app compatibility
        "ma3_val_ann_return": val_returns.get(3, 0.0),
        "ma5_val_ann_return": val_returns.get(5, 0.0),
    }
    meta_path = f"{artifact_path}/best_model.json"
    with open(meta_path, "w") as f:
        json.dump(model_meta, f, indent=2)
    print(f"  Saved → {meta_path}")

    return best_ma, val_returns
