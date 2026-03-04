import json
import os
import pandas as pd
from transfer_voting import TransferVotingModel
from backtest import run_backtest
from strategy_engine import StrategyEngine
from metrics import calculate_metrics


def optimize_ma_window(etf_list, data_dict, price_df, tbill_df,
                       artifact_path="artifacts"):

    os.makedirs(artifact_path, exist_ok=True)
    results = {}

    for ma_window in [3, 5]:

        features_dict = data_dict[ma_window]["features"]
        targets_dict = data_dict[ma_window]["targets"]

        # Common dates
        common_dates = None
        for etf in etf_list:
            dates = features_dict[etf].index
            common_dates = dates if common_dates is None else common_dates.intersection(dates)

        n = len(common_dates)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        train_dates = common_dates[:train_end]
        test_dates = common_dates[val_end:]

        X_train = {etf: features_dict[etf].loc[train_dates] for etf in etf_list}
        y_train = {etf: targets_dict[etf].loc[train_dates] for etf in etf_list}

        model = TransferVotingModel(etf_list, ma_window, artifact_path)
        model.fit(X_train, y_train, price_df.loc[train_dates])
        model.save(f"{artifact_path}/transfer_voting_MA{ma_window}.pkl")

        # OOS Predictions
        predictions_dict = {}
        for etf in etf_list:
            X_test = features_dict[etf].loc[test_dates]
            preds = model.predict_single_etf(X_test, etf)
            predictions_dict[etf] = pd.Series(preds, index=test_dates)

        engine = StrategyEngine(etf_list)
        results_bt = run_backtest(
            predictions_dict,
            price_df,
            tbill_df,
            engine,
            test_dates
        )

        metrics = calculate_metrics(
            results_bt["equity_curve"]["strategy"],
            results_bt["returns"],
            results_bt["risk_free"],
            results_bt["audit_trail"]
        )

        results[ma_window] = metrics["annualized_return"]

    best_ma = max(results, key=results.get)

    with open(f"{artifact_path}/best_model.json", "w") as f:
        json.dump({
            "best_ma_window": best_ma,
            "ma3_return": results[3],
            "ma5_return": results[5]
        }, f, indent=2)

    return best_ma, results
