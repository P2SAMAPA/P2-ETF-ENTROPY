"""
Base Models Module — SPEED OPTIMISED
Changes vs previous:
  - Dropped AdaBoost and DecisionTree (slowest to tune, weakest signal)
  - RandomForest: n_estimators 100→50, grid 4→2 combos
  - XGBoost: n_estimators 100→50, grid 12→4 combos
  - LightGBM: n_estimators 100→50, grid 12→4 combos
  - TimeSeriesSplit n_splits: 3→2
  - Result: ~5x faster than previous version
"""
import numpy as np
import os
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb


def get_base_models():
    return {
        "RandomForest": {
            "model": RandomForestRegressor(
                random_state=42, n_jobs=-1, n_estimators=50
            ),
            "params": {
                "max_depth": [10, None],   # 2 combos only
            },
        },
        "XGBoost": {
            "model": xgb.XGBRegressor(
                random_state=42, n_jobs=-1, n_estimators=50,
                verbosity=0
            ),
            "params": {
                "max_depth":     [3, 5],       # 2×2 = 4 combos
                "learning_rate": [0.05, 0.1],
            },
        },
        "LightGBM": {
            "model": lgb.LGBMRegressor(
                random_state=42, n_jobs=-1, n_estimators=50,
                verbose=-1
            ),
            "params": {
                "max_depth":     [5, 10],      # 2×2 = 4 combos
                "learning_rate": [0.05, 0.1],
            },
        },
    }


def train_base_models(X_train, y_train, artifact_path="artifacts"):
    """
    Train base models with GridSearchCV.
    TimeSeriesSplit(n_splits=2) for maximum speed while still walk-forward.
    """
    os.makedirs(artifact_path, exist_ok=True)

    tscv          = TimeSeriesSplit(n_splits=2)
    models_config = get_base_models()
    trained_models = {}

    for name, cfg in models_config.items():
        print(f"    [{name}] fitting GridSearchCV …")
        gs = GridSearchCV(
            estimator  = cfg["model"],
            param_grid = cfg["params"],
            cv         = tscv,
            scoring    = "neg_mean_squared_error",
            n_jobs     = -1,
            verbose    = 0,
            refit      = True,
        )
        gs.fit(X_train, y_train)
        trained_models[name] = {
            "model":       gs.best_estimator_,
            "best_params": gs.best_params_,
            "best_score":  gs.best_score_,
        }
        print(f"      best_params={gs.best_params_}  "
              f"cv_mse={-gs.best_score_:.6f}")

    return trained_models


def save_base_models(trained_models, ma_window, etf_name, artifact_path="artifacts"):
    path = f"{artifact_path}/base_models_{etf_name}_MA{ma_window}.pkl"
    joblib.dump(trained_models, path)
    return path


def load_base_models(ma_window, etf_name, artifact_path="artifacts"):
    path = f"{artifact_path}/base_models_{etf_name}_MA{ma_window}.pkl"
    return joblib.load(path)


def predict_base_models(trained_models, X):
    return {name: d["model"].predict(X)
            for name, d in trained_models.items()}
