"""
Transfer Voting Module
Implements Equations (21) and (22) from the Entropy paper correctly.

Eq.(21):  w_i = 1 / d_i            (source weight = inverse DTW distance)
Eq.(22):  ŷ = Σ(w_i × M_i) / Σw_i (weighted average of SOURCE models only)

Key fix vs previous version:
  - Each source model predicts on TARGET ETF's features (knowledge transfer).
  - Feature column names are renamed to match the source ETF's expected prefix
    before prediction, so sklearn does not reject the input.
    e.g. TLT_MA3 → VNQ_MA3 when asking VNQ's model to interpret TLT's features.
"""

import numpy as np
import pandas as pd
import joblib
import os

from base_models import train_base_models, predict_base_models, save_base_models
from dtw_weights import compute_dtw_matrix


class TransferVotingModel:

    def __init__(self, etf_list: list, ma_window: int, artifact_path: str = "artifacts"):
        self.etf_list      = etf_list
        self.ma_window     = ma_window
        self.artifact_path = artifact_path
        self.dtw_weights   = None
        self.base_models   = {}
        self.is_fitted     = False

    # ── Fitting ───────────────────────────────────────────────────────

    def fit(self, X_train_dict, y_train_dict, train_price_df):
        os.makedirs(self.artifact_path, exist_ok=True)

        print("Computing DTW weight matrix (train window only)...")
        available = [e for e in self.etf_list if e in train_price_df.columns]

        if len(available) < 2:
            print("  Warning: <2 ETFs — using equal weights")
            n = len(self.etf_list)
            self.dtw_weights = (np.ones((n, n)) - np.eye(n)) / max(n - 1, 1)
        else:
            price_subset     = train_price_df[available].dropna()
            raw_matrix       = compute_dtw_matrix(price_subset)
            n                = len(self.etf_list)
            self.dtw_weights = np.zeros((n, n))
            for i, etf_i in enumerate(self.etf_list):
                for j, etf_j in enumerate(self.etf_list):
                    if etf_i in available and etf_j in available:
                        ii = available.index(etf_i)
                        jj = available.index(etf_j)
                        self.dtw_weights[i, j] = raw_matrix[ii, jj]

        np.save(
            os.path.join(self.artifact_path, f"dtw_matrix_MA{self.ma_window}.npy"),
            self.dtw_weights,
        )
        print("  DTW weights saved.")

        for etf in self.etf_list:
            if etf not in X_train_dict:
                print(f"  Skipping {etf}: no training data")
                continue
            print(f"\nTraining base models — {etf} (MA{self.ma_window})")
            base_models = train_base_models(
                X_train_dict[etf], y_train_dict[etf], self.artifact_path)
            self.base_models[etf] = base_models
            save_base_models(base_models, self.ma_window, etf, self.artifact_path)

        self.is_fitted = True
        print(f"\nTransferVotingModel fitted — MA{self.ma_window}, "
              f"{len(self.base_models)} ETFs")

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _rename_for_source(X: pd.DataFrame, source_etf: str) -> pd.DataFrame:
        """
        Rename TARGET ETF feature columns to SOURCE ETF prefix so sklearn
        accepts them. e.g. TLT_MA3 → VNQ_MA3 when source_etf=VNQ.
        All ETFs share identical feature structure — only the prefix differs.
        """
        if X.empty or len(X.columns) == 0:
            return X
        first_col     = X.columns[0]
        parts         = first_col.split("_", 1)
        if len(parts) < 2:
            return X
        target_prefix = parts[0]
        if target_prefix == source_etf:
            return X  # already correct
        rename_map = {
            col: col.replace(f"{target_prefix}_", f"{source_etf}_", 1)
            for col in X.columns
            if col.startswith(f"{target_prefix}_")
        }
        return X.rename(columns=rename_map)

    def _simple_voting_pred(self, etf: str, X: pd.DataFrame) -> np.ndarray:
        """Mean of all base-model predictions. X must have correct prefix for etf."""
        if etf not in self.base_models:
            return np.zeros(len(X))
        preds = predict_base_models(self.base_models[etf], X)
        return np.mean(list(preds.values()), axis=0)

    # ── Prediction ────────────────────────────────────────────────────

    def predict_single_etf(self, X, target_etf, source_feature_dict=None):
        """
        Predict for one target ETF using Transfer Voting (Eq. 22).

        Each source ETF's model predicts on TARGET's features (renamed to
        match source's expected column prefix). This is the actual knowledge
        transfer — source models interpret the target's momentum patterns.
        """
        if target_etf not in self.base_models:
            raise ValueError(f"No model for {target_etf}")

        target_idx = self.etf_list.index(target_etf)

        if self.dtw_weights is None:
            return self._simple_voting_pred(target_etf, X)

        weighted_sum = np.zeros(len(X))
        weight_total = 0.0

        for j, source_etf in enumerate(self.etf_list):
            if source_etf == target_etf:
                continue  # exclude self per Eq.22

            w = self.dtw_weights[target_idx, j]
            if w <= 0:
                continue

            # Rename target's feature columns to source's expected prefix
            X_renamed   = self._rename_for_source(X, source_etf)
            source_pred = self._simple_voting_pred(source_etf, X_renamed)

            weighted_sum += w * source_pred
            weight_total += w

        if weight_total < 1e-10:
            return self._simple_voting_pred(target_etf, X)

        return weighted_sum / weight_total

    def predict_all_etfs(self, X_dict):
        if not self.is_fitted:
            raise ValueError("Model not fitted — call fit() first.")
        predictions = {}
        for etf in self.etf_list:
            if etf in X_dict and etf in self.base_models:
                predictions[etf] = self.predict_single_etf(
                    X_dict[etf], etf, source_feature_dict=X_dict)
        return predictions

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, filename):
        joblib.dump({
            "etf_list":    self.etf_list,
            "ma_window":   self.ma_window,
            "base_models": self.base_models,
            "dtw_weights": self.dtw_weights,
            "is_fitted":   self.is_fitted,
        }, filename)
        print(f"Model saved → {filename}")

    def load(self, filename):
        state            = joblib.load(filename)
        self.etf_list    = state["etf_list"]
        self.ma_window   = state["ma_window"]
        self.base_models = state["base_models"]
        self.dtw_weights = state["dtw_weights"]
        self.is_fitted   = state["is_fitted"]
        return self


def train_transfer_voting(etf_list, ma_window, X_train_dict, y_train_dict,
                           train_price_df, artifact_path="artifacts"):
    model = TransferVotingModel(etf_list, ma_window, artifact_path)
    model.fit(X_train_dict, y_train_dict, train_price_df)
    return model
