"""
Transfer Voting Module
Implements Equations (21) and (22) from the Entropy paper correctly.

Eq.(21):  w_i = 1 / d_i            (source weight = inverse DTW distance)
Eq.(22):  ŷ = Σ(w_i × M_i) / Σw_i (weighted average of SOURCE models only)

Key fix vs previous version:
  - Each source model now predicts on the TARGET ETF's features, not its own.
    This is the actual knowledge transfer — "what does TLT's model say about
    GLD's momentum patterns?" Previously every source was predicting its own
    MA diff, causing high-MA-diff ETFs (TLT) to dominate every vote regardless
    of the target's actual signal.
"""

import numpy as np
import pandas as pd
import joblib
import os

from base_models import train_base_models, predict_base_models, save_base_models
from dtw_weights import compute_dtw_matrix


class TransferVotingModel:
    """
    Transfer Voting Ensemble following the Entropy paper.

    Training flow
    -------------
    1. Compute DTW weight matrix on TRAINING prices only.
    2. For each ETF, train a base-model ensemble on that ETF's training data.
    3. At prediction time, for target ETF t:
         a. Run each SOURCE ETF's model on TARGET t's features (KEY FIX).
         b. Weight those predictions by DTW similarity to t (Eq. 22).
         c. Return weighted average — target ETF itself is excluded.
    """

    def __init__(self, etf_list: list, ma_window: int, artifact_path: str = "artifacts"):
        self.etf_list      = etf_list
        self.ma_window     = ma_window
        self.artifact_path = artifact_path
        self.dtw_weights   = None
        self.base_models   = {}
        self.is_fitted     = False

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

    def _simple_voting_pred(self, etf: str, X: pd.DataFrame) -> np.ndarray:
        """Mean of all base-model predictions for a single ETF."""
        if etf not in self.base_models:
            return np.zeros(len(X))
        preds = predict_base_models(self.base_models[etf], X)
        return np.mean(list(preds.values()), axis=0)

    def predict_single_etf(self, X, target_etf, source_feature_dict=None):
        """
        Predict for one target ETF using Transfer Voting (Eq. 22).

        KEY FIX: each source ETF's model predicts on TARGET's features X,
        not on the source's own features. This is the actual knowledge
        transfer — source models interpret target momentum patterns.

        source_feature_dict is kept as a parameter for API compatibility
        but is no longer used for predictions (only for availability check).
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

            # KEY FIX: source model applied to TARGET's features X
            # (was: source_feature_dict[source_etf] — source's own features)
            source_pred   = self._simple_voting_pred(source_etf, X)
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
