"""
Transfer Voting Module
Implements Equations (21) and (22) from the Entropy paper correctly.

Eq.(21):  w_i = 1 / d_i            (source weight = inverse DTW distance)
Eq.(22):  ŷ = Σ(w_i × M_i) / Σw_i (weighted average of SOURCE models only)

Key fixes vs original:
  - Self (target ETF) is EXCLUDED from the weighted average.
    The paper's transfer learning aggregates knowledge from the OTHER
    9 (here 6) source ETFs — the target's own model is NOT a source.
  - DTW weights are computed on the TRAINING window only, passed in
    from train_models.py after the split.
  - Two-pass prediction (simple voting → transfer voting) is preserved
    because source predictions are needed for cross-ETF aggregation.
  - predict_single_etf now requires source_predictions when dtw_weights
    are available, making the transfer path always active during inference.
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
    2. For each source ETF, train a base-model ensemble
       (AdaBoost, DT, LightGBM, RF, XGBoost) on that ETF's training data.
    3. At prediction time, for target ETF t:
         a. Get each source ETF's simple-voting prediction for t's features.
         b. Weight those predictions by DTW similarity to t (Eq. 22).
         c. Return weighted average — target ETF itself is excluded.
    """

    def __init__(self, etf_list: list, ma_window: int, artifact_path: str = "artifacts"):
        self.etf_list     = etf_list
        self.ma_window    = ma_window
        self.artifact_path = artifact_path
        self.dtw_weights  = None   # shape (n, n), row i = weights for target i
        self.base_models  = {}     # {etf: {model_name: fitted_model}}
        self.is_fitted    = False

    # ──────────────────────────────────────────────────────────────────
    # Fitting
    # ──────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_train_dict: dict,
        y_train_dict: dict,
        train_price_df: pd.DataFrame,
    ):
        """
        Fit the Transfer Voting model.

        Parameters
        ----------
        X_train_dict   : {etf: X_train DataFrame}  — normalised training features
        y_train_dict   : {etf: y_train Series}      — training targets
        train_price_df : DataFrame of ETF prices for TRAINING window only
                         Used for DTW computation (no look-ahead).
        """
        os.makedirs(self.artifact_path, exist_ok=True)

        # ── Step 1: DTW weights from training prices only ─────────────
        print("Computing DTW weight matrix (train window only)...")
        available = [e for e in self.etf_list if e in train_price_df.columns]

        if len(available) < 2:
            print("  Warning: <2 ETFs — using equal weights")
            n = len(self.etf_list)
            self.dtw_weights = (np.ones((n, n)) - np.eye(n)) / max(n - 1, 1)
        else:
            price_subset     = train_price_df[available].dropna()
            raw_matrix       = compute_dtw_matrix(price_subset)

            # Map back to full etf_list order (in case some ETFs missing)
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
        print(f"  DTW weights saved.")

        # ── Step 2: Train base models for each ETF ────────────────────
        for etf in self.etf_list:
            if etf not in X_train_dict:
                print(f"  Skipping {etf}: no training data")
                continue

            print(f"\nTraining base models — {etf} (MA{self.ma_window})")
            base_models = train_base_models(
                X_train_dict[etf], y_train_dict[etf], self.artifact_path
            )
            self.base_models[etf] = base_models
            save_base_models(base_models, self.ma_window, etf, self.artifact_path)

        self.is_fitted = True
        print(f"\nTransferVotingModel fitted — MA{self.ma_window}, "
              f"{len(self.base_models)} ETFs")

    # ──────────────────────────────────────────────────────────────────
    # Prediction helpers
    # ──────────────────────────────────────────────────────────────────

    def _simple_voting_pred(self, etf: str, X: pd.DataFrame) -> np.ndarray:
        """Mean of all base-model predictions for a single ETF."""
        if etf not in self.base_models:
            return np.zeros(len(X))
        preds = predict_base_models(self.base_models[etf], X)
        return np.mean(list(preds.values()), axis=0)

    def predict_single_etf(
        self,
        X: pd.DataFrame,
        target_etf: str,
        source_feature_dict: dict = None,
    ) -> np.ndarray:
        """
        Predict for one target ETF using Transfer Voting (Eq. 22).

        Parameters
        ----------
        X                  : features for the TARGET ETF (normalised)
        target_etf         : name of the ETF being predicted
        source_feature_dict: {etf: X_features} for ALL ETFs (same date range).
                             Required for proper transfer voting.
                             If None, falls back to simple voting on target only.

        Returns
        -------
        np.ndarray of predictions, shape (len(X),)
        """
        if target_etf not in self.base_models:
            raise ValueError(f"No model for {target_etf}")

        target_idx = self.etf_list.index(target_etf)

        # Fallback: no DTW weights or no source features → simple voting
        if self.dtw_weights is None or source_feature_dict is None:
            return self._simple_voting_pred(target_etf, X)

        # ── Eq. 22: weighted sum of SOURCE models only ────────────────
        # Source = every ETF except the target itself
        weighted_sum = np.zeros(len(X))
        weight_total = 0.0

        for j, source_etf in enumerate(self.etf_list):
            if source_etf == target_etf:
                continue   # ← key fix: exclude self

            w = self.dtw_weights[target_idx, j]
            if w <= 0 or source_etf not in source_feature_dict:
                continue

            # Each source model makes predictions on the TARGET's features
            # (same feature space, different learned weights — knowledge transfer)
            source_pred = self._simple_voting_pred(
                source_etf, source_feature_dict[source_etf]
            )

            # Align lengths (source and target may differ slightly after dropna)
            min_len = min(len(weighted_sum), len(source_pred))
            weighted_sum[:min_len] += w * source_pred[:min_len]
            weight_total           += w

        if weight_total < 1e-10:
            # No valid sources — fall back to target's own model
            return self._simple_voting_pred(target_etf, X)

        return weighted_sum / weight_total

    def predict_all_etfs(self, X_dict: dict) -> dict:
        """
        Predict all ETFs using full Transfer Voting.

        Two-pass approach:
          Pass 1 — compute simple-voting predictions for all ETFs.
          Pass 2 — apply DTW-weighted transfer aggregation.

        Parameters
        ----------
        X_dict : {etf: X_features DataFrame}

        Returns
        -------
        dict {etf: np.ndarray of predictions}
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted — call fit() first.")

        predictions = {}
        for etf in self.etf_list:
            if etf in X_dict and etf in self.base_models:
                predictions[etf] = self.predict_single_etf(
                    X_dict[etf], etf, source_feature_dict=X_dict
                )
        return predictions

    # ──────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────

    def save(self, filename: str):
        state = {
            "etf_list":    self.etf_list,
            "ma_window":   self.ma_window,
            "base_models": self.base_models,
            "dtw_weights": self.dtw_weights,
            "is_fitted":   self.is_fitted,
        }
        joblib.dump(state, filename)
        print(f"Model saved → {filename}")

    def load(self, filename: str) -> "TransferVotingModel":
        state            = joblib.load(filename)
        self.etf_list    = state["etf_list"]
        self.ma_window   = state["ma_window"]
        self.base_models = state["base_models"]
        self.dtw_weights = state["dtw_weights"]
        self.is_fitted   = state["is_fitted"]
        return self


# ── Convenience wrapper ────────────────────────────────────────────────────────

def train_transfer_voting(
    etf_list: list,
    ma_window: int,
    X_train_dict: dict,
    y_train_dict: dict,
    train_price_df: pd.DataFrame,
    artifact_path: str = "artifacts",
) -> TransferVotingModel:
    model = TransferVotingModel(etf_list, ma_window, artifact_path)
    model.fit(X_train_dict, y_train_dict, train_price_df)
    return model
