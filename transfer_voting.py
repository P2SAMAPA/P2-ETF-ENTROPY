"""
Transfer Voting Module
Implements Equation (22) from the Entropy paper:
Final Prediction = sum(w_i * M_i) / sum(w_i)
Where w_i = 1 / DTW_distance
"""

import numpy as np
import pandas as pd
import joblib
import os
from base_models import train_base_models, predict_base_models, save_base_models, load_base_models
from dtw_weights import compute_dtw_matrix


class TransferVotingModel:
    """
    Transfer Voting Ensemble as described in the Entropy paper
    Combines base model predictions with DTW-based transfer learning weights
    """
    
    def __init__(self, etf_list, ma_window, artifact_path='artifacts'):
        self.etf_list = etf_list
        self.ma_window = ma_window
        self.artifact_path = artifact_path
        self.dtw_weights = None
        self.base_models = {}
        self.is_fitted = False
        
    def compute_dtw_weights(self, price_df, target_etf):
        """
        Compute DTW weights for transfer learning
        Returns weights for each source ETF relative to target
        """
        # Compute full DTW matrix
        dtw_matrix = compute_dtw_matrix(price_df)
        
        # Get index of target ETF
        target_idx = self.etf_list.index(target_etf)
        
        # Extract weights for this target (row from matrix)
        # Weight = 1 / distance, already computed in matrix
        weights = dtw_matrix[target_idx, :]
        
        # Normalize weights to sum to 1 (softmax style)
        weights = weights / weights.sum()
        
        return weights
    
    def fit(self, X_dict, y_dict, price_df):
        """
        Fit Transfer Voting model for all ETFs
        
        Parameters:
        -----------
        X_dict : dict
            Dictionary of features for each ETF {etf_name: X_array}
        y_dict : dict  
            Dictionary of targets for each ETF {etf_name: y_array}
        price_df : pd.DataFrame
            Price data for DTW computation
        """
        os.makedirs(self.artifact_path, exist_ok=True)
        
        # Train base models for each ETF and store
        for etf in self.etf_list:
            print(f"\n{'='*50}")
            print(f"Training Transfer Voting for {etf} (MA{self.ma_window})")
            print(f"{'='*50}")
            
            # Train base models on this ETF's data
            base_models = train_base_models(X_dict[etf], y_dict[etf], self.artifact_path)
            self.base_models[etf] = base_models
            
            # Save base models
            save_base_models(base_models, self.ma_window, etf, self.artifact_path)
        
        # Compute and store DTW matrix
        self.dtw_weights = compute_dtw_matrix(price_df)
        
        # Save DTW matrix
        np.save(f"{self.artifact_path}/dtw_matrix.npy", self.dtw_weights)
        
        self.is_fitted = True
        print(f"\nTransfer Voting training complete for MA{self.ma_window}")
        
    def predict_single_etf(self, X, target_etf, source_predictions=None):
        """
        Predict for single ETF using Transfer Voting
        
        Equation (22): Final Prediction = sum(w_i * M_i) / sum(w_i)
        
        If source_predictions provided (from other ETFs), use transfer learning.
        Otherwise, use only self-predictions (regular voting).
        """
        target_idx = self.etf_list.index(target_etf)
        
        # Get base model predictions for target ETF
        target_base_preds = predict_base_models(self.base_models[target_etf], X)
        
        # Simple voting (mean of base models) for target
        target_voting_pred = np.mean(list(target_base_preds.values()), axis=0)
        
        if source_predictions is None or not self.is_fitted:
            return target_voting_pred
        
        # Transfer Voting: Weight predictions by DTW similarity
        weights = self.dtw_weights[target_idx, :]
        
        # Collect predictions from all ETFs (including self)
        all_predictions = [target_voting_pred]
        all_weights = [weights[target_idx]]  # Self weight
        
        for i, source_etf in enumerate(self.etf_list):
            if source_etf != target_etf and source_etf in source_predictions:
                all_predictions.append(source_predictions[source_etf])
                all_weights.append(weights[i])
        
        # Weighted average: sum(w_i * pred_i) / sum(w_i)
        all_predictions = np.array(all_predictions)
        all_weights = np.array(all_weights)
        
        # Normalize weights
        all_weights = all_weights / all_weights.sum()
        
        # Weighted prediction
        final_prediction = np.average(all_predictions, axis=0, weights=all_weights)
        
        return final_prediction
    
    def predict_all_etfs(self, X_dict):
        """
        Predict for all ETFs using full Transfer Voting
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        
        predictions = {}
        
        # First pass: Get simple voting predictions for all ETFs
        simple_preds = {}
        for etf in self.etf_list:
            if etf in X_dict:
                base_preds = predict_base_models(self.base_models[etf], X_dict[etf])
                simple_preds[etf] = np.mean(list(base_preds.values()), axis=0)
        
        # Second pass: Apply transfer voting weights
        for etf in self.etf_list:
            if etf in X_dict:
                predictions[etf] = self.predict_single_etf(X_dict[etf], etf, simple_preds)
        
        return predictions
    
    def save(self, filename):
        """Save model state"""
        state = {
            'etf_list': self.etf_list,
            'ma_window': self.ma_window,
            'base_models': self.base_models,
            'dtw_weights': self.dtw_weights,
            'is_fitted': self.is_fitted
        }
        joblib.dump(state, filename)
    
    def load(self, filename):
        """Load model state"""
        state = joblib.load(filename)
        self.etf_list = state['etf_list']
        self.ma_window = state['ma_window']
        self.base_models = state['base_models']
        self.dtw_weights = state['dtw_weights']
        self.is_fitted = state['is_fitted']
        return self


def train_transfer_voting(etf_list, ma_window, X_dict, y_dict, price_df, artifact_path='artifacts'):
    """
    Convenience function to train Transfer Voting model
    """
    model = TransferVotingModel(etf_list, ma_window, artifact_path)
    model.fit(X_dict, y_dict, price_df)
    return model
