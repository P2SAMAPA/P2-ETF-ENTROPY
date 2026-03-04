"""
Transfer Voting Module - FIXED
Implements Equation (22) from the Entropy paper
"""

import numpy as np
import pandas as pd
import joblib
import os
from base_models import train_base_models, predict_base_models, save_base_models
from dtw_weights import compute_dtw_matrix


class TransferVotingModel:
    """
    Transfer Voting Ensemble as described in the Entropy paper
    """
    
    def __init__(self, etf_list, ma_window, artifact_path='artifacts'):
        self.etf_list = etf_list
        self.ma_window = ma_window
        self.artifact_path = artifact_path
        self.dtw_weights = None
        self.base_models = {}
        self.is_fitted = False
        
    def compute_dtw_weights(self, price_df):
        """
        Compute DTW weights for transfer learning
        """
        # Compute full DTW matrix
        dtw_matrix = compute_dtw_matrix(price_df)
        
        # Store full matrix
        self.dtw_weights = dtw_matrix
        
        # Save DTW matrix
        np.save(f"{self.artifact_path}/dtw_matrix_MA{self.ma_window}.npy", dtw_matrix)
        
        return dtw_matrix
    
    def fit(self, X_dict, y_dict, price_df):
        """
        Fit Transfer Voting model for all ETFs
        
        Parameters:
        -----------
        X_dict : dict
            Dictionary of features for each ETF {etf_name: X_dataframe}
        y_dict : dict  
            Dictionary of targets for each ETF {etf_name: y_series}
        price_df : pd.DataFrame
            Price data for DTW computation (columns = ETFs)
        """
        os.makedirs(self.artifact_path, exist_ok=True)
        
        # Compute DTW weights first (before training, using price data)
        print(f"Computing DTW weights...")
        
        # Ensure price_df has the right ETFs as columns
        available_etfs = [etf for etf in self.etf_list if etf in price_df.columns]
        
        if len(available_etfs) < 2:
            print("Warning: Not enough ETFs for DTW, using simple voting")
            self.dtw_weights = np.eye(len(self.etf_list))  # Identity matrix
        else:
            price_subset = price_df[available_etfs].dropna()
            self.compute_dtw_weights(price_subset)
        
        # Train base models for each ETF
        for etf in self.etf_list:
            if etf not in X_dict:
                print(f"  Skipping {etf}: no data")
                continue
                
            print(f"\nTraining Transfer Voting for {etf} (MA{self.ma_window})")
            print("="*50)
            
            # Train base models on this ETF's data
            base_models = train_base_models(X_dict[etf], y_dict[etf], self.artifact_path)
            self.base_models[etf] = base_models
            
            # Save base models
            save_base_models(base_models, self.ma_window, etf, self.artifact_path)
        
        self.is_fitted = True
        print(f"\nTransfer Voting training complete for MA{self.ma_window}")
        
    def predict_single_etf(self, X, target_etf, source_predictions=None):
        """
        Predict for single ETF using Transfer Voting
        """
        if target_etf not in self.base_models:
            raise ValueError(f"No model trained for {target_etf}")
        
        target_idx = self.etf_list.index(target_etf)
        
        # Get base model predictions for target ETF
        target_base_preds = predict_base_models(self.base_models[target_etf], X)
        
        # Simple voting (mean of base models) for target
        target_voting_pred = np.mean(list(target_base_preds.values()), axis=0)
        
        # If no DTW weights or only self, return simple voting
        if self.dtw_weights is None or source_predictions is None:
            return target_voting_pred
        
        # Transfer Voting: Weight predictions by DTW similarity
        weights = self.dtw_weights[target_idx, :]
        
        # Collect predictions from all ETFs (including self)
        all_predictions = [target_voting_pred]
        all_weights = [weights[target_idx]]
        
        for i, source_etf in enumerate(self.etf_list):
            if source_etf != target_etf and source_etf in source_predictions:
                all_predictions.append(source_predictions[source_etf])
                all_weights.append(weights[i])
        
        # Weighted average
        all_predictions = np.array(all_predictions)
        all_weights = np.array(all_weights)
        all_weights = all_weights / all_weights.sum()
        
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
            if etf in X_dict and etf in self.base_models:
                base_preds = predict_base_models(self.base_models[etf], X_dict[etf])
                simple_preds[etf] = np.mean(list(base_preds.values()), axis=0)
        
        # Second pass: Apply transfer voting weights
        for etf in self.etf_list:
            if etf in X_dict and etf in self.base_models:
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
