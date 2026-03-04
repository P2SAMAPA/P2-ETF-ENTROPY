"""
Base Models Module
Trains 5 ensemble models: RandomForest, XGBoost, LightGBM, AdaBoost, DecisionTree
With GridSearchCV as specified in the Entropy paper
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb
import joblib
import os


def get_base_models():
    """
    Returns dictionary of base models with their parameter grids
    Using TimeSeriesSplit for cross-validation (financial time series)
    """
    
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [5, 10, -1],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50]
            }
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(random_state=42),
            'params': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'loss': ['linear', 'square', 'exponential']
            }
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }
    }
    
    return models


def train_base_models(X_train, y_train, artifact_path='artifacts'):
    """
    Trains all base models with GridSearchCV
    Returns dictionary of trained models
    """
    os.makedirs(artifact_path, exist_ok=True)
    
    models_config = get_base_models()
    trained_models = {}
    
    # Time Series Cross-Validation (5 splits as per paper)
    tscv = TimeSeriesSplit(n_splits=5)
    
    for name, config in models_config.items():
        print(f"Training {name}...")
        
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        
        trained_models[name] = {
            'model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_
        }
        
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV score: {grid_search.best_score_:.6f}")
    
    return trained_models


def save_base_models(trained_models, ma_window, etf_name, artifact_path='artifacts'):
    """Save trained base models to disk"""
    filename = f"{artifact_path}/base_models_{etf_name}_MA{ma_window}.pkl"
    joblib.dump(trained_models, filename)
    return filename


def load_base_models(ma_window, etf_name, artifact_path='artifacts'):
    """Load trained base models from disk"""
    filename = f"{artifact_path}/base_models_{etf_name}_MA{ma_window}.pkl"
    return joblib.load(filename)


def predict_base_models(trained_models, X):
    """
    Generate predictions from all base models
    Returns array of predictions (n_samples, n_models)
    """
    predictions = {}
    for name, model_dict in trained_models.items():
        predictions[name] = model_dict['model'].predict(X)
    return predictions
