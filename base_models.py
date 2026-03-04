"""
Base Models Module - OPTIMIZED for faster training
Reduced hyperparameter grids for GitHub Actions time limits
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
    Returns dictionary of base models with REDUCED parameter grids
    for faster training in CI/CD environment
    """
    
    models = {
        'RandomForest': {
            'model': RandomForestRegressor(random_state=42, n_jobs=-1, n_estimators=100),
            'params': {
                'max_depth': [5, 10],
                'min_samples_split': [5],
            }
        },
        'XGBoost': {
            'model': xgb.XGBRegressor(random_state=42, n_jobs=-1, n_estimators=100),
            'params': {
                'max_depth': [3, 5],
                'learning_rate': [0.1],
            }
        },
        'LightGBM': {
            'model': lgb.LGBMRegressor(random_state=42, n_jobs=-1, n_estimators=100, verbose=-1),
            'params': {
                'max_depth': [5, 10],
                'learning_rate': [0.1],
            }
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(random_state=42, n_estimators=50),
            'params': {
                'learning_rate': [0.1, 1.0],
            }
        },
        'DecisionTree': {
            'model': DecisionTreeRegressor(random_state=42),
            'params': {
                'max_depth': [5, 10],
                'min_samples_split': [5],
            }
        }
    }
    
    return models


def train_base_models(X_train, y_train, artifact_path='artifacts'):
    """
    Trains all base models with reduced GridSearchCV
    """
    os.makedirs(artifact_path, exist_ok=True)
    
    models_config = get_base_models()
    trained_models = {}
    
    # Reduced CV splits for speed (3 instead of 5)
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, config in models_config.items():
        print(f"  Training {name}...")
        
        # Reduce search space further if dataset is large
        n_samples = len(X_train)
        if n_samples > 2000:
            # Use best guess parameters for large datasets (skip grid search)
            model = config['model']
            model.fit(X_train, y_train)
            trained_models[name] = {
                'model': model,
                'best_params': 'default (large dataset)',
                'best_score': 0
            }
        else:
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
        
        print(f"    Done")
    
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
    """
    predictions = {}
    for name, model_dict in trained_models.items():
        predictions[name] = model_dict['model'].predict(X)
    return predictions
