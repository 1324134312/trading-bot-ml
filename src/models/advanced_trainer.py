"""
Advanced Model Training with Ensemble, Tuning, and Validation
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Optional, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
import optuna
from datetime import datetime

class AdvancedTrainer:
    """Advanced training with ensemble, tuning, and validation"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.ensemble = None
        self.is_trained = False
        
    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[List] = None,
        target_col: str = 'Target_Direction',
        test_size: float = 0.2
    ) -> Tuple:
        """Prepare data for training with proper time series split"""
        
        # Remove missing values
        data_clean = data.dropna().sort_values('Date')
        
        # Select features
        if feature_cols is None:
            exclude_cols = ['Date', 'Ticker', 'FetchDate', 'Dividends', 
                          'Stock Splits', 'Target', 'Target_Direction']
            feature_cols = [col for col in data_clean.columns 
                          if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        # Extract features and target
        X = data_clean[feature_cols].values
        y = data_clean[target_col].values
        
        # Time series split (no shuffling to preserve temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Time Series Data Split:")
        print(f"   Training: {X_train.shape[0]} samples (up to {data_clean.iloc[split_idx-1]['Date']})")
        print(f"   Test:     {X_test.shape[0]} samples (from {data_clean.iloc[split_idx]['Date']})")
        print(f"   Features: {len(feature_cols)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        use_class_balancing: bool = True,
        tune_hyperparameters: bool = True
    ):
        """Train ensemble of multiple models"""
        
        print("\n🤖 Training Advanced Ensemble Model...")
        print("   Models: XGBoost, LightGBM, Random Forest")
        print("   Method: Voting Classifier (Hard Voting)")
        
        if use_class_balancing:
            print("   Using SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        if tune_hyperparameters:
            print("   Tuning hyperparameters with Optuna...")
            
            # Tune XGBoost
            print("     → Tuning XGBoost...")
            xgb_params = self._tune_xgboost(X_train_balanced, y_train_balanced, X_val, y_val)
            
            # Tune LightGBM
            print("     → Tuning LightGBM...")
            lgb_params = self._tune_lightgbm(X_train_balanced, y_train_balanced, X_val, y_val)
            
            # Tune Random Forest
            print("     → Tuning Random Forest...")
            rf_params = self._tune_random_forest(X_train_balanced, y_train_balanced, X_val, y_val)
        else:
            xgb_params = {}
            lgb_params = {}
            rf_params = {}
        
        # Create individual models
        print("\n   Creating models...")
        
        xgb_model = xgb.XGBClassifier(**xgb_params, random_state=42)
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=42, verbose=-1)
        rf_model = RandomForestClassifier(**rf_params, random_state=42)
        
        # Train ensemble
        self.ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('lgb', lgb_model),
                ('rf', rf_model)
            ],
            voting='hard'
        )
        
        print("   Training ensemble...")
        self.ensemble.fit(X_train_balanced, y_train_balanced)
        
        # Store individual models
        self.models = {
            'xgb': self.ensemble.named_estimators_['xgb'],
            'lgb': self.ensemble.named_estimators_['lgb'],
            'rf': self.ensemble.named_estimators_['rf']
        }
        
        self.is_trained = True
        
        # Validate
        val_pred = self.ensemble.predict(X_val)
        accuracy = accuracy_score(y_val, val_pred)
        print(f"\n✅ Validation Accuracy: {accuracy:.4f}")
        
    def _tune_xgboost(self, X_train, y_train, X_val, y_val) -> dict:
        """Tune XGBoost hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'scale_pos_weight': 0.72,  # Handle class imbalance
                'random_state': 42,
                'eval_metric': 'logloss',
                'early_stopping_rounds': 10
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            y_pred = model.predict(X_val)
            return 1.0 - accuracy_score(y_val, y_pred)  # Minimize error rate
        
        study = optuna.create_study(direction='minimize', study_name='xgb_tuning')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        return study.best_params
    
    def _tune_lightgbm(self, X_train, y_train, X_val, y_val) -> dict:
        """Tune LightGBM hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
                'random_state': 42
            }
            
            model = lgb.LGBMClassifier(**params, verbose=-1)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', study_name='lgb_tuning')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        return study.best_params
    
    def _tune_random_forest(self, X_train, y_train, X_val, y_val) -> dict:
        """Tune Random Forest hyperparameters using Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'random_state': 42
            }
            
            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)
        
        study = optuna.create_study(direction='maximize', study_name='rf_tuning')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        
        return study.best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using ensemble"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        return self.ensemble.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities (soft voting)"""
        if not self.is_trained:
            raise ValueError("Model not trained!")
        
        # Average probabilities from all models
        xgb_proba = self.models['xgb'].predict_proba(X)
        lgb_proba = self.models['lgb'].predict_proba(X)
        rf_proba = self.models['rf'].predict_proba(X)
        
        # Average probabilities
        avg_proba = (xgb_proba + lgb_proba + rf_proba) / 3
        return avg_proba
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str = "Test") -> dict:
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        print(f"\n📊 {set_name} Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get combined feature importance from all models"""
        
        importance_scores = np.zeros(len(self.feature_names))
        
        # Get importance from each model
        if hasattr(self.models['xgb'], 'feature_importances_'):
            importance_scores += self.models['xgb'].feature_importances_
        if hasattr(self.models['lgb'], 'feature_importances_'):
            importance_scores += self.models['lgb'].feature_importances_
        if hasattr(self.models['rf'], 'feature_importances_'):
            importance_scores += self.models['rf'].feature_importances_
        
        # Average importance
        importance_scores = importance_scores / 3
        
        # Create DataFrame
        feat_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        return feat_df.head(top_n)
    
    def save_model(self, filepath: str):
        """Save trained ensemble model"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'ensemble': self.ensemble,
            'models': self.models,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'trained_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {filepath}")

