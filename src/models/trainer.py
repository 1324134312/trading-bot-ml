"""
Model Training Module
Trains ML models for price prediction
"""
import pandas as pd
import numpy as np
import pickle
import os
from typing import Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import xgboost as xgb
from datetime import datetime

class ModelTrainer:
    """Train and evaluate ML models for trading"""
    
    def __init__(self, model_type: str = 'xgb'):
        """
        Initialize trainer
        
        Args:
            model_type: Type of model ('xgb', 'rf', 'lgb')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_data(
        self,
        data: pd.DataFrame,
        feature_cols: Optional[list] = None,
        target_col: str = 'Target_Direction',
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Prepare data for training
        
        Args:
            data: DataFrame with features and target
            feature_cols: List of feature columns to use
            target_col: Name of target column
            test_size: Size of test set (0-1)
            val_size: Size of validation set (0-1) of remaining data
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        # Remove rows with missing values
        data_clean = data.dropna()
        
        # Select features
        if feature_cols is None:
            # Auto-select all feature columns (exclude metadata and target)
            exclude_cols = ['Date', 'Ticker', 'FetchDate', 'Dividends', 
                          'Stock Splits', 'Target', 'Target_Direction']
            feature_cols = [col for col in data_clean.columns 
                          if col not in exclude_cols]
        
        self.feature_names = feature_cols
        
        # Extract features and target
        X = data_clean[feature_cols].values
        y = data_clean[target_col].values
        
        # Train/Val/Test split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), random_state=42, stratify=y
        )
        
        val_ratio = val_size / (test_size + val_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_ratio), random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"📊 Data split:")
        print(f"   Training:   {X_train.shape[0]} samples")
        print(f"   Validation: {X_val.shape[0]} samples")
        print(f"   Test:       {X_test.shape[0]} samples")
        print(f"   Features:   {len(feature_cols)}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **model_params
    ) -> dict:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            **model_params: Additional model parameters
            
        Returns:
            Dictionary with training metrics
        """
        print(f"\n🤖 Training {self.model_type.upper()} model...")
        
        if self.model_type == 'xgb':
            self._train_xgboost(X_train, y_train, X_val, y_val, **model_params)
        elif self.model_type == 'rf':
            self._train_random_forest(X_train, y_train, X_val, y_val, **model_params)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate on validation set
        y_val_pred = self.model.predict(X_val)
        metrics = self._calculate_metrics(y_val, y_val_pred)
        
        self.is_trained = True
        
        print("\n✅ Training complete!")
        return metrics
    
    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **params
    ):
        """Train XGBoost classifier"""
        
        # Default XGBoost parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'objective': 'binary:logistic',
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        default_params.update(params)
        
        self.model = xgb.XGBClassifier(**default_params)
        
        # Train with early stopping
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    
    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **params
    ):
        """Train Random Forest classifier"""
        from sklearn.ensemble import RandomForestClassifier
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42
        }
        
        default_params.update(params)
        
        self.model = RandomForestClassifier(**default_params)
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str = "Set") -> dict:
        """
        Evaluate model performance
        
        Args:
            X: Features
            y: True labels
            set_name: Name of dataset (for display)
            
        Returns:
            Dictionary with metrics
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        y_pred = self.model.predict(X)
        metrics = self._calculate_metrics(y, y_pred)
        
        print(f"\n📊 {set_name} Results:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1']:.4f}")
        
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Calculate classification metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X: Features
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            
            # Create DataFrame
            feat_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return feat_df.head(top_n)
        else:
            return pd.DataFrame()
    
    def save_model(self, filepath: str):
        """
        Save trained model
        
        Args:
            filepath: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'trained_date': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model
        
        Args:
            filepath: Path to model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data.get('model_type', 'xgb')
        self.is_trained = True
        
        print(f"✓ Model loaded from: {filepath}")

