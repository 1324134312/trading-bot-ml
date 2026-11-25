"""
Prediction Module
Make predictions with trained models
"""
import pandas as pd
import numpy as np
from typing import Optional
from .trainer import ModelTrainer

class ModelPredictor:
    """Make predictions using trained models"""
    
    def __init__(self, model_path: str):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to trained model file
        """
        self.trainer = ModelTrainer()
        self.trainer.load_model(model_path)
    
    def predict_direction(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict next day price direction
        
        Args:
            data: DataFrame with features
            
        Returns:
            DataFrame with predictions added
        """
        if not self.trainer.is_trained:
            raise ValueError("Model not loaded!")
        
        # Prepare features
        feature_cols = self.trainer.feature_names
        
        # Handle missing columns - create zeros if feature doesn't exist in data
        missing_cols = [col for col in feature_cols if col not in data.columns]
        if missing_cols:
            print(f"⚠️  Warning: Missing columns in data: {missing_cols}")
            for col in missing_cols:
                data[col] = 0
        
        X = data[feature_cols].values
        
        # Scale features
        X_scaled = self.trainer.scaler.transform(X)
        
        # Make predictions
        predictions = self.trainer.predict(X_scaled)
        probabilities = self.trainer.predict_proba(X_scaled)
        
        # Add to dataframe
        data['Prediction'] = predictions
        data['Prediction_Prob'] = probabilities[:, 1]  # Probability of up movement
        data['Confidence'] = np.abs(probabilities[:, 1] - 0.5) * 2  # 0-1 scale
        
        return data
    
    def get_buy_signals(self, data: pd.DataFrame, min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Get buy signals from predictions
        
        Args:
            data: DataFrame with predictions
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            DataFrame with buy signals only
        """
        buy_signals = data[
            (data['Prediction'] == 1) & 
            (data['Confidence'] >= min_confidence)
        ].copy()
        
        return buy_signals
    
    def get_sell_signals(self, data: pd.DataFrame, min_confidence: float = 0.6) -> pd.DataFrame:
        """
        Get sell signals from predictions
        
        Args:
            data: DataFrame with predictions
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            DataFrame with sell signals only
        """
        sell_signals = data[
            (data['Prediction'] == 0) & 
            (data['Confidence'] >= min_confidence)
        ].copy()
        
        return sell_signals

