"""
Train Advanced Ensemble Model with Optimizations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.advanced_trainer import AdvancedTrainer
from config.settings import PROCESSED_DATA_DIR
import glob

def train_advanced_model():
    """Train advanced ensemble model on all tickers"""
    
    print("=" * 70)
    print("🚀 ADVANCED TRAINING - Ensemble + Hyperparameter Tuning")
    print("=" * 70)
    print()
    
    # Initialize advanced trainer
    trainer = AdvancedTrainer()
    
    # Load and combine data from all tickers
    print("📂 Loading data from all tickers...")
    all_data = []
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
    
    for ticker in tickers:
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'], utc=True)
            all_data.append(df)
            print(f"   ✓ {ticker}: {len(df)} rows")
    
    if not all_data:
        print("❌ No data found!")
        return None
    
    # Combine all data
    data = pd.concat(all_data, ignore_index=True)
    data = data.sort_values('Date')
    
    print(f"\n📊 Total dataset: {len(data)} rows")
    print(f"   Date range: {data['Date'].min()} to {data['Date'].max()}")
    
    # Prepare data with time series split
    print("\n" + "=" * 70)
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        data,
        target_col='Target_Direction',
        test_size=0.3  # Hold out last 30% for testing
    )
    
    print(f"\n📈 Target distribution:")
    print(f"   Class 0 (Down): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"   Class 1 (Up):   {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    
    # Further split training data for validation
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val = y_train[split_idx:]
    
    print(f"\n   Further split training data:")
    print(f"   Training: {X_train_split.shape[0]} samples")
    print(f"   Validation: {X_val.shape[0]} samples")
    
    # Train advanced ensemble model
    print("\n" + "=" * 70)
    trainer.train_ensemble(
        X_train_split,
        y_train_split,
        X_val,
        y_val,
        use_class_balancing=True,
        tune_hyperparameters=True
    )
    
    # Evaluate on test set
    print("\n" + "=" * 70)
    test_metrics = trainer.evaluate(X_test, y_test, "Test Set")
    
    # Show feature importance
    print("\n" + "=" * 70)
    print("🔍 Top 20 Most Important Features:")
    feat_importance = trainer.get_feature_importance(top_n=20)
    for idx, row in feat_importance.iterrows():
        print(f"   {idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    # Save model
    model_dir = 'src/models/trained_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'advanced_ensemble_model.pkl')
    trainer.save_model(model_path)
    
    print("\n" + "=" * 70)
    print("✅ ADVANCED TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n🎯 Final Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"💾 Model saved to: {model_path}")
    print("\n🚀 Model ready for predictions and backtesting!")
    
    return trainer

if __name__ == "__main__":
    train_advanced_model()

