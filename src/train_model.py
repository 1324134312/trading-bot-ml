"""
Train ML model for trading predictions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
import glob

def train_model_on_tickers(tickers: list, model_type: str = 'xgb', save_name: str = 'default'):
    """
    Train model on multiple tickers
    
    Args:
        tickers: List of ticker symbols
        model_type: Type of model ('xgb', 'rf')
        save_name: Name for saved model
    """
    print("=" * 60)
    print(f"🤖 Training {model_type.upper()} Model")
    print("=" * 60)
    print()
    
    # Load and combine data from all tickers
    all_data = []
    for ticker in tickers:
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.csv")
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            all_data.append(df)
            print(f"✓ Loaded {ticker} ({len(df)} rows)")
        else:
            print(f"⚠️  Missing file: {filepath}")
    
    if not all_data:
        print("❌ No data found!")
        return None
    
    # Combine all data
    data = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total dataset: {len(data)} rows")
    
    # Initialize trainer
    trainer = ModelTrainer(model_type=model_type)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare_data(
        data,
        target_col='Target_Direction',
        test_size=0.2,
        val_size=0.2
    )
    
    print(f"\n📈 Target distribution:")
    print(f"   Class 0 (Down): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"   Class 1 (Up):   {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    
    # Train model
    val_metrics = trainer.train(X_train, y_train, X_val, y_val)
    
    print(f"\n✅ Validation Metrics:")
    print(f"   Accuracy:  {val_metrics['accuracy']:.4f}")
    print(f"   F1-Score:  {val_metrics['f1']:.4f}")
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(X_test, y_test, "Test")
    
    # Feature importance
    print("\n🔍 Top 20 Most Important Features:")
    feat_importance = trainer.get_feature_importance(top_n=20)
    for idx, row in feat_importance.iterrows():
        print(f"   {idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    # Save model
    model_dir = 'src/models/trained_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{save_name}_model.pkl')
    trainer.save_model(model_path)
    
    return trainer

def main():
    """Main training function"""
    
    # Train on all available tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
    
    print("🎯 Training on multiple tickers for robustness")
    
    # Train XGBoost model
    trainer = train_model_on_tickers(
        tickers=tickers,
        model_type='xgb',
        save_name='multi_ticker_xgb'
    )
    
    if trainer:
        print("\n" + "=" * 60)
        print("✅ Model Training Complete!")
        print("=" * 60)
        print(f"\n💾 Model saved to: src/models/trained_models/multi_ticker_xgb_model.pkl")
        print("\n🎯 Ready for predictions and backtesting!")
    else:
        print("\n❌ Training failed!")

if __name__ == "__main__":
    main()

