"""
Train UK Day Trading Model for 1-hour intraday predictions
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
from config.settings import PROCESSED_DATA_DIR
import glob

def train_uk_day_trading_model():
    """Train model for UK day trading"""
    
    print("=" * 70)
    print("🇬🇧 Training UK Day Trading Model")
    print("=" * 70)
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(model_type='xgb')
    
    # Find all UK processed files
    uk_data_dir = "data/uk/processed"
    files = glob.glob(os.path.join(uk_data_dir, "*_features.csv"))
    
    if not files:
        print("❌ No UK data found!")
        return None
    
    print(f"📂 Loading UK intraday data...")
    all_data = []
    
    for filepath in files:
        df = pd.read_csv(filepath)
        all_data.append(df)
        ticker = os.path.basename(filepath).split('_')[0]
        print(f"   ✓ {ticker}: {len(df)} hour-bars")
    
    # Combine all UK stocks
    data = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total dataset: {len(data)} hour-bars")
    
    # Prepare data manually to avoid Date column issues
    print(f"\n🔄 Preparing data for training...")
    
    # Select features (exclude date columns)
    exclude_cols = ['Date', 'Datetime', 'Ticker', 'FetchDate', 'Target_2h']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    # Prepare for training
    data_clean = data.dropna()
    split_idx = int(len(data_clean) * 0.7)
    
    train_data = data_clean.iloc[:split_idx]
    test_data = data_clean.iloc[split_idx:]
    
    X_train = train_data[feature_cols].values
    y_train = train_data['Target_2h'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['Target_2h'].values
    
    # Further split training for validation
    split_idx_val = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx_val]
    X_val = X_train[split_idx_val:]
    y_train_split = y_train[:split_idx_val]
    y_val = y_train[split_idx_val:]
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    trainer.scaler = scaler
    trainer.feature_names = feature_cols
    
    # Split for validation
    split_idx = int(len(X_train) * 0.8)
    X_train_split = X_train[:split_idx]
    X_val = X_train[split_idx:]
    y_train_split = y_train[:split_idx]
    y_val = y_train[split_idx:]
    
    print(f"\n📈 Target Distribution:")
    print(f"   Class 0 (Down/Hold): {np.sum(y_train == 0)} ({np.sum(y_train == 0)/len(y_train)*100:.1f}%)")
    print(f"   Class 1 (Buy/Up): {np.sum(y_train == 1)} ({np.sum(y_train == 1)/len(y_train)*100:.1f}%)")
    
    # Train model
    print("\n🤖 Training UK Day Trading Model...")
    val_metrics = trainer.train(
        X_train_split, y_train_split,
        X_val, y_val,
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1
    )
    
    # Test
    print("\n📊 Testing model...")
    test_metrics = trainer.evaluate(X_test, y_test, "Test Set")
    
    # Feature importance
    print("\n🔍 Top 20 Features for UK Day Trading:")
    feat_importance = trainer.get_feature_importance(top_n=20)
    for idx, row in feat_importance.iterrows():
        print(f"   {idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    # Save model
    model_dir = 'src/models/trained_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'uk_day_trading_model.pkl')
    trainer.save_model(model_path)
    
    print("\n" + "=" * 70)
    print("✅ UK DAY TRADING MODEL COMPLETE!")
    print("=" * 70)
    print(f"🎯 Final Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"💾 Model saved: {model_path}")
    print("\n📝 Model predicts: Will price go UP in next 2 hours?")
    print("🎯 Designed for UK stock day trading!")
    
    return trainer, test_metrics

if __name__ == "__main__":
    train_uk_day_trading_model()

