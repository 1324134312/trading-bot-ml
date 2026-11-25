"""
Train UK 5-Minute Day Trading Model
Uses 41K+ bars from 62 stocks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_uk_5min_model():
    """Train UK day trading model on 5-minute data"""
    
    print("=" * 70)
    print("🇬🇧 Training UK 5-Minute Day Trading Model")
    print("=" * 70)
    print()
    
    # Initialize trainer
    trainer = ModelTrainer(model_type='xgb')
    
    # Load all processed data
    data_dir = "data/uk/processed_5min"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"📂 Loading data from {len(files)} stocks...")
    all_data = []
    
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            ticker = os.path.basename(filepath).split('_')[0]
            print(f"   ✓ {ticker}: {len(df)} bars")
            all_data.append(df)
        except:
            pass
    
    if not all_data:
        print("❌ No data found!")
        return None
    
    # Combine all data
    data = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total dataset: {len(data)} bars")
    print(f"   Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    
    # Prepare features
    print(f"\n🔄 Preparing data...")
    
    # Exclude non-feature columns
    exclude_cols = ['Datetime', 'Date', 'Ticker', 'FetchDate', 'Dividends', 
                    'Stock Splits', 'Target_15min', 'Capital Gains']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"   Features: {len(feature_cols)}")
    
    # Clean data
    data_clean = data.dropna()
    print(f"   Valid rows: {len(data_clean)}")
    
    # Time series split (70/30)
    split_idx = int(len(data_clean) * 0.7)
    
    train_data = data_clean.iloc[:split_idx]
    test_data = data_clean.iloc[split_idx:]
    
    # Prepare features and targets
    X_train = train_data[feature_cols].values
    y_train = train_data['Target_15min'].values
    X_test = test_data[feature_cols].values
    y_test = test_data['Target_15min'].values
    
    # Split training into train/val
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    trainer.scaler = scaler
    trainer.feature_names = feature_cols
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train_split)} bars")
    print(f"   Validation: {len(X_val)} bars")
    print(f"   Test: {len(X_test)} bars")
    
    print(f"\n📈 Target Distribution:")
    buy_pct = np.sum(y_train == 1) / len(y_train) * 100
    print(f"   Buy signals: {np.sum(y_train == 1):,} ({buy_pct:.1f}%)")
    print(f"   Hold/Sell: {np.sum(y_train == 0):,} ({100-buy_pct:.1f}%)")
    
    # Train model
    print("\n🤖 Training model...")
    val_metrics = trainer.train(
        X_train_scaled, y_train_split,
        X_val_scaled, y_val,
        n_estimators=150,
        max_depth=8,
        learning_rate=0.05,
        scale_pos_weight=1.2  # Handle class imbalance
    )
    
    # Test
    print("\n📊 Testing...")
    test_metrics = trainer.evaluate(X_test_scaled, y_test, "Test Set")
    
    # Feature importance
    print("\n🔍 Top 20 Features:")
    feat_importance = trainer.get_feature_importance(top_n=20)
    for idx, row in feat_importance.iterrows():
        print(f"   {idx+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    # Save
    model_dir = 'src/models/trained_models'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'uk_5min_day_trading_model.pkl')
    trainer.save_model(model_path)
    
    print("\n" + "=" * 70)
    print("✅ UK 5-MINUTE MODEL COMPLETE!")
    print("=" * 70)
    print(f"🎯 Test Accuracy: {test_metrics['accuracy']:.2%}")
    print(f"💾 Model saved: {model_path}")
    print("\n📝 Model predicts: Will price go UP in next 15 minutes?")
    print("🎯 Designed for UK stock day trading on 5-minute bars!")
    
    return trainer

if __name__ == "__main__":
    train_uk_5min_model()

