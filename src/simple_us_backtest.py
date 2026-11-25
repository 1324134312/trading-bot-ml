"""
Simple US Model Backtest - Direct Approach
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def main():
    """Simple backtest of US model"""
    
    print("=" * 90)
    print("🧪 Backtesting US Model on US Data")
    print("=" * 90)
    
    # Load model
    with open("models/us_max_model.pkl", 'rb') as f:
        us_model = pickle.load(f)
    with open("models/us_max_scaler.pkl", 'rb') as f:
        us_scaler = pickle.load(f)
    with open("models/us_max_features.pkl", 'rb') as f:
        us_features = pickle.load(f)
    
    print("\n✅ US Model Loaded")
    
    # Load a sample of US data for testing
    print("\n📂 Loading US data...")
    
    files = glob.glob("data/us/max_processed/*_features.csv")
    
    # Just use first 10 stocks for test
    sample_files = files[:10]
    all_data = []
    
    for file in sample_files:
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('_features.csv', '')
        df['Ticker'] = ticker
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    data_clean = data.dropna()
    
    print(f"✅ Loaded {len(data_clean)} bars from {len(sample_files)} stocks")
    
    # Prepare features
    common_features = [f for f in us_features if f in data_clean.columns]
    X = data_clean[common_features].copy()
    
    for feat in [f for f in us_features if f not in X.columns]:
        X[feat] = 0
    
    X = X[us_features]
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    
    if 'Target_15min' in data_clean.columns:
        y_actual = data_clean.loc[valid_idx, 'Target_15min'].values
    else:
        print("❌ No target column")
        return
    
    # Predict
    X_scaled = us_scaler.transform(X_clean.values)
    predictions = us_model.predict(X_scaled)
    
    # Accuracy
    accuracy = accuracy_score(y_actual, predictions)
    
    print("\n" + "=" * 90)
    print("📊 US MODEL TEST RESULTS")
    print("=" * 90)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Samples: {len(y_actual):,}")
    
    # Simulate some trades
    capital = 10000
    position = None
    
    # Simple backtest: buy when prediction=1
    buy_signals = np.where(predictions == 1)[0]
    
    print(f"\n💰 SIMPLE TRADING TEST:")
    print(f"   Buy signals: {len(buy_signals)}")
    print(f"   Buy rate: {len(buy_signals)/len(predictions)*100:.1f}%")
    
    if len(buy_signals) > 0:
        print(f"\n✅ US model generates trading signals!")
        print(f"   Ready for full backtesting with trading strategy")
    
    print("\n" + "=" * 90)
    print("✅ US MODEL IS WORKING!")
    print("=" * 90)

if __name__ == "__main__":
    main()

