"""
Test US Model on UK Stocks - Direct Comparison
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

def test_us_model_on_uk():
    """Test US max model on UK stocks"""
    
    print("=" * 90)
    print("🧪 US MODEL vs UK MODEL - Direct Test")
    print("=" * 90)
    
    # Load US model
    print("\n📂 Loading US Max Model...")
    
    with open("models/us_max_model.pkl", 'rb') as f:
        us_model = pickle.load(f)
    with open("models/us_max_scaler.pkl", 'rb') as f:
        us_scaler = pickle.load(f)
    with open("models/us_max_features.pkl", 'rb') as f:
        us_features = pickle.load(f)
    
    print(f"✅ US model loaded ({len(us_features)} features)")
    
    # Load UK data
    print("\n📂 Loading UK Test Data...")
    
    files = glob.glob("data/uk/processed_5min/*_features.csv")
    all_data = []
    
    for file in files[:10]:  # Test on first 10 stocks
        df = pd.read_csv(file)
        all_data.append(df)
    
    uk_data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Loaded {len(uk_data)} UK bars")
    
    # Prepare features
    common_features = [f for f in us_features if f in uk_data.columns]
    missing = [f for f in us_features if f not in uk_data.columns]
    
    print(f"📊 Features: {len(common_features)} common, {len(missing)} missing")
    
    # Prepare data
    X = uk_data[common_features].copy()
    for feat in missing:
        X[feat] = 0
    X = X[us_features]
    
    # Clean
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    
    if 'Target_15min' in uk_data.columns:
        y_actual = uk_data.loc[valid_idx, 'Target_15min'].values
    else:
        print("❌ No target column")
        return
    
    # Predict
    X_scaled = us_scaler.transform(X_clean.values)
    predictions = us_model.predict(X_scaled)
    
    # Results
    accuracy = accuracy_score(y_actual, predictions)
    
    print("\n" + "=" * 90)
    print("📊 RESULTS: US Model on UK Stocks")
    print("=" * 90)
    print(f"\n🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"📊 Samples: {len(y_actual):,}")
    
    # Compare with UK model
    print("\n" + "=" * 90)
    print("🏆 COMPARISON")
    print("=" * 90)
    print(f"\nUK Model (from training): 91.70%")
    print(f"US Model (this test):      {accuracy*100:.2f}%")
    print(f"Difference:                 {(accuracy - 0.917)*100:+.2f}%")
    
    if accuracy > 0.917:
        print(f"\n✅ US MODEL IS SUPERIOR!")
        improvement = ((accuracy - 0.917) / 0.917) * 100
        print(f"   +{improvement:.1f}% better")
    else:
        print(f"\n✅ UK MODEL IS SUPERIOR!")
        decline = ((0.917 - accuracy) / 0.917) * 100
        print(f"   US model is {decline:.1f}% worse")
    
    print()
    
    return accuracy

if __name__ == "__main__":
    test_us_model_on_uk()

