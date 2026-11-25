"""
Test the Improved Intraday Model

Compare the new model's accuracy to the old one!
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.get_today_5min_data import get_today_5min_data
from src.add_intraday_features import IntradayFeatureExtractor
from src.feature_engineering.feature_extractor import FeatureExtractor

def test_improved_model():
    """Test the new intraday model on today's data"""
    
    print("=" * 60)
    print("🎯 TESTING IMPROVED INTRADAY MODEL")
    print("=" * 60)
    
    # Get today's data
    print("📊 Fetching today's AAPL data...")
    today_data = get_today_5min_data("AAPL")
    
    if today_data is None or today_data.empty:
        print("❌ No data")
        return
    
    # Process data
    data = today_data.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Calculate actual direction
    data['Next_Close'] = data['Close'].shift(-1)
    data['Actual_Direction'] = (data['Next_Close'] > data['Close']).astype(int)
    data = data[:-1]  # Remove last row
    
    # Add features
    print("🔧 Adding features...")
    feature_extractor = FeatureExtractor()
    data = feature_extractor.add_all_features(data)
    
    intraday_extractor = IntradayFeatureExtractor()
    data = intraday_extractor.add_intraday_features(data)
    
    # Load model
    print("🤖 Loading improved model...")
    try:
        with open("models/intraday_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("models/intraday_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open("models/intraday_features.pkl", 'rb') as f:
            feature_cols = pickle.load(f)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Prepare features
    print("🔧 Preparing features...")
    missing_features = [col for col in feature_cols if col not in data.columns]
    for feature in missing_features:
        data[feature] = 0
    
    feature_data = data[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    scaled_features = scaler.transform(feature_data)
    
    # Get predictions
    print("🔮 Getting predictions...")
    predictions = model.predict(scaled_features)
    probabilities = model.predict_proba(scaled_features)
    
    data['Prediction'] = predictions
    data['Confidence'] = np.max(probabilities, axis=1)
    data['Actual_Direction'] = data['Actual_Direction']
    
    # Calculate accuracy
    correct = (data['Prediction'] == data['Actual_Direction']).sum()
    total = len(data)
    accuracy = correct / total * 100
    
    print("\n" + "=" * 60)
    print("📊 ACCURACY RESULTS")
    print("=" * 60)
    print(f"🎯 Accuracy: {correct}/{total} = {accuracy:.1f}%")
    
    # Accuracy by confidence
    print(f"\n📊 Accuracy by Confidence:")
    for min_conf, max_conf in [(0.9, 1.0), (0.8, 0.9), (0.7, 0.8), (0.6, 0.7), (0.5, 0.6)]:
        mask = (data['Confidence'] >= min_conf) & (data['Confidence'] < max_conf)
        subset = data[mask]
        if len(subset) > 0:
            sub_accuracy = (subset['Prediction'] == subset['Actual_Direction']).sum() / len(subset) * 100
            print(f"   {min_conf:.1f}-{max_conf:.1f}: {len(subset)} bars, {sub_accuracy:.1f}% accurate")
    
    # Compare UP vs DOWN predictions
    print(f"\n📊 Prediction Breakdown:")
    up_preds = data[data['Prediction'] == 1]
    down_preds = data[data['Prediction'] == 0]
    
    if len(up_preds) > 0:
        up_acc = (up_preds['Prediction'] == up_preds['Actual_Direction']).sum() / len(up_preds) * 100
        print(f"   UP predictions: {len(up_preds)} ({len(up_preds)/len(data)*100:.1f}%), Accuracy: {up_acc:.1f}%")
    
    if len(down_preds) > 0:
        down_acc = (down_preds['Prediction'] == down_preds['Actual_Direction']).sum() / len(down_preds) * 100
        print(f"   DOWN predictions: {len(down_preds)} ({len(down_preds)/len(data)*100:.1f}%), Accuracy: {down_acc:.1f}%")
    
    # Show recent predictions
    print(f"\n📊 Recent Predictions (Last 10):")
    print(f"{'Time':<8} {'Pred':<4} {'Actual':<6} {'Correct':<7} {'Conf':<5}")
    print("-" * 45)
    
    recent = data.tail(10)
    for idx, row in recent.iterrows():
        pred = "UP" if row['Prediction'] == 1 else "DOWN"
        actual = "UP" if row['Actual_Direction'] == 1 else "DOWN"
        correct = "✅" if row['Prediction'] == row['Actual_Direction'] else "❌"
        time_str = idx.strftime('%H:%M') if isinstance(idx, pd.Timestamp) else str(idx)[-8:]
        print(f"{time_str:<8} {pred:<4} {actual:<6} {correct:<7} {row['Confidence']:.3f}")
    
    print("\n" + "=" * 60)
    
    # Compare to old model
    print("\n📊 COMPARISON:")
    print(f"   Old Model: 53.2% accuracy")
    print(f"   New Model: {accuracy:.1f}% accuracy")
    
    if accuracy > 53.2:
        print(f"   ✅ IMPROVED by {accuracy - 53.2:.1f}%!")
    elif accuracy == 53.2:
        print(f"   ➡️  SAME")
    else:
        print(f"   ⚠️  DOWN by {53.2 - accuracy:.1f}%")
    
    print("=" * 60)

if __name__ == "__main__":
    test_improved_model()
