"""
Test Bot Prediction Accuracy on Today's Data

Compare what the bot predicted vs what actually happened!
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.get_today_5min_data import get_today_5min_data
from src.feature_engineering.feature_extractor import FeatureExtractor

def calculate_actual_direction(data):
    """Calculate what actually happened (next bar direction)"""
    
    print("🔍 Calculating actual price movements...")
    
    # Calculate next bar's direction
    data['Next_Close'] = data['Close'].shift(-1)
    data['Actual_Direction'] = (data['Next_Close'] > data['Close']).astype(int)
    
    # Remove last row (no next bar)
    data = data[:-1]
    
    print(f"✅ Calculated actual directions for {len(data)} bars")
    return data

def test_prediction_accuracy():
    """Test how well the bot predicts vs reality"""
    
    print("=" * 60)
    print("🎯 TESTING BOT PREDICTION ACCURACY")
    print("=" * 60)
    
    # Get today's data
    print("📊 Fetching today's AAPL data...")
    today_data = get_today_5min_data("AAPL")
    
    if today_data is None or today_data.empty:
        print("❌ No data available")
        return
    
    # Process data
    print("🔧 Processing data...")
    data_processed = today_data.copy()
    data_processed = data_processed.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    data_processed = data_processed.set_index('Date')
    
    # Calculate actual directions
    data_with_actual = calculate_actual_direction(data_processed)
    
    # Load model and get predictions
    print("🤖 Loading model and getting predictions...")
    
    try:
        # Load model
        with open("models/us_max_model.pkl", 'rb') as f:
            model = pickle.load(f)
        with open("models/us_max_scaler.pkl", 'rb') as f:
            scaler = pickle.load(f)
        with open("models/us_max_features.pkl", 'rb') as f:
            feature_cols = pickle.load(f)
        
        # Extract features
        extractor = FeatureExtractor()
        data_with_features = extractor.add_all_features(data_with_actual)
        
        # Handle missing features
        missing_features = [col for col in feature_cols if col not in data_with_features.columns]
        for feature in missing_features:
            data_with_features[feature] = 0
        
        # Get predictions
        feature_data = data_with_features[feature_cols]
        scaled_features = scaler.transform(feature_data)
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        # Add predictions to data
        data_with_features['Prediction'] = predictions
        data_with_features['Confidence'] = np.max(probabilities, axis=1)
        data_with_features['Actual_Direction'] = data_with_features['Actual_Direction']
        
        print("✅ Got predictions!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Calculate accuracy
    print("\n" + "=" * 60)
    print("📊 ACCURACY ANALYSIS")
    print("=" * 60)
    
    # Overall accuracy
    correct = (data_with_features['Prediction'] == data_with_features['Actual_Direction']).sum()
    total = len(data_with_features)
    accuracy = correct / total * 100
    
    print(f"🎯 Overall Accuracy: {correct}/{total} = {accuracy:.1f}%")
    
    # Accuracy by confidence level
    print(f"\n📊 Accuracy by Confidence Level:")
    
    confidence_ranges = [
        (0.9, 1.0, "Very High (90-100%)"),
        (0.8, 0.9, "High (80-90%)"),
        (0.7, 0.8, "Medium (70-80%)"),
        (0.6, 0.7, "Low (60-70%)"),
        (0.5, 0.6, "Very Low (50-60%)")
    ]
    
    for min_conf, max_conf, label in confidence_ranges:
        mask = (data_with_features['Confidence'] >= min_conf) & (data_with_features['Confidence'] < max_conf)
        subset = data_with_features[mask]
        
        if len(subset) > 0:
            subset_correct = (subset['Prediction'] == subset['Actual_Direction']).sum()
            subset_accuracy = subset_correct / len(subset) * 100
            print(f"   {label}: {subset_correct}/{len(subset)} = {subset_accuracy:.1f}%")
    
    # Show some examples
    print(f"\n📊 Sample Predictions vs Reality:")
    print(f"{'Time':<8} {'Pred':<4} {'Actual':<6} {'Correct':<7} {'Conf':<5} {'Price':<8}")
    print("-" * 50)
    
    # Show last 10 predictions
    recent = data_with_features.tail(10)
    for idx, row in recent.iterrows():
        pred = "UP" if row['Prediction'] == 1 else "DOWN"
        actual = "UP" if row['Actual_Direction'] == 1 else "DOWN"
        correct = "✅" if row['Prediction'] == row['Actual_Direction'] else "❌"
        
        print(f"{idx.strftime('%H:%M'):<8} {pred:<4} {actual:<6} {correct:<7} {row['Confidence']:.3f} ${row['Close']:.2f}")
    
    # Analyze prediction patterns
    print(f"\n📊 Prediction Patterns:")
    
    up_predictions = data_with_features[data_with_features['Prediction'] == 1]
    down_predictions = data_with_features[data_with_features['Prediction'] == 0]
    
    if len(up_predictions) > 0:
        up_accuracy = (up_predictions['Prediction'] == up_predictions['Actual_Direction']).sum() / len(up_predictions) * 100
        print(f"   UP predictions: {len(up_predictions)} total, {up_accuracy:.1f}% accurate")
    
    if len(down_predictions) > 0:
        down_accuracy = (down_predictions['Prediction'] == down_predictions['Actual_Direction']).sum() / len(down_predictions) * 100
        print(f"   DOWN predictions: {len(down_predictions)} total, {down_accuracy:.1f}% accurate")
    
    # Show worst predictions (high confidence but wrong)
    print(f"\n⚠️  High Confidence Wrong Predictions:")
    wrong_high_conf = data_with_features[
        (data_with_features['Prediction'] != data_with_features['Actual_Direction']) & 
        (data_with_features['Confidence'] > 0.8)
    ]
    
    if len(wrong_high_conf) > 0:
        print(f"   Found {len(wrong_high_conf)} high-confidence wrong predictions")
        for idx, row in wrong_high_conf.head(5).iterrows():
            pred = "UP" if row['Prediction'] == 1 else "DOWN"
            actual = "UP" if row['Actual_Direction'] == 1 else "DOWN"
            print(f"   {idx.strftime('%H:%M')}: Predicted {pred}, Actual {actual}, Conf: {row['Confidence']:.3f}")
    else:
        print("   ✅ No high-confidence wrong predictions!")
    
    print("\n" + "=" * 60)
    print("✅ PREDICTION ACCURACY TEST COMPLETE!")
    print("=" * 60)
    
    # Summary
    if accuracy > 70:
        print("🎯 EXCELLENT: Bot is highly accurate!")
    elif accuracy > 60:
        print("✅ GOOD: Bot is reasonably accurate")
    elif accuracy > 50:
        print("⚠️  FAIR: Bot is slightly better than random")
    else:
        print("❌ POOR: Bot needs improvement")

if __name__ == "__main__":
    test_prediction_accuracy()
