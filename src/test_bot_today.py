"""
Test Bot on Today's Real-Time Data

Using the 5-minute data we just fetched to get live predictions!
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
from src.models.predictor import ModelPredictor

def process_today_data_for_model(data):
    """Process today's data to match model features"""
    
    print("🔧 Processing data for model...")
    
    # Ensure we have the required columns
    required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in data.columns:
            print(f"❌ Missing column: {col}")
            return None
    
    # Rename columns to match expected format
    data_processed = data.copy()
    data_processed = data_processed.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High', 
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Set Date as index
    data_processed = data_processed.set_index('Date')
    
    # Add ticker column (placeholder for now)
    data_processed['Ticker'] = 'AAPL'  # We'll update this per symbol
    
    print(f"✅ Processed {len(data_processed)} bars")
    return data_processed

def get_model_predictions(data, model_path="models/us_max_model.pkl"):
    """Get predictions from the trained model"""
    
    print(f"🤖 Loading model from {model_path}...")
    
    try:
        # Load the US model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler
        scaler_path = "models/us_max_scaler.pkl"
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load features
        features_path = "models/us_max_features.pkl"
        with open(features_path, 'rb') as f:
            feature_cols = pickle.load(f)
        
        print(f"✅ Model loaded with {len(feature_cols)} features")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    # Extract features
    print("🔧 Extracting features...")
    extractor = FeatureExtractor()
    
    try:
        # Process the data to add features
        data_with_features = extractor.add_all_features(data)
        
        # Get only the feature columns
        if feature_cols:
            # Ensure all required features exist
            missing_features = [col for col in feature_cols if col not in data_with_features.columns]
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
                # Fill missing features with 0
                for feature in missing_features:
                    data_with_features[feature] = 0
            
            feature_data = data_with_features[feature_cols]
        else:
            print("❌ No feature columns found")
            return None
        
        print(f"✅ Extracted {len(feature_data.columns)} features")
        
    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return None
    
    # Scale features
    print("🔧 Scaling features...")
    try:
        scaled_features = scaler.transform(feature_data)
        print("✅ Features scaled")
    except Exception as e:
        print(f"❌ Error scaling features: {e}")
        return None
    
    # Get predictions
    print("🔮 Getting predictions...")
    try:
        predictions = model.predict(scaled_features)
        probabilities = model.predict_proba(scaled_features)
        
        # Add predictions to data
        data_with_features['Prediction'] = predictions
        data_with_features['Confidence'] = np.max(probabilities, axis=1)
        data_with_features['Buy_Signal'] = (predictions == 1) & (np.max(probabilities, axis=1) > 0.65)
        
        print(f"✅ Got {len(predictions)} predictions")
        print(f"📊 Buy signals: {sum(data_with_features['Buy_Signal'])}")
        print(f"📊 Avg confidence: {data_with_features['Confidence'].mean():.3f}")
        
        return data_with_features
        
    except Exception as e:
        print(f"❌ Error getting predictions: {e}")
        return None

def analyze_today_predictions(data_with_predictions):
    """Analyze today's predictions"""
    
    print("\n" + "=" * 60)
    print("📊 TODAY'S PREDICTION ANALYSIS")
    print("=" * 60)
    
    if data_with_predictions is None:
        print("❌ No predictions to analyze")
        return
    
    # Overall stats
    total_bars = len(data_with_predictions)
    buy_signals = sum(data_with_predictions['Buy_Signal'])
    avg_confidence = data_with_predictions['Confidence'].mean()
    
    print(f"📈 Total bars analyzed: {total_bars}")
    print(f"🎯 Buy signals: {buy_signals} ({buy_signals/total_bars*100:.1f}%)")
    print(f"🎯 Average confidence: {avg_confidence:.3f}")
    
    # Show recent predictions
    print(f"\n📊 Recent predictions (last 5 bars):")
    recent = data_with_predictions.tail(5)
    
    for idx, row in recent.iterrows():
        signal = "🟢 BUY" if row['Buy_Signal'] else "🔴 HOLD"
        direction = "📈 UP" if row['Prediction'] == 1 else "📉 DOWN"
        print(f"   {idx.strftime('%H:%M')}: {signal} | {direction} | Conf: {row['Confidence']:.3f} | Price: ${row['Close']:.2f}")
    
    # Show high confidence signals
    high_conf = data_with_predictions[data_with_predictions['Confidence'] > 0.8]
    if not high_conf.empty:
        print(f"\n🎯 High confidence signals (>0.8):")
        for idx, row in high_conf.iterrows():
            signal = "🟢 BUY" if row['Buy_Signal'] else "🔴 HOLD"
            print(f"   {idx.strftime('%H:%M')}: {signal} | Conf: {row['Confidence']:.3f} | Price: ${row['Close']:.2f}")

def test_bot_on_today():
    """Main function to test bot on today's data"""
    
    print("=" * 60)
    print("🤖 TESTING BOT ON TODAY'S REAL-TIME DATA")
    print("=" * 60)
    
    # Get today's data for AAPL
    print("📊 Fetching today's data for AAPL...")
    today_data = get_today_5min_data("AAPL")
    
    if today_data is None or today_data.empty:
        print("❌ No data available for today")
        return
    
    print(f"✅ Got {len(today_data)} bars of data")
    
    # Process data for model
    processed_data = process_today_data_for_model(today_data)
    
    if processed_data is None:
        print("❌ Failed to process data")
        return
    
    # Get predictions
    predictions = get_model_predictions(processed_data)
    
    if predictions is None:
        print("❌ Failed to get predictions")
        return
    
    # Analyze results
    analyze_today_predictions(predictions)
    
    print("\n" + "=" * 60)
    print("✅ BOT TEST COMPLETE!")
    print("=" * 60)
    
    # Show trading opportunities
    buy_signals = predictions[predictions['Buy_Signal']]
    if not buy_signals.empty:
        print(f"\n🎯 TRADING OPPORTUNITIES FOUND:")
        print(f"   {len(buy_signals)} buy signals today")
        print(f"   Latest signal: {buy_signals.index[-1].strftime('%H:%M')}")
        print(f"   Price at signal: ${buy_signals['Close'].iloc[-1]:.2f}")
        print(f"   Confidence: {buy_signals['Confidence'].iloc[-1]:.3f}")
    else:
        print(f"\n⚠️  No trading opportunities today")
        print(f"   Bot is being conservative")
        print(f"   Wait for better signals")

if __name__ == "__main__":
    test_bot_on_today()
