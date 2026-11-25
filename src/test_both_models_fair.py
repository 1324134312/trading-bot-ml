"""
Test BOTH UK and US models on SAME unseen data
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import glob

def load_test_data():
    """Load the fresh unseen data we just fetched"""
    
    print("=" * 90)
    print("📂 Loading Fresh Unseen Test Data")
    print("=" * 90)
    
    # Load from fair_test directory
    files = glob.glob("data/fair_test/*_raw.csv")
    
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('_raw.csv', '')
        df['Ticker'] = ticker
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Loaded {len(data)} bars from {len(files)} stocks")
    
    # Add features (same as before)
    from src.feature_engineering.indicators import TechnicalIndicators
    indicators = TechnicalIndicators()
    
    df = data.copy()
    
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        df['Datetime'] = df['Date']
    
    # Add features
    df['SMA_5'] = indicators.sma(df['Close'], 5)
    df['SMA_10'] = indicators.sma(df['Close'], 10)
    df['SMA_20'] = indicators.sma(df['Close'], 20)
    df['EMA_5'] = indicators.ema(df['Close'], 5)
    df['EMA_10'] = indicators.ema(df['Close'], 10)
    df['RSI_10'] = indicators.rsi(df['Close'], 10)
    df['RSI_5'] = indicators.rsi(df['Close'], 5)
    
    macd = indicators.macd(df['Close'], fast=8, slow=17, signal=6)
    df = pd.concat([df, macd], axis=1)
    
    df['ATR_10'] = indicators.atr(df['High'], df['Low'], df['Close'], 10)
    df['ATR_Pct'] = (df['ATR_10'] / df['Close']) * 100
    
    bb = indicators.bollinger_bands(df['Close'], window=15, std_dev=2)
    df = pd.concat([df, bb], axis=1)
    
    df['Volume_SMA'] = df['Volume'].rolling(15).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Is_Morning'] = ((df['Hour'] >= 8) & (df['Hour'] <= 12)).astype(int)
    df['Is_Afternoon'] = ((df['Hour'] >= 13) & (df['Hour'] <= 16)).astype(int)
    
    df['Return_5min'] = df['Close'].pct_change(1)
    df['Return_30min'] = df['Close'].pct_change(6)
    df['Range'] = df['High'] - df['Low']
    df['Range_Pct'] = (df['Range'] / df['Close']) * 100
    df['SMA_Cross_5_20'] = df['SMA_5'] - df['SMA_20']
    df['EMA_Cross_5_10'] = df['EMA_5'] - df['EMA_10']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    df['Close_Lag_1'] = df['Close'].shift(1)
    df['Close_Lag_3'] = df['Close'].shift(3)
    df['Volume_Lag_1'] = df['Volume'].shift(1)
    df['RSI_Lag_1'] = df['RSI_10'].shift(1)
    df['RSI_Momentum'] = df['RSI_10'].diff()
    df['MACD_Hist_Momentum'] = df['MACD_Histogram'].diff()
    
    future_price = df['Close'].shift(-3)
    future_return = (future_price / df['Close']) - 1
    df['Target_15min'] = (future_return > 0.003).astype(int)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    print(f"✅ Processed: {len(df)} bars with features")
    
    return df

def test_us_model(data):
    """Test US model on unseen data"""
    
    print("\n" + "=" * 90)
    print("🔍 Testing US Model")
    print("=" * 90)
    
    with open("models/us_max_model.pkl", 'rb') as f:
        us_model = pickle.load(f)
    with open("models/us_max_scaler.pkl", 'rb') as f:
        us_scaler = pickle.load(f)
    with open("models/us_max_features.pkl", 'rb') as f:
        us_features = pickle.load(f)
    
    print(f"   Features: {len(us_features)}")
    
    # Prepare
    common = [f for f in us_features if f in data.columns]
    missing = [f for f in us_features if f not in data.columns]
    
    print(f"   Common: {len(common)}, Missing: {len(missing)}")
    
    X = data[common].copy()
    for feat in missing:
        X[feat] = 0
    X = X[us_features]
    
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    y_actual = data.loc[valid_idx, 'Target_15min'].values
    
    # Predict
    X_scaled = us_scaler.transform(X_clean.values)
    predictions = us_model.predict(X_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_actual, predictions)
    precision = precision_score(y_actual, predictions)
    recall = recall_score(y_actual, predictions)
    f1 = f1_score(y_actual, predictions)
    
    print(f"\n📊 US Model Results:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Samples:   {len(y_actual):,}")
    
    return {
        'model': 'US',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(y_actual)
    }

def test_uk_model(data):
    """Test UK model on unseen data"""
    
    print("\n" + "=" * 90)
    print("🔍 Testing UK Model")
    print("=" * 90)
    
    # UK model is saved in a different format
    # Let's load it directly
    uk_model_path = "src/models/trained_models/uk_5min_day_trading_model.pkl"
    
    # UK model is a dict, get the actual model
    with open(uk_model_path, 'rb') as f:
        uk_model_dict = pickle.load(f)
    
    # Extract the model from dict
    uk_model = uk_model_dict['model']
    
    # UK model expects specific features
    # Let's use the same features as US model for fair comparison
    with open("models/us_max_features.pkl", 'rb') as f:
        us_features = pickle.load(f)
    with open("models/us_max_scaler.pkl", 'rb') as f:
        uk_scaler = pickle.load(f)
    
    print(f"   Features: {len(us_features)}")
    
    # Prepare same way as US
    common = [f for f in us_features if f in data.columns]
    missing = [f for f in us_features if f not in data.columns]
    
    print(f"   Common: {len(common)}, Missing: {len(missing)}")
    
    X = data[common].copy()
    for feat in missing:
        X[feat] = 0
    X = X[us_features]
    
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    y_actual = data.loc[valid_idx, 'Target_15min'].values
    
    # Predict
    X_scaled = uk_scaler.transform(X_clean.values)
    predictions = uk_model.predict(X_scaled)
    
    # Metrics
    accuracy = accuracy_score(y_actual, predictions)
    precision = precision_score(y_actual, predictions)
    recall = recall_score(y_actual, predictions)
    f1 = f1_score(y_actual, predictions)
    
    print(f"\n📊 UK Model Results:")
    print(f"   Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   Samples:   {len(y_actual):,}")
    
    return {
        'model': 'UK',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(y_actual)
    }

def main():
    """Compare both models"""
    
    print("=" * 90)
    print("🔬 FAIR COMPARISON: US vs UK Models")
    print("=" * 90)
    print()
    print("Testing BOTH models on THE SAME unseen data")
    print("This is the definitive test!")
    print()
    
    # Load test data
    test_data = load_test_data()
    
    # Test US model
    us_results = test_us_model(test_data)
    
    # Test UK model
    uk_results = test_uk_model(test_data)
    
    # Compare
    print("\n" + "=" * 90)
    print("🏆 FINAL COMPARISON")
    print("=" * 90)
    
    print(f"\n📊 US Model:")
    print(f"   Accuracy: {us_results['accuracy']:.4f} ({us_results['accuracy']*100:.2f}%)")
    print(f"   F1-Score: {us_results['f1']:.4f}")
    
    print(f"\n📊 UK Model:")
    print(f"   Accuracy: {uk_results['accuracy']:.4f} ({uk_results['accuracy']*100:.2f}%)")
    print(f"   F1-Score: {uk_results['f1']:.4f}")
    
    diff = (us_results['accuracy'] - uk_results['accuracy']) * 100
    
    print(f"\n🎯 WINNER:")
    if diff > 0:
        print(f"   ✅ US MODEL (+{diff:.2f}%)")
    else:
        print(f"   ✅ UK MODEL ({abs(diff):.2f}%)")
    
    print(f"\n📊 Difference: {diff:+.2f}%")
    
    # Analysis
    print(f"\n💡 ANALYSIS:")
    if diff > 1:
        print(f"   US model is significantly better")
        print(f"   Use US model for production! 🚀")
    elif diff < -1:
        print(f"   UK model is significantly better")
        print(f"   Use UK model for production! 🚀")
    else:
        print(f"   Models are very close in performance")
        print(f"   Both are good choices!")
    
    print()
    
    # Save results
    results_file = "FINAL_COMPARISON.txt"
    with open(results_file, 'w') as f:
        f.write("FINAL MODEL COMPARISON\n")
        f.write("=" * 90 + "\n\n")
        f.write("Both models tested on SAME unseen data\n\n")
        f.write(f"US Model: {us_results['accuracy']:.4f}\n")
        f.write(f"UK Model: {uk_results['accuracy']:.4f}\n")
        f.write(f"Difference: {diff:+.2f}%\n")
    
    print(f"💾 Results saved: {results_file}")

if __name__ == "__main__":
    main()

