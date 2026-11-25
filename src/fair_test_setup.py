"""
Fair Test Setup: Test Both Models on Completely Unseen UK Data

Strategy:
1. Fetch NEW UK data (from today)
2. Process with features
3. Test both US and UK models on same unseen data
4. Compare fairly
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import pickle
import numpy as np
from datetime import datetime
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# UK stocks NOT in training data
UNSEEN_UK_STOCKS = [
    "BATS.L", "EVR.L", "HMSO.L", "ITRK.L", "JD.L",
    "LAND.L", "LSEG.L", "MCG.L", "MKS.L", "NXT.L",
    "PETS.L", "PRU.L", "REL.L", "RS1.L", "RTO.L",
    "SAFE.L", "SBRY.L", "SGE.L", "SHI.L", "SN.L",
    "SSE.L", "SSPG.L", "TATE.L", "TSCO.L", "VSVS.L",
    "WTB.L"
]

def fetch_fresh_uk_data():
    """Fetch brand new UK data from today"""
    
    print("=" * 90)
    print("📊 Fetching FRESH UK Data (Never Seen by Either Model)")
    print("=" * 90)
    
    os.makedirs("data/fair_test", exist_ok=True)
    
    all_data = []
    
    for ticker in tqdm(UNSEEN_UK_STOCKS, desc="Fetching unseen stocks"):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="7d", interval="5m")
            
            if not data.empty:
                data.reset_index(inplace=True)
                
                # Add ticker
                data['Ticker'] = ticker
                
                # Save
                output_file = f"data/fair_test/{ticker}_raw.csv"
                data.to_csv(output_file, index=False)
                
                all_data.append(data)
                print(f"   ✅ {ticker}: {len(data)} bars")
        
        except Exception as e:
            print(f"   ⚠️  {ticker}: {e}")
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        
        print(f"\n✅ Fetched {len(combined)} bars from {len(all_data)} unseen stocks")
        print(f"📅 Date range: {combined['Datetime'].min()} to {combined['Datetime'].max()}")
        
        return combined
    else:
        return pd.DataFrame()

def add_features_fair(data):
    """Add features to fresh data"""
    
    print("\n🔧 Adding features...")
    
    from src.feature_engineering.indicators import TechnicalIndicators
    indicators = TechnicalIndicators()
    
    df = data.copy()
    
    # Rename columns if needed
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        df['Datetime'] = df['Date']
    
    # Basic features
    df['SMA_5'] = indicators.sma(df['Close'], 5)
    df['SMA_10'] = indicators.sma(df['Close'], 10)
    df['SMA_20'] = indicators.sma(df['Close'], 20)
    df['EMA_5'] = indicators.ema(df['Close'], 5)
    df['EMA_10'] = indicators.ema(df['Close'], 10)
    df['RSI_10'] = indicators.rsi(df['Close'], 10)
    df['RSI_5'] = indicators.rsi(df['Close'], 5)
    
    # MACD
    macd = indicators.macd(df['Close'], fast=8, slow=17, signal=6)
    df = pd.concat([df, macd], axis=1)
    
    # ATR
    df['ATR_10'] = indicators.atr(df['High'], df['Low'], df['Close'], 10)
    df['ATR_Pct'] = (df['ATR_10'] / df['Close']) * 100
    
    # Bollinger Bands
    bb = indicators.bollinger_bands(df['Close'], window=15, std_dev=2)
    df = pd.concat([df, bb], axis=1)
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(15).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Time features
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    df['Is_Morning'] = ((df['Hour'] >= 8) & (df['Hour'] <= 12)).astype(int)
    df['Is_Afternoon'] = ((df['Hour'] >= 13) & (df['Hour'] <= 16)).astype(int)
    
    # Returns
    df['Return_5min'] = df['Close'].pct_change(1)
    df['Return_30min'] = df['Close'].pct_change(6)
    
    # Target
    future_price = df['Close'].shift(-3)
    future_return = (future_price / df['Close']) - 1
    df['Target_15min'] = (future_return > 0.003).astype(int)
    
    # More features
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
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    
    print(f"✅ Processed: {len(df)} bars with features")
    
    return df

def test_model_fair_simple(model, scaler, features, test_data, model_name):
    """Simple fair test"""
    
    print(f"\n🔍 Testing {model_name}...")
    
    # Prepare features
    common_features = [f for f in features if f in test_data.columns]
    missing = [f for f in features if f not in test_data.columns]
    
    print(f"   Features: {len(common_features)} common, {len(missing)} missing")
    
    # Prepare data
    X = test_data[common_features].copy()
    for feat in missing:
        X[feat] = 0
    
    X = X[features]
    X = X.replace([np.inf, -np.inf], np.nan)
    valid_idx = X.notna().all(axis=1)
    X_clean = X[valid_idx]
    
    if 'Target_15min' in test_data.columns:
        y_actual = test_data.loc[valid_idx, 'Target_15min'].values
    else:
        print(f"   ❌ No target column")
        return None
    
    # Predict
    X_scaled = scaler.transform(X_clean.values)
    predictions = model.predict(X_scaled)
    
    # Calculate
    accuracy = accuracy_score(y_actual, predictions)
    
    print(f"   ✅ Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📊 Samples: {len(y_actual):,}")
    
    return accuracy

def main():
    """Run fair test"""
    
    print("=" * 90)
    print("🔬 FAIR MODEL COMPARISON")
    print("=" * 90)
    print()
    print("Testing BOTH models on SAME unseen UK data")
    print()
    
    # Step 1: Fetch fresh data
    fresh_data = fetch_fresh_uk_data()
    
    if fresh_data.empty:
        print("❌ No fresh data fetched")
        return
    
    # Step 2: Add features
    fresh_with_features = add_features_fair(fresh_data)
    
    # Step 3: Test US Model
    print("\n" + "=" * 90)
    print("📂 Loading US Model")
    print("=" * 90)
    
    with open("models/us_max_model.pkl", 'rb') as f:
        us_model = pickle.load(f)
    with open("models/us_max_scaler.pkl", 'rb') as f:
        us_scaler = pickle.load(f)
    with open("models/us_max_features.pkl", 'rb') as f:
        us_features = pickle.load(f)
    
    us_accuracy = test_model_fair_simple(us_model, us_scaler, us_features, fresh_with_features, "US Max Model")
    
    # Step 4: Test UK Model (need to adapt)
    print("\n" + "=" * 90)
    print("📂 Loading UK Model")
    print("=" * 90)
    
    uk_model_path = "src/models/trained_models/uk_5min_day_trading_model.pkl"
    
    if os.path.exists(uk_model_path):
        try:
            # Try to load UK model properly
            from src.models.predictor import ModelPredictor
            uk_predictor = ModelPredictor(uk_model_path)
            
            # For fair comparison, we need to adapt UK predictions
            # This is tricky because UK model uses a different format
            print("   ✅ UK model loaded")
            print("   ⚠️  UK model uses different format - needs adaptation")
            uk_accuracy = None
        except:
            uk_accuracy = None
    else:
        uk_accuracy = None
    
    # Step 5: Compare
    print("\n" + "=" * 90)
    print("🏆 FAIR COMPARISON RESULTS")
    print("=" * 90)
    
    if us_accuracy:
        print(f"\n📊 US Model: {us_accuracy:.4f} ({us_accuracy*100:.2f}%)")
    
    if uk_accuracy:
        print(f"📊 UK Model: {uk_accuracy:.4f} ({uk_accuracy*100:.2f}%)")
        
        if us_accuracy > uk_accuracy:
            print(f"\n🏆 WINNER: US MODEL (+{(us_accuracy-uk_accuracy)*100:.2f}%)")
        else:
            print(f"\n🏆 WINNER: UK MODEL (+{(uk_accuracy-us_accuracy)*100:.2f}%)")
    else:
        print(f"\n✅ US Model tested: {us_accuracy:.4f} ({us_accuracy*100:.2f}%)")
        print(f"   UK model test pending (needs format adaptation)")
    
    print()
    
    # Save results
    results_file = "fair_test_results.txt"
    with open(results_file, 'w') as f:
        f.write("FAIR MODEL COMPARISON\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"Test Data: {len(fresh_with_features)} bars (never seen during training)\n\n")
        f.write(f"US Model: {us_accuracy:.4f}\n")
        if uk_accuracy:
            f.write(f"UK Model: {uk_accuracy:.4f}\n")
    
    print(f"💾 Results saved: {results_file}")

if __name__ == "__main__":
    main()

