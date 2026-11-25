"""
Test: Can a model trained on US data work on UK stocks?

This tests the hypothesis that US and UK markets have similar patterns.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
from src.feature_engineering.indicators import TechnicalIndicators
from src.models.predictor import ModelPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

def prepare_us_data():
    """Load and prepare US data"""
    print("=" * 70)
    print("📊 Loading US Data")
    print("=" * 70)
    
    files = glob.glob("data/us/intraday/*_5m.csv")
    
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        df['Ticker'] = os.path.basename(file).replace('_5m.csv', '')
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Loaded {len(data)} bars from {len(files)} US stocks")
    print(f"📅 Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    
    return data

def prepare_uk_data():
    """Load and prepare UK data"""
    print("\n" + "=" * 70)
    print("📊 Loading UK Data")
    print("=" * 70)
    
    files = glob.glob("data/uk/processed_5min/*_features.csv")
    
    all_data = []
    for file in files:
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('_features.csv', '')
        df['Ticker'] = ticker
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    print(f"✅ Loaded {len(data)} bars from {len(files)} UK stocks")
    
    return data

def add_features_us(data):
    """Add features to US data (similar to UK)"""
    print("\n🔧 Adding features to US data...")
    
    indicators = TechnicalIndicators()
    
    df = data.copy()
    
    # Add basic features (similar to UK 5-min features)
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
    
    # Volume
    df['Volume_SMA'] = df['Volume'].rolling(15).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Time features
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Is_Morning'] = ((df['Hour'] >= 9) & (df['Hour'] <= 12)).astype(int)  # US hours
    df['Is_Afternoon'] = ((df['Hour'] >= 13) & (df['Hour'] <= 16)).astype(int)
    
    # Returns
    df['Return_5min'] = df['Close'].pct_change(1)
    df['Return_30min'] = df['Close'].pct_change(6)
    
    # Target (price up in 3 bars = 15 minutes)
    future_price = df['Close'].shift(-3)
    future_return = (future_price / df['Close']) - 1
    df['Target'] = (future_return > 0.003).astype(int)
    
    # Clean
    df = df.replace([float('inf'), float('-inf')], float('nan'))
    df = df.dropna()
    
    print(f"✅ Added features: {len(df)} valid bars")
    
    return df

def train_us_model(us_data):
    """Train model on US data"""
    print("\n" + "=" * 70)
    print("🤖 Training Model on US Data")
    print("=" * 70)
    
    # Prepare features
    feature_cols = [col for col in us_data.columns 
                   if col not in ['Ticker', 'Datetime', 'Target', 'Date', 'Datetime']]
    
    X = us_data[feature_cols].values
    y = us_data['Target'].values
    
    # Split
    train_idx = int(len(X) * 0.8)
    X_train, X_test = X[:train_idx], X[train_idx:]
    y_train, y_test = y[:train_idx], y[train_idx:]
    
    # Train
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Test on US data
    y_pred = model.predict(X_test)
    us_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"📊 US Data Accuracy: {us_accuracy:.2%}")
    
    return model, feature_cols

def test_on_uk(model, feature_cols):
    """Test US-trained model on UK stocks"""
    print("\n" + "=" * 70)
    print("🧪 Testing US Model on UK Stocks")
    print("=" * 70)
    
    uk_data = prepare_uk_data()
    
    # Match features
    uk_feature_cols = [col for col in uk_data.columns 
                      if col in feature_cols and col != 'Target_15min']
    
    # Get common features
    common_features = [f for f in feature_cols if f in uk_data.columns]
    
    print(f"📊 Common features: {len(common_features)}")
    
    # Filter UK data to only columns we have
    uk_filtered = uk_data[common_features].copy()
    
    # Add missing columns with zeros
    for col in feature_cols:
        if col not in uk_filtered.columns:
            uk_filtered[col] = 0
    
    uk_filtered = uk_filtered[feature_cols]  # Same order
    
    # Clean
    uk_filtered = uk_filtered.replace([float('inf'), float('-inf')], float('nan'))
    uk_filtered = uk_filtered.dropna()
    
    # Get actual targets
    if 'Target_15min' in uk_data.columns:
        actual_targets = uk_data.loc[uk_filtered.index, 'Target_15min'].values
    else:
        print("⚠️  No target column found in UK data")
        return
    
    # Predict
    X_uk = uk_filtered.values
    predictions = model.predict(X_uk)
    
    # Calculate accuracy
    accuracy = accuracy_score(actual_targets, predictions)
    
    print(f"\n📊 CROSS-MARKET RESULTS:")
    print(f"   US-trained model accuracy on UK stocks: {accuracy:.2%}")
    
    if accuracy > 0.60:
        print(f"   ✅ GOOD - Patterns are similar!")
    elif accuracy > 0.55:
        print(f"   ⚠️  OKAY - Some patterns transfer")
    else:
        print(f"   ❌ POOR - Markets are different")
    
    return accuracy

def main():
    """Main function"""
    print("=" * 70)
    print("🧪 CROSS-MARKET HYPOTHESIS TEST")
    print("=" * 70)
    print()
    print("Hypothesis: US and UK markets have similar patterns")
    print("Test: Train on US data, test on UK stocks")
    print()
    
    # 1. Prepare US data
    us_raw = prepare_us_data()
    us_data = add_features_us(us_raw)
    
    # 2. Train on US
    model, feature_cols = train_us_model(us_data)
    
    # 3. Test on UK
    accuracy = test_on_uk(model, feature_cols)
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE")
    print("=" * 70)
    
    if accuracy > 0.60:
        print("\n💡 CONCLUSION:")
        print("   ✅ US and UK markets ARE similar!")
        print("   ✅ You can train on US data (60 days)")
        print("   ✅ And trade on UK stocks (avoid PDT)")
        print("   ✅ Best of both worlds!")
    else:
        print("\n💡 CONCLUSION:")
        print("   ⚠️  Markets are different")
        print("   ✅ Train separate models for each")
        print("   ✅ Use US for testing, UK for trading")

if __name__ == "__main__":
    main()

