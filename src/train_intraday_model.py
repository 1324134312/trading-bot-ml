"""
Train Model with Intraday Features

Retrain the model specifically for 5-minute trading with new intraday features.
"""
import sys
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.add_intraday_features import IntradayFeatureExtractor
from src.feature_engineering.feature_extractor import FeatureExtractor

def collect_training_data():
    """Collect 5-minute data for training"""
    
    print("=" * 60)
    print("📊 COLLECTING 5-MINUTE TRAINING DATA")
    print("=" * 60)
    
    # Import here to avoid circular import
    from src.get_today_5min_data import get_today_5min_data
    
    # Get multiple days of data (limited by API, so we'll use what we have)
    symbols = ["AAPL", "MSFT", "TSLA", "GOOGL"]
    
    all_data = []
    
    for symbol in symbols:
        print(f"\n📈 Fetching {symbol}...")
        data = get_today_5min_data(symbol)
        
        if data is not None and not data.empty:
            # Add Ticker column
            data['Ticker'] = symbol
            all_data.append(data)
            print(f"   ✅ Got {len(data)} bars")
    
    if not all_data:
        print("❌ No data collected")
        return None
    
    # Combine all data
    df_all = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Total bars collected: {len(df_all)}")
    
    return df_all

def process_data_with_intraday_features(df):
    """Process data and add intraday features"""
    
    print("\n🔧 Processing data with intraday features...")
    
    # Rename columns
    df = df.rename(columns={
        'datetime': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by Date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add technical indicators
    print("   Adding technical indicators...")
    feature_extractor = FeatureExtractor()
    df = feature_extractor.add_all_features(df)
    
    # Add intraday features
    print("   Adding intraday features...")
    intraday_extractor = IntradayFeatureExtractor()
    df = intraday_extractor.add_intraday_features(df)
    
    print(f"✅ Total features: {len(df.columns)}")
    
    return df

def create_targets(df):
    """Create targets based on future price movement"""
    
    print("\n🎯 Creating targets...")
    
    # Calculate next bar's direction
    df['Next_Close'] = df.groupby('Ticker')['Close'].shift(-1)
    
    # Target: Will price go up in next 5 minutes?
    df['Target_Direction'] = (df['Next_Close'] > df['Close']).astype(int)
    
    # Remove last row for each ticker (no next bar)
    df = df.dropna(subset=['Next_Close'])
    
    print(f"✅ Created targets: {df['Target_Direction'].sum()}/{len(df)} UP signals ({df['Target_Direction'].mean()*100:.1f}%)")
    
    return df

def train_model(df):
    """Train XGBoost model"""
    
    print("\n🤖 Training model...")
    
    # Get feature columns (exclude non-numeric and target)
    exclude_cols = ['Date', 'Ticker', 'Datetime', 'Next_Close', 'Target_Direction']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Only numeric columns
    feature_cols = [col for col in feature_cols if df[col].dtype in ['float64', 'int64', 'float32', 'int32']]
    
    print(f"   Using {len(feature_cols)} features")
    
    # Prepare data
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['Target_Direction']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"   Train set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # Scale features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("   Training XGBoost...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    test_acc = model.score(X_test_scaled, y_test)
    
    print(f"\n✅ Model trained!")
    print(f"   Train accuracy: {train_acc*100:.1f}%")
    print(f"   Test accuracy: {test_acc*100:.1f}%")
    
    # Save model
    print("\n💾 Saving model...")
    os.makedirs("models", exist_ok=True)
    
    with open("models/intraday_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    
    with open("models/intraday_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    with open("models/intraday_features.pkl", 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print("✅ Model saved to models/intraday_model.pkl")
    
    return model, scaler, feature_cols

def main():
    """Main training function"""
    
    # Collect data
    df = collect_training_data()
    
    if df is None:
        return
    
    # Process with features
    df = process_data_with_intraday_features(df)
    
    # Create targets
    df = create_targets(df)
    
    # Train model
    model, scaler, features = train_model(df)
    
    print("\n" + "=" * 60)
    print("✅ INTRADAY MODEL TRAINING COMPLETE!")
    print("=" * 60)
    
    print("\n📊 Summary:")
    print(f"   Model: models/intraday_model.pkl")
    print(f"   Features: {len(features)}")
    print(f"   Train accuracy: {model.score(scaler.transform(df[[col for col in features if col in df.columns]].fillna(0)), df['Target_Direction'])*100:.1f}%")
    
    print("\n🚀 Next step: Test the model on live data!")

if __name__ == "__main__":
    main()
