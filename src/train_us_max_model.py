"""
Train Model on US Max Data (897k+ bars)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from tqdm import tqdm

def load_us_data():
    """Load all US processed data"""
    
    print("=" * 90)
    print("📂 Loading US Max Data")
    print("=" * 90)
    
    data_dir = "data/us/max_processed"
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    print(f"\n📁 Found {len(files)} stocks")
    
    all_data = []
    
    for file in tqdm(files, desc="Loading stocks"):
        try:
            df = pd.read_csv(file)
            ticker = os.path.basename(file).replace('_features.csv', '')
            df['Ticker'] = ticker
            all_data.append(df)
        except:
            pass
    
    data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Loaded {len(data)} bars total")
    print(f"📅 Date range: {data['Datetime'].min()} to {data['Datetime'].max()}")
    print(f"📈 Unique stocks: {data['Ticker'].nunique()}")
    
    return data

def main():
    """Train US model"""
    
    # Load data
    data = load_us_data()
    
    # Prepare features
    exclude_cols = ['Datetime', 'Date', 'Ticker', 'Target_15min', 'FetchDate']
    feature_cols = [col for col in data.columns if col not in exclude_cols]
    
    print(f"\n🔧 Features: {len(feature_cols)}")
    
    # Clean data
    data_clean = data.dropna()
    print(f"📊 Clean data: {len(data_clean)} bars")
    
    # Prepare
    X = data_clean[feature_cols].values
    y = data_clean['Target_15min'].values
    
    print(f"\n📊 Target Distribution:")
    print(f"   Buy signals (1): {y.sum():,} ({y.mean()*100:.1f}%)")
    print(f"   Hold (0): {(len(y) - y.sum()):,} ({(1-y.mean())*100:.1f}%)")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Further split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train):,} bars")
    print(f"   Validation: {len(X_val):,} bars")
    print(f"   Test: {len(X_test):,} bars")
    
    # Scale
    print("\n🔧 Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    print("\n🤖 Training XGBoost model...")
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(
        X_train_scaled,
        y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )
    
    # Evaluate
    print("\n📊 Evaluating...")
    
    train_pred = model.predict(X_train_scaled)
    val_pred = model.predict(X_val_scaled)
    test_pred = model.predict(X_test_scaled)
    
    print(f"\n📊 Results:")
    print(f"   Train Accuracy: {accuracy_score(y_train, train_pred):.4f}")
    print(f"   Val Accuracy:   {accuracy_score(y_val, val_pred):.4f}")
    print(f"   Test Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    
    print(f"\n📊 Test Set Detailed:")
    precision = precision_score(y_test, test_pred)
    recall = recall_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)
    
    print(f"   Accuracy:  {accuracy_score(y_test, test_pred):.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n🔍 Top 10 Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/us_max_model.pkl"
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    scaler_path = "models/us_max_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    feature_names_path = "models/us_max_features.pkl"
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"\n💾 Model saved to: {model_path}")
    print(f"💾 Scaler saved to: {scaler_path}")
    print(f"💾 Features saved to: {feature_names_path}")
    
    print("\n" + "=" * 90)
    print("✅ US MAX MODEL TRAINING COMPLETE!")
    print("=" * 90)
    print(f"\n🎯 Test Accuracy: {accuracy_score(y_test, test_pred):.4f}")
    print(f"💾 Model: {model_path}")

if __name__ == "__main__":
    main()

