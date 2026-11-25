"""
Compare US Model vs UK Model on UK Stocks

This will determine which is superior!
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

def load_uk_data():
    """Load UK test data"""
    
    print("=" * 90)
    print("📂 Loading UK Test Data")
    print("=" * 90)
    
    files = glob.glob("data/uk/processed_5min/*_features.csv")
    
    all_data = []
    for file in tqdm(files, desc="Loading UK stocks"):
        df = pd.read_csv(file)
        ticker = os.path.basename(file).replace('_features.csv', '')
        df['Ticker'] = ticker
        all_data.append(df)
    
    data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Loaded {len(data)} bars from {len(files)} UK stocks")
    
    return data

def test_model(model, scaler, feature_cols, test_data):
    """Test a model on UK stocks"""
    
    # Get common features
    common_features = [f for f in feature_cols if f in test_data.columns]
    missing_features = [f for f in feature_cols if f not in test_data.columns]
    
    print(f"\n📊 Feature matching:")
    print(f"   Common: {len(common_features)}")
    print(f"   Missing: {len(missing_features)}")
    
    # Prepare data
    X_test = test_data[common_features].copy()
    
    # Add missing features as zeros
    for feat in missing_features:
        X_test[feat] = 0
    
    # Reorder to match training features
    X_test = X_test[feature_cols]
    
    # Clean
    X_test = X_test.replace([float('inf'), float('-inf')], float('nan'))
    valid_idx = X_test.notna().all(axis=1)
    X_test_clean = X_test[valid_idx]
    
    if 'Target_15min' in test_data.columns:
        y_test = test_data.loc[valid_idx, 'Target_15min'].values
    else:
        print("⚠️  No target column found")
        return None
    
    # Scale
    X_test_scaled = scaler.transform(X_test_clean.values)
    
    # Predict
    predictions = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(y_test)
    }

def main():
    """Compare models"""
    
    print("=" * 90)
    print("🔬 MODEL COMPARISON: US vs UK")
    print("=" * 90)
    print()
    print("Testing both models on UK stocks to see which is superior")
    print()
    
    # Load UK test data
    uk_data = load_uk_data()
    
    # Load UK model
    print("\n" + "=" * 90)
    print("📂 Loading UK Model (Trained on UK data)")
    print("=" * 90)
    
    uk_model_path = "src/models/trained_models/uk_5min_day_trading_model.pkl"
    
    if os.path.exists(uk_model_path):
        # For UK model, we need to adapt
        from src.models.predictor import ModelPredictor
        uk_predictor = ModelPredictor(uk_model_path)
        
        print("✅ UK model loaded")
        
        # Test UK model on UK data
        uk_results = test_model(us_model, us_scaler, us_features, uk_data)
        
        # For comparison, just note that UK model was 91.70% from training
        print(f"\n📊 UK Model (from training):")
        print(f"   Accuracy: 0.9170 (91.70%)")
    
    # Load US model
    print("\n" + "=" * 90)
    print("📂 Loading US Model (Trained on US data)")
    print("=" * 90)
    
    us_model_path = "models/us_max_model.pkl"
    us_scaler_path = "models/us_max_scaler.pkl"
    us_features_path = "models/us_max_features.pkl"
    
    if os.path.exists(us_model_path):
        with open(us_model_path, 'rb') as f:
            us_model = pickle.load(f)
        with open(us_scaler_path, 'rb') as f:
            us_scaler = pickle.load(f)
        with open(us_features_path, 'rb') as f:
            us_features = pickle.load(f)
        
        print("✅ US model loaded")
        
        # Test US model on UK data
        us_results = test_model(us_model, us_scaler, us_features, uk_data)
        
        if us_results:
            print(f"\n📊 US Model Results on UK Stocks:")
            print(f"   Accuracy: {us_results['accuracy']:.4f}")
            print(f"   Precision: {us_results['precision']:.4f}")
            print(f"   Recall: {us_results['recall']:.4f}")
            print(f"   F1-Score: {us_results['f1']:.4f}")
            print(f"   Samples: {us_results['samples']:,}")
    
    # Comparison
    print("\n" + "=" * 90)
    print("🏆 COMPARISON RESULTS")
    print("=" * 90)
    
    if 'uk_accuracy' in locals() and 'us_results' in locals():
        print(f"\n📊 UK Model (Trained on UK, Tested on UK):")
        print(f"   Accuracy: {uk_accuracy:.2%}")
        
        print(f"\n📊 US Model (Trained on US, Tested on UK):")
        print(f"   Accuracy: {us_results['accuracy']:.2%}")
        
        print(f"\n🎯 WINNER:")
        if us_results['accuracy'] > uk_accuracy:
            print(f"   ✅ US MODEL is superior!")
            print(f"   +{(us_results['accuracy'] - uk_accuracy)*100:.2f}% better")
        else:
            print(f"   ✅ UK MODEL is superior!")
            print(f"   +{(uk_accuracy - us_results['accuracy'])*100:.2f}% better")
        
        improvement = ((us_results['accuracy'] - uk_accuracy) / uk_accuracy) * 100
        print(f"\n📈 Improvement: {improvement:+.1f}%")
    
    print("\n" + "=" * 90)

if __name__ == "__main__":
    main()

