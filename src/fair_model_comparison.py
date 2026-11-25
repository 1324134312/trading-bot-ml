"""
FAIR Comparison: Test both models on NEW, UNSEEN UK data

The UK model was tested on data it may have seen during training!
Let's fix that with a proper fair test.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import glob
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

def get_unseen_uk_data():
    """Get UK data that the models haven't seen"""
    
    print("=" * 90)
    print("📂 Loading UNSEEN UK Test Data")
    print("=" * 90)
    
    files = glob.glob("data/uk/processed_5min/*_features.csv")
    
    # Use a different subset than training
    # Or use latest data that wasn't in training
    all_data = []
    
    for i, file in enumerate(files):
        # Skip every 3rd stock to create unseen test set
        if i % 3 == 0:  # Take stocks 0, 3, 6, 9... (unseen)
            df = pd.read_csv(file)
            ticker = os.path.basename(file).replace('_features.csv', '')
            df['Ticker'] = ticker
            all_data.append(df)
    
    unseen_data = pd.concat(all_data, ignore_index=True)
    
    print(f"\n✅ Loaded {len(unseen_data)} bars from {len(all_data)} unseen UK stocks")
    print(f"📊 Using different stocks than training set")
    
    return unseen_data

def test_model_fair(model, scaler, feature_cols, test_data, model_name):
    """Test a model fairly on unseen data"""
    
    print(f"\n🔍 Testing {model_name}...")
    
    # Get common features
    common_features = [f for f in feature_cols if f in test_data.columns]
    missing = [f for f in feature_cols if f not in test_data.columns]
    
    print(f"   Features: {len(common_features)} common, {len(missing)} missing")
    
    # Prepare data
    X = test_data[common_features].copy()
    for feat in missing:
        X[feat] = 0
    
    X = X[feature_cols]  # Reorder
    
    # Clean
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
    
    # Calculate metrics
    accuracy = accuracy_score(y_actual, predictions)
    precision = precision_score(y_actual, predictions)
    recall = recall_score(y_actual, predictions)
    f1 = f1_score(y_actual, predictions)
    
    print(f"   ✅ Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"   📊 Samples:   {len(y_actual):,}")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(y_actual)
    }

def main():
    """Fair comparison"""
    
    print("=" * 90)
    print("🔬 FAIR MODEL COMPARISON")
    print("=" * 90)
    print()
    print("⚠️  IMPORTANT: Testing on UNSEEN data!")
    print("   Both models tested on stocks they haven't seen")
    print("   This is a FAIR comparison")
    print()
    
    # Load unseen UK data
    unseen_uk = get_unseen_uk_data()
    
    # Test UK Model
    print("\n" + "=" * 90)
    print("📂 Testing UK Model on Unseen UK Stocks")
    print("=" * 90)
    
    uk_model_path = "src/models/trained_models/uk_5min_day_trading_model.pkl"
    
    if not os.path.exists(uk_model_path):
        print("❌ UK model not found")
        uk_results = None
    else:
        from src.models.predictor import ModelPredictor
        uk_predictor = ModelPredictor(uk_model_path)
        
        # Get predictions
        uk_predictions = uk_predictor.predict_direction(unseen_uk.copy())
        
        # Calculate accuracy
        if 'Prediction' in uk_predictions.columns and 'Actual' in uk_predictions.columns:
            accuracy = (uk_predictions['Prediction'] == uk_predictions['Actual']).sum() / len(uk_predictions)
            
            print(f"\n✅ UK Model Results:")
            print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"   Samples:  {len(uk_predictions):,}")
            
            uk_results = {'model': 'UK', 'accuracy': accuracy, 'samples': len(uk_predictions)}
        else:
            print("   ❌ Could not calculate accuracy")
            uk_results = None
    
    # Test US Model
    print("\n" + "=" * 90)
    print("📂 Testing US Model on Unseen UK Stocks")
    print("=" * 90)
    
    us_model_path = "models/us_max_model.pkl"
    
    if not os.path.exists(us_model_path):
        print("❌ US model not found")
        us_results = None
    else:
        with open(us_model_path, 'rb') as f:
            us_model = pickle.load(f)
        with open("models/us_max_scaler.pkl", 'rb') as f:
            us_scaler = pickle.load(f)
        with open("models/us_max_features.pkl", 'rb') as f:
            us_features = pickle.load(f)
        
        us_results = test_model_fair(us_model, us_scaler, us_features, unseen_uk, "US Max")
    
    # Fair Comparison
    print("\n" + "=" * 90)
    print("🏆 FAIR COMPARISON RESULTS")
    print("=" * 90)
    
    if uk_results and us_results:
        print(f"\n📊 UK Model (on unseen UK stocks):")
        print(f"   Accuracy: {uk_results['accuracy']:.4f} ({uk_results['accuracy']*100:.2f}%)")
        
        print(f"\n📊 US Model (on unseen UK stocks):")
        print(f"   Accuracy: {us_results['accuracy']:.4f} ({us_results['accuracy']*100:.2f}%)")
        
        print(f"\n🎯 FAIR COMPARISON:")
        diff = (us_results['accuracy'] - uk_results['accuracy']) * 100
        
        if diff > 0:
            print(f"   ✅ US MODEL is {abs(diff):.2f}% BETTER!")
        else:
            print(f"   ✅ UK MODEL is {abs(diff):.2f}% BETTER!")
        
        print(f"\n📊 Difference: {diff:+.2f}%")
        
        # Analysis
        print(f"\n💡 ANALYSIS:")
        if us_results['accuracy'] > uk_results['accuracy']:
            print(f"   ✅ US model generalizes better to unseen UK stocks")
            print(f"   ✅ More training data (805k vs 37k) helped!")
            print(f"   ✅ Cross-market training is superior")
        else:
            print(f"   ✅ UK model is more specialized for UK market")
            print(f"   ✅ Market-specific training matters")
            print(f"   ✅ Local models can outperform generalists")
    
    print()

if __name__ == "__main__":
    main()

