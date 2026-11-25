"""
Train individual models for each stock and compare performance
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.models.trainer import ModelTrainer
from config.settings import PROCESSED_DATA_DIR
import pickle
from sklearn.model_selection import train_test_split

def train_stock_specific_models():
    """Train separate model for each stock"""
    
    print("=" * 70)
    print("📊 Training Stock-Specific Models")
    print("=" * 70)
    print("\nBuilding individual model for each stock...")
    print("This allows each model to learn stock-specific patterns\n")
    
    # List of stocks to train
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
    
    results = []
    models = {}
    
    for ticker in tickers:
        print(f"\n{'='*70}")
        print(f"Training {ticker}...")
        print(f"{'='*70}")
        
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.csv")
        
        if not os.path.exists(filepath):
            print(f"⚠️  {ticker}: File not found")
            continue
        
        # Load data
        data = pd.read_csv(filepath)
        data['Date'] = pd.to_datetime(data['Date'], utc=True)
        data = data.sort_values('Date')
        
        print(f"📊 Loaded {len(data)} rows for {ticker}")
        
        # Initialize trainer
        trainer = ModelTrainer(model_type='xgb')
        
        # Prepare data - time series split (last 30% for test)
        data_clean = data.dropna()
        split_idx = int(len(data_clean) * 0.7)
        
        X_train = data_clean.iloc[:split_idx]
        X_test = data_clean.iloc[split_idx:]
        
        y_train = X_train['Target_Direction'].values
        y_test = X_test['Target_Direction'].values
        
        # Prepare features
        exclude_cols = ['Date', 'Ticker', 'FetchDate', 'Dividends', 
                      'Stock Splits', 'Target', 'Target_Direction']
        feature_cols = [col for col in data.columns if col not in exclude_cols]
        
        X_train_features = X_train[feature_cols].values
        X_test_features = X_test[feature_cols].values
        
        # Split training into train/val
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_features, y_train, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = trainer.scaler.fit_transform(X_train_split)
        X_val_scaled = trainer.scaler.transform(X_val)
        X_test_scaled = trainer.scaler.transform(X_test_features)
        
        print(f"   Training: {len(X_train_split)} samples")
        print(f"   Validation: {len(X_val)} samples")
        print(f"   Test: {len(X_test)} samples")
        
        # Train model
        val_metrics = trainer.train(
            X_train_scaled, y_train_split,
            X_val_scaled, y_val
        )
        
        # Test on unseen data
        test_pred = trainer.predict(X_test_scaled)
        test_accuracy = np.mean(test_pred == y_test)
        
        # Calculate metrics
        buy_signals = np.sum(test_pred == 1)
        sell_signals = np.sum(test_pred == 0)
        
        results.append({
            'Ticker': ticker,
            'Training_Samples': len(X_train_split),
            'Validation_Accuracy': val_metrics['accuracy'],
            'Test_Accuracy': test_accuracy,
            'Buy_Signals': buy_signals,
            'Sell_Signals': sell_signals,
        })
        
        # Save model
        model_dir = f'src/models/trained_models/stock_specific'
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f'{ticker}_model.pkl')
        trainer.save_model(model_path)
        
        models[ticker] = trainer
        
        print(f"✅ {ticker} Results:")
        print(f"   Validation: {val_metrics['accuracy']:.4f}")
        print(f"   Test: {test_accuracy:.4f}")
        print(f"   Saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY - Stock-Specific vs Combined Model")
    print("=" * 70)
    
    results_df = pd.DataFrame(results)
    print("\nStock-Specific Models Performance:")
    print(results_df.to_string(index=False))
    
    print(f"\nAverage Test Accuracy: {results_df['Test_Accuracy'].mean():.2%}")
    print(f"Best: {results_df['Test_Accuracy'].max():.2%} ({results_df.loc[results_df['Test_Accuracy'].idxmax(), 'Ticker']})")
    print(f"Worst: {results_df['Test_Accuracy'].min():.2%} ({results_df.loc[results_df['Test_Accuracy'].idxmin(), 'Ticker']})")
    
    # Compare with combined model
    print("\n" + "=" * 70)
    print("📊 COMPARISON")
    print("=" * 70)
    print("\nCombined Model Performance (from previous test):")
    print("  Overall Average: 57.55%")
    print("  ETFs: 76-77%")
    print("  Individual Stocks: 49-53%")
    
    print("\nStock-Specific Model Performance:")
    print(f"  Average: {results_df['Test_Accuracy'].mean():.2%}")
    
    improvement = results_df['Test_Accuracy'].mean() - 0.5755
    if improvement > 0:
        print(f"\n✅ Improvement: +{improvement*100:.2f} percentage points")
    else:
        print(f"\n⚠️  No improvement: {improvement*100:.2f} percentage points")
    
    # Analyze by stock type
    results_df['Type'] = results_df['Ticker'].apply(
        lambda x: 'ETF' if x in ['SPY', 'QQQ'] else 'Individual Stock'
    )
    
    print("\n" + "=" * 70)
    print("📈 Performance by Stock Type")
    print("=" * 70)
    for stock_type in ['ETF', 'Individual Stock']:
        subset = results_df[results_df['Type'] == stock_type]
        print(f"\n{stock_type}:")
        print(f"  Average: {subset['Test_Accuracy'].mean():.2%}")
        print(f"  Best: {subset['Test_Accuracy'].max():.2%}")
        print(f"  Worst: {subset['Test_Accuracy'].min():.2%}")
    
    return results_df, models

if __name__ == "__main__":
    train_stock_specific_models()

