"""
Test the trained model's performance on different stocks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.models.predictor import ModelPredictor
from config.settings import PROCESSED_DATA_DIR

def test_model_performance():
    """Test model performance on individual stocks"""
    
    print("=" * 60)
    print("🧪 Testing Model Performance")
    print("=" * 60)
    print()
    
    # Load trained model
    model_path = 'src/models/trained_models/multi_ticker_xgb_model.pkl'
    predictor = ModelPredictor(model_path)
    
    # Test on each ticker
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'SPY', 'QQQ']
    
    results = []
    
    for ticker in tickers:
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.csv")
        
        if not os.path.exists(filepath):
            print(f"⚠️  {ticker}: File not found")
            continue
        
        # Load data
        data = pd.read_csv(filepath)
        
        # Make predictions
        predictions = predictor.predict_direction(data.copy())
        
        # Calculate accuracy
        if 'Target_Direction' in predictions.columns:
            correct = (predictions['Prediction'] == predictions['Target_Direction']).sum()
            total = len(predictions)
            accuracy = correct / total * 100
            
            # Count buy/sell signals
            buy_signals = len(predictor.get_buy_signals(predictions, min_confidence=0.6))
            sell_signals = len(predictor.get_sell_signals(predictions, min_confidence=0.6))
            
            avg_confidence = predictions['Confidence'].mean()
            
            results.append({
                'Ticker': ticker,
                'Accuracy': accuracy,
                'Buy Signals': buy_signals,
                'Sell Signals': sell_signals,
                'Avg Confidence': avg_confidence,
                'Total': total
            })
            
            print(f"✅ {ticker}:")
            print(f"   Accuracy: {accuracy:.2f}%")
            print(f"   Buy signals: {buy_signals}")
            print(f"   Sell signals: {sell_signals}")
            print(f"   Avg confidence: {avg_confidence:.2%}")
            print()
    
    # Summary
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print("=" * 60)
        print("📊 Summary")
        print("=" * 60)
        print(results_df.to_string(index=False))
        print()
        print(f"Average Accuracy: {results_df['Accuracy'].mean():.2f}%")
        print(f"Best Performer: {results_df.loc[results_df['Accuracy'].idxmax(), 'Ticker']} ({results_df['Accuracy'].max():.2f}%)")
    
    return results_df

def analyze_predictions_by_confidence():
    """Analyze prediction quality by confidence level"""
    
    print("=" * 60)
    print("🔍 Analyzing Predictions by Confidence Level")
    print("=" * 60)
    print()
    
    # Load model
    model_path = 'src/models/trained_models/multi_ticker_xgb_model.pkl'
    predictor = ModelPredictor(model_path)
    
    # Test on SPY
    data = pd.read_csv('data/processed/SPY_features.csv')
    predictions = predictor.predict_direction(data.copy())
    
    if 'Target_Direction' not in predictions.columns:
        print("⚠️  No target column found")
        return
    
    # Bin predictions by confidence
    confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for i in range(len(confidence_bins) - 1):
        lower = confidence_bins[i]
        upper = confidence_bins[i + 1]
        
        mask = (predictions['Confidence'] >= lower) & (predictions['Confidence'] < upper)
        subset = predictions[mask]
        
        if len(subset) > 0:
            correct = (subset['Prediction'] == subset['Target_Direction']).sum()
            accuracy = correct / len(subset) * 100
            
            print(f"Confidence {lower:.1f}-{upper:.1f}:")
            print(f"   Predictions: {len(subset)}")
            print(f"   Accuracy:    {accuracy:.2f}%")
            print()

def show_recent_signals():
    """Show recent buy/sell signals"""
    
    print("=" * 60)
    print("📈 Recent Trading Signals")
    print("=" * 60)
    print()
    
    # Load model
    model_path = 'src/models/trained_models/multi_ticker_xgb_model.pkl'
    predictor = ModelPredictor(model_path)
    
    tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in tickers:
        filepath = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_features.csv")
        
        if not os.path.exists(filepath):
            continue
        
        data = pd.read_csv(filepath)
        predictions = predictor.predict_direction(data.copy())
        
        # Get recent high-confidence signals
        buy_signals = predictor.get_buy_signals(predictions, min_confidence=0.7)
        sell_signals = predictor.get_sell_signals(predictions, min_confidence=0.7)
        
        print(f"\n📊 {ticker}:")
        
        if len(buy_signals) > 0:
            print(f"   🟢 Recent BUY signals (last 3):")
            recent_buys = buy_signals[['Date', 'Close', 'Prediction_Prob', 'Confidence']].tail(3)
            for _, row in recent_buys.iterrows():
                print(f"      {row['Date'][:10]} | ${row['Close']:.2f} | Prob: {row['Prediction_Prob']:.2%} | Conf: {row['Confidence']:.2%}")
        
        if len(sell_signals) > 0:
            print(f"   🔴 Recent SELL signals (last 3):")
            recent_sells = sell_signals[['Date', 'Close', 'Prediction_Prob', 'Confidence']].tail(3)
            for _, row in recent_sells.iterrows():
                print(f"      {row['Date'][:10]} | ${row['Close']:.2f} | Prob: {row['Prediction_Prob']:.2%} | Conf: {row['Confidence']:.2%}")
        
        if len(buy_signals) == 0 and len(sell_signals) == 0:
            print("   ⚪ No high-confidence signals (>70%)")

def main():
    """Run all tests"""
    results = test_model_performance()
    print()
    analyze_predictions_by_confidence()
    print()
    show_recent_signals()

if __name__ == "__main__":
    main()

